# GCCF — Granular Corpora Control Funnel

Desktop tool (Rust + Tauri 2) for getting granular control over hundreds of
gigabytes of pretraining corpora **without ever copying or loading them**:

1. **Index** a corpus folder — a metadata-only walk builds a file catalogue
   (path, size, format, compression) in SQLite.
2. **Sample** — a session opens N random files and cuts M samples from each:
   one *record* per sample for parquet / jsonl / csv, a window of *L lines* for
   prose and code. I/O cost scales with the number of samples, not file size.
3. **Review** — swipe-style annotation. `−` bad, `=` good, *Skip* parks the
   sample in a queue (no hotkey by design). Tags via number keys or a free-form
   input; new tags join the vocabulary instantly.
4. **Edit** — *Reveal file* opens Explorer at the source file so fixes happen
   in your editor, not through the app.
5. **Receipt** — export a versioned JSON *ingestion receipt* (file verdicts,
   per-record exclusions, tags, locators) that the training-prep tool consumes.
6. **Classifier loop** — annotate → train → evaluate blind on real data →
   finetune on the disagreements → run once it clears the gate. Training
   happens in-app (built-in Rust model, or your own trainer command); up to 8
   workers score in parallel; evaluation is read-only and blind; a report
   shows accuracy, calibration, tag agreement and every disagreement.

Everything GCCF knows lives in one SQLite file. Corpora on disk are read-only
to it.

## Build & run

Prerequisites: Rust (MSVC toolchain), Node 20+, WebView2 (ships with Windows
11). Python 3 + scikit-learn only for the reference classifier scripts.

```powershell
npm install
npm run app:dev        # dev build with hot reload (opens the window)
npm run app:build      # release bundle → src-tauri/target/release/bundle
cd src-tauri; cargo test   # backend unit tests (sampling, scan, db, protocol)
```

The catalogue lives at `%APPDATA%\dev.cemery.gccf\gccf.sqlite3`. Set
`GCCF_DB=<path>` to use a different one (e.g. one per project).

## Workflow

**Corpora** → *Add corpus folder*. Scanning ~1M files takes a minute or two;
progress shows in the sidebar. The *Formats* table shows what is sampleable
(`rows` / `lines`) and what is not (`unsupported`: binaries, `.xz`/`.bz2`,
parquet inside gzip).

**Sessions** → choose corpus, optional format filter, *files to open*,
*samples per file*, *lines per sample*, seed. Files that already have samples
are skipped by default so successive sessions cover new ground. Same seed =
identical picks, so a session is reproducible.

**Review** → the card shows the sample, its file, locator and size, duplicate
warning (identical text seen elsewhere), and the latest classifier prediction.
JSON rows render as fields (`p` toggles raw). Every tag toggle is saved
immediately; the verdict snapshots the tags and the time spent looking.

| key | action |
|---|---|
| `−` | bad |
| `=` / `+` | good |
| `1`…`9` | toggle tag bound to that number |
| `Backspace` / `←` | previous sample (re-deciding overrides) |
| `→` | peek at next without deciding |
| `F1`…`F7` | switch screens |

**Queue** → skipped samples (and anything marked good/bad, for re-review).

**Receipt** → pick sessions, verbosity (`summary` / `files` / `full`) and the
include/exclude thresholds, preview the outcome, write the JSON. Schema:
[docs/receipt-schema.md](docs/receipt-schema.md).

**Classifier** → the screen walks the loop top to bottom:

1. **Annotate.** At least 20 good/bad decisions (more is better).
2. **Train.** *Built-in* trains a hashed word/char n-gram logistic regression
   in-process (no Python) with a stratified hold-out; *External* exports the
   JSONL and runs your command with `{data}` / `{model}` / `{parent_model}`
   substituted, streaming its output into the app. Both register a model.
3. **Evaluate.** Pick the model, target a session, *Start evaluation*. The
   model scores pending (unseen) samples read-only; you then review them in
   the normal card, blind by default — after each decision GCCF reveals what
   the model said. Your decisions are real annotations too. The report shows
   accuracy, precision/recall, calibration by confidence bucket, tag
   agreement, and a list of disagreements to open.
4. **Finetune.** *Finetune #N on this feedback* warm-starts from the model
   (same feature space) with evaluation disagreements up-weighted; external
   trainers get the parent path. Repeat 3 → 4 until the report clears the
   capability gate (Settings: min accuracy, min blind decisions).
5. **Run.** Score mode stores predictions as hints; *Apply to pending* above a
   confidence threshold turns them into classifier-sourced verdicts, never
   overriding a human one.

Reference scripts for the external path:

```powershell
py -3 -m pip install -r classifier/requirements.txt
# trainer command template (default in the UI):
#   py -3 classifier/train_classifier.py {data} --out {model} --parent {parent_model}
# worker command template:
#   py -3 classifier/worker.py --model {model}
```

Worker protocol (JSON lines, blank line terminates a batch each way):

```
→ {"id": 12, "text": "...", "path": "shard-1.jsonl.zst", "format": "jsonl"}
→
← {"id": 12, "verdict": "good", "score": 0.93, "tags": ["english"]}
←
```

Trainer contract: read `{data}`, write `{model}`, optionally print a final
`{"gccf_metrics": {...}}` line. Any language works. Each worker process gets
`GCCF_WORKER_INDEX` so it can pin a GPU. Models, runs, predictions and
evaluation feedback all live in the catalogue; model files go to
`<catalogue dir>/models/`.

## How sampling works

| container | method | locator |
|---|---|---|
| parquet | footer row count → seeded distinct row indices → arrow `RowSelection` (only touched pages are read); binary columns dropped | `row { index }` |
| jsonl / csv / tsv / text / code (uncompressed) | ≤ 8 MiB: exact windowing of the whole file. Larger: seeded random byte offsets, align to the next line start, read the window; overlapping windows rejected | `bytes { byte_start, byte_end, n_lines }` |
| `.gz` / `.zst` | streaming decompress + reservoir sampling of windows (optionally capped at N decompressed bytes) | `stream_lines { line_start, n_lines }` |
| `.json` | sniffed: newline-delimited objects → rows; otherwise treated as text lines | as above |

Samples are stored as text in the catalogue (default cap 256 KB each) with a
BLAKE3 hash, so review, export and receipts never re-open the corpus.

## Layout

```
src/                 frontend (vanilla TypeScript + Vite; views/ one file per screen)
src-tauri/src/
  formats.rs         extension → format / kind / compression, sniffers
  scan.rs            directory walk → file index
  sample.rs          sampling engines + locators
  session.rs         parallel session builder (rayon pool, streamed DB writes)
  db.rs              schema + queries (SQLite, WAL)
  receipt.rs         ingestion receipt builder
  export.rs          training-data JSONL export
  classifier/        worker pool, heuristic baseline, built-in trainable model,
                     trainers (in-process + external command), external-process backend
  jobs.rs            background job registry + progress events
  commands.rs        Tauri command surface
classifier/          reference Python worker + trainer
docs/                receipt schema
```
