# Corpora ingestion receipt — schema `gccf.receipt/1`

The receipt is the artifact GCCF produces for the training-prep tool. It never
contains corpus data by default; it describes **which files to ingest, which
records inside them to drop, and how the reviewer tagged things**, all keyed by
path and locator so the consumer can address the original bytes directly.

Written by *Receipt → Write receipt…*. Pretty-printed JSON, UTF-8.

## Top level

| field | type | meaning |
|---|---|---|
| `schema` | string | `"gccf.receipt/1"`. Bump on breaking changes. |
| `generated_at` | RFC 3339 | when the receipt was written |
| `generator` | object | `{ "name": "gccf", "version": "0.1.0" }` |
| `note` | string \| null | free text entered in the UI |
| `policy` | object | the thresholds used to derive file verdicts (see below) plus a `locator_types` legend |
| `tags` | array | tag vocabulary: `{ name, builtin, sample_count }` |
| `roots` | array | corpora in scope: `{ id, path, scanned_at, file_count, total_bytes }` |
| `sessions` | array | sessions in scope with their parameters and status counts |
| `summary` | object | totals (always present) |
| `files` | array | present for verbosity `files` and `full`; absent for `summary` |

### `policy`

```json
{
  "verbosity": "files",
  "include_min_good_ratio": 0.8,
  "exclude_max_good_ratio": 0.5,
  "min_decided": 1,
  "skip_counts_as_bad": false,
  "include_classifier_decisions": true,
  "include_unsampled_files": false,
  "file_verdicts": ["include", "exclude", "review", "unsampled"],
  "locator_types": { "row": "...", "bytes": "...", "stream_lines": "..." }
}
```

File verdict derivation, per file, over the samples in scope:

```
decided    = good + bad            (+ skip if skip_counts_as_bad)
good_ratio = good / decided        (null if decided == 0)
verdict    = "include"  if decided >= min_decided and good_ratio >= include_min_good_ratio
           = "exclude"  if decided >= min_decided and good_ratio <= exclude_max_good_ratio
           = "review"   otherwise
           = "unsampled" for files listed via include_unsampled_files
```

With `include_classifier_decisions = false`, verdicts applied by a classifier
run are treated as pending; only human decisions count.

### `summary`

```json
{
  "files_include": 41, "files_exclude": 3, "files_review": 6, "files_unsampled": 0,
  "bytes_include": 91234567890, "bytes_exclude": 812345678, "bytes_review": 5123456789,
  "samples": { "samples": 2500, "good": 2210, "bad": 190, "skip": 40, "pending": 60, "human": 2380, "classifier": 60 },
  "exclusions": 190,
  "tags": { "english": 1900, "code": 400, "boilerplate": 120 }
}
```

### `files[]`

```json
{
  "root_id": 1,
  "path": "pile/shard-0007.jsonl.zst",
  "abs_path": "D:\\corpora\\pile\\shard-0007.jsonl.zst",
  "format": "jsonl",
  "kind": "rows",
  "compression": "zstd",
  "size_bytes": 734003200,
  "mtime": 1725400000,
  "row_count": 1204411,
  "verdict": "include",
  "good_ratio": 0.94,
  "counts": { "samples": 50, "good": 47, "bad": 3, "skip": 0, "pending": 0, "human": 50, "classifier": 0 },
  "tags": { "english": 44, "web": 6 },
  "exclusions": [ { "type": "stream_lines", "line_start": 48123, "n_lines": 1 }, ... ],
  "samples": [ ... ]               // verbosity "full" only
}
```

* `format`: `parquet | jsonl | csv | tsv | json | text | markdown | code | html | xml | unknown | binary`
* `kind`: `rows` (one sample = one record) or `lines` (one sample = a window of N lines)
* `compression`: `none | gzip | zstd | unsupported`
* `row_count`: exact for parquet; for compressed streams it is the number of
  lines seen when the whole file was read (absent if the read was capped).
* `exclusions`: locators of every sample marked **bad** (plus **skip** if
  `skip_counts_as_bad`). An ingester that honours these can keep an
  `include` file while dropping precisely the reviewed-bad records.

### `files[].samples[]` (verbosity `full`)

```json
{
  "id": 8123,
  "session_id": 3,
  "locator": { "type": "bytes", "byte_start": 104857600, "byte_end": 104861234, "n_lines": 50 },
  "verdict": "good",
  "source": "human",
  "tags": ["english", "academic"],
  "content_hash": "blake3 hex of the sample text",
  "decided_at": "2026-09-04T15:02:11.123Z",
  "text": "..."                    // only when include_text was set
}
```

## Locators

| `type` | fields | how to re-read the record |
|---|---|---|
| `row` | `index` | parquet absolute row index (0-based across all row groups) |
| `bytes` | `byte_start`, `byte_end`, `n_lines` | uncompressed file; seek to `byte_start`, read `byte_end - byte_start` bytes. Line-aligned; the range includes the trailing newline. For CSV/TSV the header line is **not** part of the range (it was prepended for display only). |
| `stream_lines` | `line_start`, `n_lines` | gzip / zstd stream; decompress from the start, skip `line_start` lines (0-based; CSV/TSV header counts as line 0), take `n_lines`. |

`content_hash` (BLAKE3 of the sample text as GCCF stored it) lets a consumer
verify that the file has not changed at that locator since review. Note that
sample text was normalised: CRLF → LF, trailing newline stripped, and cut at
`max_bytes_per_sample` when `truncated` is set.

## Training-data export (separate file)

*Classifier → Export JSONL* writes one object per line:

```json
{ "sample_id": 8123, "session_id": 3, "root": "D:\\corpora", "path": "pile/shard-0007.jsonl.zst",
  "format": "jsonl", "kind": "rows", "compression": "zstd",
  "locator": { ... }, "content_hash": "...",
  "verdict": "good", "label": 1, "source": "human", "decided_at": "...",
  "tags": ["english"], "text": "...", "text_bytes": 3401, "truncated": false }
```

`label` is `1` for good, `0` for bad, `null` for skipped / pending rows.

After an evaluation round each row also carries `eval` (or `null`):

```json
"eval": { "run_id": 7, "predicted": "bad", "score": 0.82, "predicted_tags": ["boilerplate"],
          "human": "good", "agree": false, "blind": true }
```

`agree: false` rows are the model's mistakes on unseen data; trainers should
up-weight them (both reference trainers do, `--feedback-weight`).
