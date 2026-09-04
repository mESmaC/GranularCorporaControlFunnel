//! External-process backend. Each worker spawns one child process and talks a
//! tiny JSON-lines protocol over stdin/stdout:
//!
//! ```text
//! -> {"id": 12, "text": "...", "path": "shard-1.jsonl", "format": "jsonl"}
//! -> {"id": 13, "text": "...", "path": "...", "format": "..."}
//! ->                                  (empty line = end of batch)
//! <- {"id": 12, "verdict": "good", "score": 0.93, "tags": ["english"]}
//! <- {"id": 13, "verdict": "bad",  "score": 0.71, "tags": []}
//! <-                                  (empty line = end of batch)
//! ```
//!
//! The child may answer out of order and may omit items it cannot score (they
//! are recorded as failed). Anything the child prints to stderr is logged.
//! GPU work happens inside the child — GCCF only schedules and records.

use super::{Backend, ClassifyItem, ClassifyResult};
use anyhow::{anyhow, Context};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

pub struct CommandBackend {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    label: String,
}

/// Split a command line the way a shell would for the simple cases we need:
/// whitespace-separated, with single or double quotes grouping.
pub fn split_command_line(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut quote: Option<char> = None;
    let mut has_token = false;
    for c in s.chars() {
        match quote {
            Some(q) => {
                if c == q {
                    quote = None;
                } else {
                    cur.push(c);
                }
            }
            None => {
                if c == '"' || c == '\'' {
                    quote = Some(c);
                    has_token = true;
                } else if c.is_whitespace() {
                    if has_token {
                        out.push(std::mem::take(&mut cur));
                        has_token = false;
                    }
                } else {
                    cur.push(c);
                    has_token = true;
                }
            }
        }
    }
    if has_token {
        out.push(cur);
    }
    out
}

impl CommandBackend {
    pub fn spawn(command: &str, args: &[String], cwd: Option<&str>, worker_index: usize) -> anyhow::Result<Self> {
        // Allow `command` to carry its own arguments ("py -3 worker.py --model m.joblib").
        let mut parts = split_command_line(command);
        if parts.is_empty() {
            return Err(anyhow!("empty classifier command"));
        }
        let program = parts.remove(0);
        parts.extend(args.iter().cloned());

        let mut cmd = Command::new(&program);
        cmd.args(&parts)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .env("GCCF_WORKER_INDEX", worker_index.to_string())
            .env("PYTHONUNBUFFERED", "1")
            .env("PYTHONIOENCODING", "utf-8");
        if let Some(dir) = cwd {
            if !dir.trim().is_empty() {
                cmd.current_dir(dir);
            }
        }
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            cmd.creation_flags(CREATE_NO_WINDOW);
        }
        let mut child = cmd.spawn().with_context(|| format!("spawning classifier worker `{program}`"))?;
        let stdin = child.stdin.take().ok_or_else(|| anyhow!("no stdin"))?;
        let stdout = child.stdout.take().ok_or_else(|| anyhow!("no stdout"))?;
        Ok(CommandBackend {
            child,
            stdin,
            stdout: BufReader::with_capacity(1 << 16, stdout),
            label: format!("{program} #{worker_index}"),
        })
    }
}

impl Backend for CommandBackend {
    fn classify(&mut self, batch: &[ClassifyItem]) -> anyhow::Result<Vec<ClassifyResult>> {
        // Send.
        {
            let mut buf = Vec::with_capacity(batch.len() * 1024);
            for item in batch {
                serde_json::to_writer(&mut buf, item)?;
                buf.push(b'\n');
            }
            buf.push(b'\n');
            self.stdin.write_all(&buf).with_context(|| format!("{}: writing batch (worker exited?)", self.label))?;
            self.stdin.flush()?;
        }
        // Receive until the blank terminator. The terminator is mandatory so
        // that a slow worker never desynchronises batches.
        let want: std::collections::HashSet<i64> = batch.iter().map(|i| i.id).collect();
        let mut results = Vec::with_capacity(batch.len());
        let mut line = String::new();
        loop {
            line.clear();
            let n = self.stdout.read_line(&mut line)?;
            if n == 0 {
                let status = self.child.try_wait().ok().flatten();
                return Err(anyhow!(
                    "{}: worker closed stdout after {} of {} results (exit: {:?})",
                    self.label,
                    results.len(),
                    batch.len(),
                    status
                ));
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                break;
            }
            match serde_json::from_str::<ClassifyResult>(trimmed) {
                Ok(r) => {
                    if want.contains(&r.id) {
                        results.push(r);
                    } else {
                        log::warn!("{}: result for unknown id {}", self.label, r.id);
                    }
                }
                Err(e) => {
                    // Tolerate chatter on stdout (e.g. progress prints) but log it.
                    log::warn!("{}: non-JSON line ignored ({e}): {}", self.label, trimmed.chars().take(200).collect::<String>());
                }
            }
        }
        Ok(results)
    }
}

impl Drop for CommandBackend {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_command_lines() {
        assert_eq!(split_command_line("py -3 worker.py"), vec!["py", "-3", "worker.py"]);
        assert_eq!(
            split_command_line("\"C:\\Program Files\\Python311\\python.exe\" worker.py --model 'my model.joblib'"),
            vec!["C:\\Program Files\\Python311\\python.exe", "worker.py", "--model", "my model.joblib"]
        );
        assert!(split_command_line("   ").is_empty());
    }

    /// End-to-end protocol check against the reference Python worker. Skips
    /// quietly when no Python launcher is available on this machine.
    #[test]
    fn talks_to_python_worker() {
        let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../classifier/worker.py");
        let launcher = ["py", "python3", "python"]
            .iter()
            .find(|p| Command::new(p).arg("--version").output().map(|o| o.status.success()).unwrap_or(false))
            .copied();
        let Some(launcher) = launcher else {
            eprintln!("no python launcher found; skipping");
            return;
        };
        let args: Vec<String> = if launcher == "py" { vec!["-3".into()] } else { vec![] }
            .into_iter()
            .chain([script.to_string_lossy().to_string(), "--dummy".to_string()])
            .collect();
        let mut backend = CommandBackend::spawn(launcher, &args, None, 0).expect("spawn worker");
        let mk = |id: i64, text: &str| ClassifyItem {
            id,
            text: text.to_string(),
            path: "x".into(),
            format: "text".into(),
        };
        let batch1 = vec![mk(1, &"long enough text ".repeat(10)), mk(2, "hi")];
        let r1 = backend.classify(&batch1).expect("batch 1");
        assert_eq!(r1.len(), 2);
        assert_eq!(r1.iter().find(|r| r.id == 1).unwrap().verdict, "good");
        assert_eq!(r1.iter().find(|r| r.id == 2).unwrap().verdict, "bad");
        // Second batch must not be confused by the first batch's terminator.
        let batch2 = vec![mk(3, &"more text ".repeat(20))];
        let r2 = backend.classify(&batch2).expect("batch 2");
        assert_eq!(r2.len(), 1);
        assert_eq!(r2[0].id, 3);
        assert_eq!(r2[0].tags, vec!["dummy".to_string()]);
    }
}
