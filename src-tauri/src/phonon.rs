//! Bundled Phonon-2 Core ML engine, isolated in a persistent Swift process.
use anyhow::{anyhow, bail, Context, Result};
use serde_json::Value;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::time::{Duration, Instant};
use tauri::{path::BaseDirectory, AppHandle, Manager};

pub const MODEL_ID: &str = "phonon-2-coreml";
// ANE preparation can recur after idle, even with a compiled-model cache.
// Preparation runs independently of interactive dictation, which uses a
// temporary backup while the native engine becomes ready.
const STARTUP_TIMEOUT: Duration = Duration::from_secs(20 * 60);
const RESTART_TIMEOUT: Duration = Duration::from_secs(15);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
static CANCEL_EPOCH: AtomicU64 = AtomicU64::new(0);

pub fn cancel_pending() {
    CANCEL_EPOCH.fetch_add(1, Ordering::AcqRel);
}

pub fn cancel_epoch() -> u64 {
    CANCEL_EPOCH.load(Ordering::Acquire)
}

pub fn ensure_not_cancelled(epoch: u64) -> Result<()> {
    if cancel_epoch() != epoch {
        bail!("Phonon operation cancelled");
    }
    Ok(())
}

fn wait_reply(receiver: &Receiver<Reply>, timeout: Duration, epoch: u64) -> Result<Value> {
    let deadline = Instant::now() + timeout;
    loop {
        ensure_not_cancelled(epoch)?;
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            bail!("Phonon helper timed out");
        }
        match receiver.recv_timeout(remaining.min(Duration::from_millis(25))) {
            Ok(reply) => return reply.map_err(|e| anyhow!(e)),
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => bail!("Phonon helper exited"),
        }
    }
}
const MAX_AUDIO_BYTES: usize = 64 * 1024 * 1024;
const MAX_RESPONSE_BYTES: u64 = 1024 * 1024;

fn supported_platform() -> bool {
    if !cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        return false;
    }
    Command::new("/usr/bin/sw_vers")
        .arg("-productVersion")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().split('.').next()?.parse::<u32>().ok())
        .is_some_and(|major| major >= 15)
}

pub fn bundled_model_path(app: &AppHandle) -> Result<PathBuf> {
    if !supported_platform() {
        bail!("Phonon Core ML requires Apple silicon and macOS 15 or later");
    }
    let root = app
        .path()
        .resolve("resources/phonon-coreml", BaseDirectory::Resource)?;
    let model = root.join("model");
    for file in [
        root.join("handy-phonon"),
        model.join("manifest.json"),
        model.join("decoder.bin"),
        model.join("Phonon-2.mlpackage/Data/com.apple.CoreML/weights/weight.bin"),
    ] {
        if !file.is_file() {
            bail!("Phonon bundle is incomplete: {}", file.display());
        }
    }
    Ok(model)
}

type Reply = std::result::Result<Value, String>;
struct Request {
    audio: Vec<u8>,
    reply: Sender<Reply>,
}

struct Process {
    child: Child,
    requests: Sender<Request>,
}

impl Drop for Process {
    fn drop(&mut self) {
        // Also unblocks the worker if it is stuck writing to or reading a pipe.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn read_response(reader: &mut impl BufRead) -> Result<Value> {
    let mut line = String::new();
    reader.take(MAX_RESPONSE_BYTES + 1).read_line(&mut line)?;
    if line.is_empty() {
        bail!("Phonon helper closed stdout");
    }
    if line.len() as u64 > MAX_RESPONSE_BYTES || !line.ends_with('\n') {
        bail!("Phonon helper response exceeds the protocol limit");
    }
    let value: Value = serde_json::from_str(&line).context("Invalid Phonon response")?;
    if let Some(error) = value.get("error").and_then(Value::as_str) {
        bail!("Phonon helper: {error}");
    }
    Ok(value)
}

impl Process {
    fn spawn(
        binary: &Path,
        args: &[&std::ffi::OsStr],
        startup_timeout: Duration,
        epoch: u64,
    ) -> Result<Self> {
        let mut child = Command::new(binary)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .context("Failed to launch Phonon helper")?;
        let stdin = child.stdin.take();
        let stdout = child.stdout.take();
        let (requests, receiver) = mpsc::channel::<Request>();
        let (ready_tx, ready_rx) = mpsc::channel::<Reply>();
        let process = Self { child, requests };
        let mut stdin = stdin.ok_or_else(|| anyhow!("Missing Phonon stdin"))?;
        let mut stdout = BufReader::new(stdout.ok_or_else(|| anyhow!("Missing Phonon stdout"))?);
        std::thread::spawn(move || {
            let ready = read_response(&mut stdout).map_err(|e| e.to_string());
            let success = ready
                .as_ref()
                .is_ok_and(|v| v.get("ready") == Some(&Value::Bool(true)));
            let _ = ready_tx.send(ready);
            if !success {
                return;
            }
            for request in receiver {
                let response = (|| -> Result<Value> {
                    stdin.write_all(&(request.audio.len() as u64).to_le_bytes())?;
                    stdin.write_all(&request.audio)?;
                    stdin.flush()?;
                    read_response(&mut stdout)
                })();
                let failed = response.is_err();
                let _ = request.reply.send(response.map_err(|e| e.to_string()));
                if failed {
                    break;
                }
            }
        });
        let ready = wait_reply(&ready_rx, startup_timeout, epoch)
            .context("Phonon startup did not complete")?;
        if ready.get("ready") != Some(&Value::Bool(true)) {
            bail!("Phonon helper did not become ready");
        }
        Ok(process)
    }

    fn transcribe(&self, samples: &[f32], timeout: Duration) -> Result<String> {
        let epoch = cancel_epoch();
        if samples.len() > MAX_AUDIO_BYTES / 4 {
            bail!("Recording exceeds Phonon's 64 MB PCM request limit");
        }
        if samples.iter().any(|s| !s.is_finite()) {
            bail!("Recording contains invalid audio samples");
        }
        let audio = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        let (reply, response) = mpsc::channel();
        self.requests
            .send(Request { audio, reply })
            .context("Phonon helper is unavailable")?;
        let value = wait_reply(&response, timeout, epoch)
            .context("Phonon transcription did not complete")?;
        let text = value
            .get("text")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("Phonon response is missing its text"))?;
        log::info!(
            "Phonon Core ML compute time: {}ms",
            value.get("ms").unwrap_or(&Value::Null)
        );
        Ok(text.to_string())
    }
}

pub struct PhononEngine {
    binary: PathBuf,
    model: PathBuf,
    process: Option<Process>,
}

impl PhononEngine {
    pub fn load(app: &AppHandle, model: &Path, epoch: u64) -> Result<Self> {
        let binary = app.path().resolve(
            "resources/phonon-coreml/handy-phonon",
            BaseDirectory::Resource,
        )?;
        ensure_not_cancelled(epoch)?;
        let process = Process::spawn(&binary, &[model.as_os_str()], STARTUP_TIMEOUT, epoch)?;
        Ok(Self {
            binary,
            model: model.to_path_buf(),
            process: Some(process),
        })
    }

    pub fn transcribe(&mut self, samples: &[f32]) -> Result<String> {
        if samples.is_empty() {
            return Ok(String::new());
        }
        if self.process.is_none() {
            self.process = Some(Process::spawn(
                &self.binary,
                &[self.model.as_os_str()],
                RESTART_TIMEOUT,
                cancel_epoch(),
            )?);
        }
        let result = self
            .process
            .as_ref()
            .context("Phonon helper is unavailable")?
            .transcribe(samples, REQUEST_TIMEOUT);
        if result.is_err() {
            self.process = None;
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    // Serialize tests that exercise the process-wide cancellation generation.
    static CANCEL_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn protocol_rejects_eof_errors_and_unterminated_responses() {
        assert!(read_response(&mut Cursor::new(b"")).is_err());
        assert!(read_response(&mut Cursor::new(b"{\"error\":\"failed\"}\n")).is_err());
        assert!(read_response(&mut Cursor::new(b"{\"text\":\"partial\"}")).is_err());
        let value = read_response(&mut Cursor::new(b"{\"text\":\"Hello\",\"ms\":12}\n")).unwrap();
        assert_eq!(value["text"], "Hello");
    }

    #[test]
    fn cancelled_wait_returns_without_waiting_for_the_helper() {
        let _serial = CANCEL_TEST_LOCK.lock().unwrap();
        let (_sender, receiver) = mpsc::channel::<Reply>();
        let started = Instant::now();
        assert!(wait_reply(
            &receiver,
            Duration::from_secs(30),
            cancel_epoch().wrapping_sub(1)
        )
        .is_err());
        assert!(started.elapsed() < Duration::from_secs(1));
    }

    #[test]
    fn cancel_interrupts_a_wait_that_is_already_running() {
        let _serial = CANCEL_TEST_LOCK.lock().unwrap();
        let (_sender, receiver) = mpsc::channel::<Reply>();
        let epoch = cancel_epoch();
        let cancel = std::thread::spawn(|| {
            std::thread::sleep(Duration::from_millis(50));
            cancel_pending();
        });
        let started = Instant::now();
        let error = wait_reply(&receiver, Duration::from_secs(30), epoch).unwrap_err();
        cancel.join().unwrap();
        assert!(error.to_string().contains("cancelled"));
        assert!(started.elapsed() < Duration::from_secs(1));
    }

    #[cfg(unix)]
    #[test]
    fn startup_timeout_reaps_a_helper_that_never_becomes_ready() {
        let _serial = CANCEL_TEST_LOCK.lock().unwrap();
        let started = Instant::now();
        let error = Process::spawn(
            Path::new("/bin/sh"),
            &[
                std::ffi::OsStr::new("-c"),
                std::ffi::OsStr::new("exec sleep 30"),
            ],
            Duration::from_millis(50),
            cancel_epoch(),
        )
        .err()
        .expect("hung startup must time out");
        assert!(format!("{error:#}").contains("timed out"));
        assert!(started.elapsed() >= Duration::from_millis(50));
        assert!(started.elapsed() < Duration::from_secs(2));
    }

    #[cfg(unix)]
    #[test]
    fn hung_helper_has_a_bounded_request_and_is_reaped() {
        let _serial = CANCEL_TEST_LOCK.lock().unwrap();
        let mut process = Process::spawn(
            Path::new("/bin/sh"),
            &[
                std::ffi::OsStr::new("-c"),
                std::ffi::OsStr::new("printf '{\"ready\":true}\\n'; exec sleep 30"),
            ],
            Duration::from_secs(2),
            cancel_epoch(),
        )
        .unwrap();
        let started = Instant::now();
        let error = process
            .transcribe(&[0.1; 64], Duration::from_millis(50))
            .unwrap_err();
        assert!(format!("{error:#}").contains("timed out"));
        assert!(started.elapsed() >= Duration::from_millis(50));
        process.child.kill().unwrap();
        assert!(process.child.wait().is_ok());
    }
}
