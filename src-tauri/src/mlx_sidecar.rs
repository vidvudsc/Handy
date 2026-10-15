//! MLX transcription sidecar (macOS, Apple Silicon).
//!
//! Runs `resources/mlx_sidecar.py` as a persistent child process wrapping
//! parakeet-mlx with 8-bit quantized weights on the Apple GPU. The protocol
//! is a u64-le byte length followed by raw f32-le PCM on stdin, one JSON
//! line per result on stdout.
//!
//! This path is opt-in (`mlx_sidecar` setting) and best-effort: any failure
//! marks the sidecar unavailable and the caller falls back to the regular
//! ORT engine. The Python environment is bootstrapped on first use into
//! `<app-data>/mlx-env`.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, SystemTime};

use anyhow::{anyhow, Context, Result};
use log::{info, warn};
use tauri::path::BaseDirectory;
use tauri::{AppHandle, Manager};

struct SidecarProcess {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl Drop for SidecarProcess {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

static PROCESS: OnceLock<Mutex<Option<SidecarProcess>>> = OnceLock::new();
static STARTING: AtomicBool = AtomicBool::new(false);
static WARMING: AtomicBool = AtomicBool::new(false);
static LAST_USE: OnceLock<Mutex<Option<SystemTime>>> = OnceLock::new();
const WARM_AFTER_IDLE: Duration = Duration::from_secs(15 * 60);

fn process_slot() -> &'static Mutex<Option<SidecarProcess>> {
    PROCESS.get_or_init(|| Mutex::new(None))
}

fn last_use_slot() -> &'static Mutex<Option<SystemTime>> {
    LAST_USE.get_or_init(|| Mutex::new(None))
}

fn mark_used() {
    if let Ok(mut last_use) = last_use_slot().lock() {
        *last_use = Some(SystemTime::now());
    }
}

pub fn is_ready() -> bool {
    process_slot().lock().map(|g| g.is_some()).unwrap_or(false)
}

/// Kick off sidecar startup in the background if enabled and not running.
pub fn ensure_started(app: &AppHandle) {
    if !cfg!(target_os = "macos") || is_ready() {
        return;
    }
    if STARTING.swap(true, Ordering::SeqCst) {
        return;
    }
    let app = app.clone();
    std::thread::spawn(move || {
        match start_sidecar(&app) {
            Ok(proc) => {
                *process_slot().lock().unwrap() = Some(proc);
                mark_used();
                info!("MLX sidecar ready");
            }
            Err(e) => warn!("MLX sidecar failed to start: {e:#}"),
        }
        STARTING.store(false, Ordering::SeqCst);
    });
}

pub fn shutdown() {
    if let Ok(mut guard) = process_slot().lock() {
        *guard = None;
    }
    if let Ok(mut last_use) = last_use_slot().lock() {
        *last_use = None;
    }
}

/// Rehydrate the MLX model after a long idle period while the user is recording.
/// The real transcription waits for this request if recording ends first.
pub fn warm_if_stale() {
    let is_stale = last_use_slot()
        .lock()
        .ok()
        .and_then(|last_use| *last_use)
        .is_some_and(|last_use| {
            SystemTime::now()
                .duration_since(last_use)
                .is_ok_and(|idle| idle >= WARM_AFTER_IDLE)
        });
    if !is_stale || WARMING.swap(true, Ordering::SeqCst) {
        return;
    }

    std::thread::spawn(|| {
        let warm_result = process_slot().lock().ok().and_then(|mut guard| {
            let proc = guard.as_mut()?;
            Some(request(proc, &vec![0.0; 2 * 16_000]))
        });
        match warm_result {
            Some(Ok(_)) => {
                mark_used();
                info!("MLX sidecar idle warm-up completed");
            }
            Some(Err(e)) => warn!("MLX sidecar idle warm-up failed: {e:#}"),
            None => {}
        }
        WARMING.store(false, Ordering::SeqCst);
    });
}

/// Transcribe via the sidecar if it is running. Returns None when the
/// sidecar is unavailable so the caller can fall back to the ORT engine.
pub fn transcribe_if_ready(samples: &[f32]) -> Option<Result<String>> {
    let mut guard = process_slot().lock().ok()?;
    let proc = guard.as_mut()?;
    match request(proc, samples) {
        Ok(text) => {
            mark_used();
            Some(Ok(text))
        }
        Err(e) => {
            warn!("MLX sidecar request failed, dropping process: {e:#}");
            *guard = None;
            None
        }
    }
}

fn request(proc: &mut SidecarProcess, samples: &[f32]) -> Result<String> {
    let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
    proc.stdin
        .write_all(&(bytes.len() as u64).to_le_bytes())
        .and_then(|_| proc.stdin.write_all(&bytes))
        .and_then(|_| proc.stdin.flush())
        .context("failed to write audio to sidecar")?;

    let mut line = String::new();
    proc.stdout
        .read_line(&mut line)
        .context("failed to read sidecar response")?;
    if line.is_empty() {
        return Err(anyhow!("sidecar closed stdout"));
    }
    let value: serde_json::Value =
        serde_json::from_str(&line).context("invalid sidecar response")?;
    if let Some(err) = value.get("error").and_then(|v| v.as_str()) {
        return Err(anyhow!("sidecar error: {err}"));
    }
    if let Some(ms) = value.get("ms").and_then(|v| v.as_f64()) {
        info!("MLX sidecar compute time: {ms:.0}ms");
    }
    value
        .get("text")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow!("sidecar response missing text"))
}

fn start_sidecar(app: &AppHandle) -> Result<SidecarProcess> {
    let script = app
        .path()
        .resolve("resources/mlx_sidecar.py", BaseDirectory::Resource)
        .context("failed to resolve mlx_sidecar.py resource")?;
    let python = ensure_python_env(app)?;

    info!("starting MLX sidecar: {} {}", python, script.display());
    let mut child = Command::new(&python)
        .arg(&script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .context("failed to spawn MLX sidecar")?;

    let stdin = child.stdin.take().ok_or_else(|| anyhow!("no stdin"))?;
    let stdout = child.stdout.take().ok_or_else(|| anyhow!("no stdout"))?;
    let mut reader = BufReader::new(stdout);

    // Wait for the ready line; model load takes seconds (or minutes on
    // first run while weights download).
    let mut line = String::new();
    reader
        .read_line(&mut line)
        .context("sidecar exited before ready")?;
    if !line.contains("\"ready\"") {
        let _ = child.kill();
        return Err(anyhow!("unexpected sidecar startup message: {line}"));
    }

    Ok(SidecarProcess {
        child,
        stdin,
        stdout: reader,
    })
}

/// Return a python interpreter with parakeet-mlx installed, bootstrapping a
/// venv under app data if needed. `HANDY_MLX_PYTHON` overrides everything.
fn ensure_python_env(app: &AppHandle) -> Result<String> {
    if let Ok(p) = std::env::var("HANDY_MLX_PYTHON") {
        return Ok(p);
    }

    let env_dir = crate::portable::app_data_dir(app)
        .context("failed to resolve app data dir")?
        .join("mlx-env");
    let python = env_dir.join("bin").join("python3");
    if python.exists() {
        return Ok(python.to_string_lossy().into_owned());
    }

    info!("bootstrapping MLX python env at {}", env_dir.display());
    let status = Command::new("python3")
        .args(["-m", "venv"])
        .arg(&env_dir)
        .status()
        .context("failed to run python3 -m venv")?;
    if !status.success() {
        return Err(anyhow!("python3 -m venv failed: {status}"));
    }

    let pip = env_dir.join("bin").join("pip");
    let status = Command::new(&pip)
        .args(["install", "--quiet", "parakeet-mlx"])
        .status()
        .context("failed to run pip install parakeet-mlx")?;
    if !status.success() {
        let _ = std::fs::remove_dir_all(&env_dir);
        return Err(anyhow!("pip install parakeet-mlx failed: {status}"));
    }

    Ok(python.to_string_lossy().into_owned())
}
