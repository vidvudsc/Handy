#[cfg(feature = "ort-coreml")]
use ort::ep::CoreML;
#[cfg(feature = "ort-directml")]
use ort::ep::DirectML;
#[cfg(feature = "ort-rocm")]
use ort::ep::ROCm;
#[cfg(feature = "ort-webgpu")]
use ort::ep::WebGPU;
#[cfg(feature = "ort-xnnpack")]
use ort::ep::XNNPACK;
use ort::ep::CPU;
#[cfg(feature = "ort-cuda")]
use ort::ep::CUDA;

use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
#[cfg(feature = "ort-xnnpack")]
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use crate::accel::{get_ort_accelerator, OrtAccelerator};

fn env_bool(name: &str) -> Option<bool> {
    std::env::var(name)
        .ok()
        .and_then(|value| match value.to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Some(true),
            "0" | "false" | "no" | "off" => Some(false),
            _ => None,
        })
}

#[cfg(feature = "ort-xnnpack")]
fn xnnpack_enabled() -> bool {
    env_bool("TRANSCRIBE_RS_ORT_XNNPACK").unwrap_or(false)
}

#[cfg(feature = "ort-xnnpack")]
fn xnnpack_threads() -> NonZeroUsize {
    std::env::var("TRANSCRIBE_RS_ORT_XNNPACK_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .and_then(NonZeroUsize::new)
        .unwrap_or_else(|| NonZeroUsize::new(4).expect("constant is non-zero"))
}

/// Build the execution provider list based on an accelerator preference.
fn execution_providers_for(pref: OrtAccelerator) -> Vec<ort::ep::ExecutionProviderDispatch> {
    let mut eps = Vec::new();

    match pref {
        OrtAccelerator::CpuOnly => {
            // CPU only — no GPU providers
        }
        OrtAccelerator::Cuda => {
            #[cfg(feature = "ort-cuda")]
            eps.push(CUDA::default().build());
            #[cfg(not(feature = "ort-cuda"))]
            log::warn!(
                "Accelerator set to CUDA but ort-cuda feature is not enabled; falling back to CPU"
            );
        }
        OrtAccelerator::DirectMl => {
            #[cfg(feature = "ort-directml")]
            eps.push(DirectML::default().build());
            #[cfg(not(feature = "ort-directml"))]
            log::warn!("Accelerator set to DirectML but ort-directml feature is not enabled; falling back to CPU");
        }
        OrtAccelerator::Rocm => {
            #[cfg(feature = "ort-rocm")]
            eps.push(ROCm::default().build());
            #[cfg(not(feature = "ort-rocm"))]
            log::warn!(
                "Accelerator set to ROCm but ort-rocm feature is not enabled; falling back to CPU"
            );
        }
        OrtAccelerator::CoreMl => {
            #[cfg(feature = "ort-coreml")]
            eps.push(CoreML::default().build());
            #[cfg(not(feature = "ort-coreml"))]
            log::warn!(
                "Accelerator set to CoreML but ort-coreml feature is not enabled; falling back to CPU"
            );
        }
        OrtAccelerator::WebGpu => {
            #[cfg(feature = "ort-webgpu")]
            eps.push(WebGPU::default().build());
            #[cfg(not(feature = "ort-webgpu"))]
            log::warn!(
                "Accelerator set to WebGPU but ort-webgpu feature is not enabled; falling back to CPU"
            );
        }
        OrtAccelerator::Auto => {
            // Add compiled-in GPU EPs in priority order.
            // DirectML and WebGPU are excluded from Auto because they require
            // parallel_execution(false) and memory_pattern(false),
            // which would penalize other backends. Use the explicit variant
            // to opt in.
            // Ref: https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html
            //      https://onnxruntime.ai/docs/execution-providers/WebGPU-ExecutionProvider.html
            #[cfg(feature = "ort-cuda")]
            eps.push(CUDA::default().build());
            #[cfg(feature = "ort-rocm")]
            eps.push(ROCm::default().build());
            // CoreML is safe for Auto on macOS — analogous to CUDA on NVIDIA
            // and ROCm on AMD. It does not require sequential execution or
            // disabled memory patterns.
            #[cfg(feature = "ort-coreml")]
            eps.push(CoreML::default().build());
        }
    }

    #[cfg(feature = "ort-xnnpack")]
    if xnnpack_enabled() {
        eps.push(
            XNNPACK::default()
                .with_intra_op_num_threads(xnnpack_threads())
                .build(),
        );
    }

    // CPU is always the final fallback.
    let cpu = CPU::default()
        .with_arena_allocator(env_bool("TRANSCRIBE_RS_ORT_CPU_ARENA").unwrap_or(false));
    eps.push(cpu.build());
    eps
}

/// Build the execution provider list based on the global accelerator preference.
fn execution_providers() -> Vec<ort::ep::ExecutionProviderDispatch> {
    execution_providers_for(get_ort_accelerator())
}

/// Returns true if the selected execution provider requires sequential execution
/// and disabled memory patterns (DirectML, WebGPU).
fn requires_sequential_session() -> bool {
    let pref = get_ort_accelerator();
    (pref == OrtAccelerator::DirectMl && cfg!(feature = "ort-directml"))
        || (pref == OrtAccelerator::WebGpu && cfg!(feature = "ort-webgpu"))
}

fn env_intra_threads() -> Option<usize> {
    std::env::var("TRANSCRIBE_RS_ORT_INTRA_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&threads| threads > 0)
}

fn env_parallel_execution() -> Option<bool> {
    env_bool("TRANSCRIBE_RS_ORT_PARALLEL")
}

fn env_memory_pattern() -> Option<bool> {
    env_bool("TRANSCRIBE_RS_ORT_MEMORY_PATTERN")
}

fn env_dynamic_block_base() -> Option<u32> {
    std::env::var("TRANSCRIBE_RS_ORT_DYNAMIC_BLOCK_BASE")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .filter(|&value| value > 0)
}

fn env_profile_path(model_path: &Path) -> Option<PathBuf> {
    let dir = std::env::var("TRANSCRIBE_RS_ORT_PROFILE_DIR").ok()?;
    let stem = model_path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("model");
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_nanos();

    Some(PathBuf::from(dir).join(format!("{stem}-{pid}-{nanos}.json")))
}

fn env_graph_optimization_level() -> Option<GraphOptimizationLevel> {
    std::env::var("TRANSCRIBE_RS_ORT_OPT_LEVEL")
        .ok()
        .and_then(|value| match value.to_ascii_lowercase().as_str() {
            "disable" | "0" => Some(GraphOptimizationLevel::Disable),
            "level1" | "1" => Some(GraphOptimizationLevel::Level1),
            "level2" | "2" => Some(GraphOptimizationLevel::Level2),
            "level3" | "3" => Some(GraphOptimizationLevel::Level3),
            "all" => Some(GraphOptimizationLevel::All),
            _ => None,
        })
}

/// Internal session builder with full control over threading and EP selection.
fn build_session(
    path: &Path,
    intra_threads: Option<usize>,
    parallel_execution: bool,
    accelerator: Option<OrtAccelerator>,
) -> Result<Session, ort::Error> {
    let opt_level = env_graph_optimization_level().unwrap_or(GraphOptimizationLevel::Level3);
    let mut builder = Session::builder()?.with_optimization_level(opt_level)?;

    if let Some(profile_path) = env_profile_path(path) {
        if let Some(parent) = profile_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        builder = builder.with_profiling(profile_path)?;
    }

    let intra_threads = {
        #[cfg(feature = "ort-xnnpack")]
        {
            if xnnpack_enabled() {
                Some(1)
            } else {
                intra_threads.or_else(env_intra_threads)
            }
        }
        #[cfg(not(feature = "ort-xnnpack"))]
        {
            intra_threads.or_else(env_intra_threads)
        }
    };
    if let Some(n) = intra_threads {
        if n > 0 {
            builder = builder.with_intra_threads(n)?;
        }
    }

    if env_bool("TRANSCRIBE_RS_ORT_FLUSH_TO_ZERO").unwrap_or(false) {
        builder = builder.with_flush_to_zero()?;
    }

    if env_bool("TRANSCRIBE_RS_ORT_QDQ_CLEANUP").unwrap_or(false) {
        builder = builder.with_qdq_cleanup()?;
    }

    if env_bool("TRANSCRIBE_RS_ORT_APPROX_GELU").unwrap_or(false) {
        builder = builder.with_approximate_gelu()?;
    }

    if let Some(size) = env_dynamic_block_base() {
        builder = builder.with_dynamic_block_base(size)?;
    }

    #[cfg(feature = "ort-xnnpack")]
    if xnnpack_enabled() {
        builder = builder.with_intra_op_spinning(false)?;
    }

    let parallel_execution = env_parallel_execution().unwrap_or(parallel_execution);

    // DirectML and WebGPU require parallel_execution(false) and memory_pattern(false)
    let use_parallel = if requires_sequential_session() {
        false
    } else {
        parallel_execution
    };

    builder = builder.with_parallel_execution(use_parallel)?;

    if let Some(memory_pattern) = env_memory_pattern() {
        builder = builder.with_memory_pattern(memory_pattern)?;
    } else if requires_sequential_session() {
        builder = builder.with_memory_pattern(false)?;
    }

    let session = builder
        .with_execution_providers(match accelerator {
            Some(accelerator) => execution_providers_for(accelerator),
            None => execution_providers(),
        })?
        .commit_from_file(path)?;

    for input in session.inputs() {
        log::info!(
            "Model input: name={}, type={:?}",
            input.name(),
            input.dtype()
        );
    }
    for output in session.outputs() {
        log::info!(
            "Model output: name={}, type={:?}",
            output.name(),
            output.dtype()
        );
    }

    Ok(session)
}

/// Create an ONNX session with standard settings.
pub fn create_session(path: &Path) -> Result<Session, ort::Error> {
    build_session(path, None, true, None)
}

/// Create an ONNX session with configurable thread count.
pub fn create_session_with_threads(path: &Path, num_threads: usize) -> Result<Session, ort::Error> {
    build_session(path, Some(num_threads), true, None)
}

/// Create an ONNX session with configurable thread count and execution mode.
pub fn create_session_with_options(
    path: &Path,
    num_threads: Option<usize>,
    parallel_execution: bool,
) -> Result<Session, ort::Error> {
    build_session(path, num_threads, parallel_execution, None)
}

/// Create an ONNX session with configurable thread count, execution mode, and accelerator.
pub fn create_session_with_options_and_accelerator(
    path: &Path,
    num_threads: Option<usize>,
    parallel_execution: bool,
    accelerator: Option<OrtAccelerator>,
) -> Result<Session, ort::Error> {
    build_session(path, num_threads, parallel_execution, accelerator)
}

/// Resolve a model file path for the requested quantization level.
///
/// Looks for `{name}.{suffix}.onnx` based on the quantization variant,
/// falling back to `{name}.onnx` (FP32) if the requested file doesn't exist.
pub fn resolve_model_path(
    dir: &Path,
    name: &str,
    quantization: &super::Quantization,
) -> std::path::PathBuf {
    let suffix = match quantization {
        super::Quantization::FP32 => None,
        super::Quantization::FP16 => Some("fp16"),
        super::Quantization::Int8 => Some("int8"),
        super::Quantization::Int4 => Some("int4"),
    };

    if let Some(suffix) = suffix {
        let path = dir.join(format!("{}.{}.onnx", name, suffix));
        if path.exists() {
            log::info!("Loading {} model: {}", suffix, path.display());
            return path;
        }
        log::warn!(
            "{} model not found at {}, falling back to {}.onnx",
            suffix,
            path.display(),
            name
        );
    }

    dir.join(format!("{}.onnx", name))
}

/// Read a custom metadata string from an ONNX session.
pub fn read_metadata_str(session: &Session, key: &str) -> Result<Option<String>, ort::Error> {
    let meta = session.metadata()?;
    Ok(meta.custom(key).filter(|s| !s.is_empty()))
}

/// Read a custom metadata i32 value, with optional default.
pub fn read_metadata_i32(
    session: &Session,
    key: &str,
    default: Option<i32>,
) -> Result<Option<i32>, crate::TranscribeError> {
    let str_val = read_metadata_str(session, key).map_err(|e| {
        crate::TranscribeError::Config(format!("failed to read metadata '{}': {}", key, e))
    })?;
    match str_val {
        Some(v) => Ok(Some(v.parse::<i32>().map_err(|e| {
            crate::TranscribeError::Config(format!("failed to parse '{}': {}", key, e))
        })?)),
        None => Ok(default),
    }
}

/// Read a comma-separated float vector from metadata.
pub fn read_metadata_float_vec(
    session: &Session,
    key: &str,
) -> Result<Option<Vec<f32>>, crate::TranscribeError> {
    let str_val = read_metadata_str(session, key).map_err(|e| {
        crate::TranscribeError::Config(format!("failed to read metadata '{}': {}", key, e))
    })?;
    match str_val {
        Some(v) => {
            let floats: Result<Vec<f32>, _> =
                v.split(',').map(|s| s.trim().parse::<f32>()).collect();
            Ok(Some(floats.map_err(|e| {
                crate::TranscribeError::Config(format!(
                    "failed to parse floats in '{}': {}",
                    key, e
                ))
            })?))
        }
        None => Ok(None),
    }
}
