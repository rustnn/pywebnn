//! Caller-configured ORT environments and dynamically loaded EP libraries.
use ort::logging::LogLevel;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};

// ort only permits one environment build, even after a failed EP registration.
static ENVIRONMENT: Mutex<Option<PendingEnvironment>> = Mutex::new(None);

#[derive(Clone)]
struct PendingEnvironment {
    environment: Arc<ort::environment::Environment>,
    // u8::MAX preserves the default tracing route; other values enable stdout.
    stdout_level: Arc<AtomicU8>,
}

fn ort_logger(stdout_level: Arc<AtomicU8>) -> ort::logging::LoggerFunction {
    Arc::new(move |level, category, id, code_location, message| {
        let minimum = stdout_level.load(Ordering::Relaxed);
        if minimum != u8::MAX {
            if (level as u8) >= minimum {
                // ORT may log from worker threads. Avoid the Python GIL and never
                // panic across its FFI callback if stdout has been closed.
                let mut stdout = std::io::stdout().lock();
                let _ = writeln!(stdout, "[ORT {level:?}] [{category}] [{id}] {message}");
                let _ = stdout.flush();
            }
        } else {
            // Preserve ort's default tracing route when Python did not opt in.
            match level {
                LogLevel::Verbose => {
                    tracing::trace!(target: "ort::logging", category, id, code_location, "{message}")
                }
                LogLevel::Info => {
                    tracing::info!(target: "ort::logging", category, id, code_location, "{message}")
                }
                LogLevel::Warning => {
                    tracing::warn!(target: "ort::logging", category, id, code_location, "{message}")
                }
                LogLevel::Error => {
                    tracing::error!(target: "ort::logging", category, id, code_location, "{message}")
                }
                LogLevel::Fatal => {
                    tracing::error!(target: "ort::logging", category, id, code_location, "(FATAL): {message}")
                }
            }
        }
    })
}

pyo3::create_exception!(_rustnn, OrtAlreadyInitialized, PyRuntimeError);

/// Initialize ORT with (registration name, library path) EP libraries.
///
/// Call before creating contexts. Raises OrtAlreadyInitialized if an environment
/// is already installed, or RuntimeError if loading the runtime or an EP fails.
/// `log_level` accepts verbose, info, warning, error, or fatal (case-insensitive).
/// Explicit levels print matching ORT records to native stdout, including during
/// environment construction. None preserves the default tracing route. Invalid
/// levels raise ValueError before loading ORT.
#[pyfunction]
#[pyo3(signature = (execution_provider_libraries, *, log_level=None))]
pub fn init_global_ort_env(
    execution_provider_libraries: Vec<(String, PathBuf)>,
    log_level: Option<&str>,
) -> PyResult<()> {
    let log_level = log_level
        .map(|level| match level.to_ascii_lowercase().as_str() {
            "verbose" => Ok(LogLevel::Verbose),
            "info" => Ok(LogLevel::Info),
            "warning" => Ok(LogLevel::Warning),
            "error" => Ok(LogLevel::Error),
            "fatal" => Ok(LogLevel::Fatal),
            _ => Err(PyValueError::new_err(
                "log_level must be one of: verbose, info, warning, error, fatal",
            )),
        })
        .transpose()?;
    let mut state = ENVIRONMENT.lock().unwrap_or_else(|e| e.into_inner());
    if rustnn::executors::onnx::is_ort_initialized() {
        return Err(OrtAlreadyInitialized::new_err(
            "the global ONNX Runtime environment is already initialized",
        ));
    }
    let result = (|| -> ort::Result<_> {
        if let Some(environment) = state.as_ref() {
            return Ok(environment.clone());
        }
        let path = std::env::var_os("ORT_DYLIB_PATH")
            .filter(|path| !path.is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                format!(
                    "{}onnxruntime{}",
                    std::env::consts::DLL_PREFIX,
                    std::env::consts::DLL_SUFFIX
                )
                .into()
            });
        let stdout_level = Arc::new(AtomicU8::new(
            log_level.map_or(u8::MAX, |level| level as u8),
        ));
        let environment = ort::init_from(path)
            .map_err(|e| ort::Error::new(e.to_string()))?
            .with_name("pywebnn")
            .with_logger(ort_logger(stdout_level.clone()))
            .with_execution_providers([ort::ep::CPU::default().build()])
            .with_telemetry(false)
            .build()?;
        let pending = PendingEnvironment {
            environment: Arc::new(environment),
            stdout_level,
        };
        *state = Some(pending.clone());
        Ok(pending)
    })();
    let pending = result.map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    // A retry after a failed plugin load can change both routing and severity.
    pending.stdout_level.store(
        log_level.map_or(u8::MAX, |level| level as u8),
        Ordering::Relaxed,
    );
    let environment = pending.environment;
    environment.set_log_level(log_level.unwrap_or(LogLevel::Verbose));
    for (name, path) in execution_provider_libraries {
        environment
            .register_ep_library(name, path)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    }
    rustnn::executors::onnx::init_global_ort_env(environment)
        .map_err(|e| OrtAlreadyInitialized::new_err(e.to_string()))
}
