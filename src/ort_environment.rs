//! Caller-configured ORT environments and dynamically loaded EP libraries.
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

// ort only permits one environment build, even after a failed EP registration.
static ENVIRONMENT: Mutex<Option<Arc<ort::environment::Environment>>> = Mutex::new(None);

pyo3::create_exception!(_rustnn, OrtAlreadyInitialized, PyRuntimeError);

/// Initialize ORT with (registration name, library path) EP libraries.
///
/// Call before creating contexts. Raises OrtAlreadyInitialized if an environment
/// is already installed, or RuntimeError if loading the runtime or an EP fails.
#[pyfunction]
#[pyo3(signature = (execution_provider_libraries))]
pub fn init_global_ort_env(execution_provider_libraries: Vec<(String, PathBuf)>) -> PyResult<()> {
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
        let environment = ort::init_from(path)
            .map_err(|e| ort::Error::new(e.to_string()))?
            .with_name("pywebnn")
            .with_execution_providers([ort::ep::CPU::default().build()])
            .with_telemetry(false)
            .build()?;
        let environment = Arc::new(environment);
        *state = Some(environment.clone());
        Ok(environment)
    })();
    let environment = result.map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    for (name, path) in execution_provider_libraries {
        environment
            .register_ep_library(name, path)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    }
    rustnn::executors::onnx::init_global_ort_env(environment)
        .map_err(|e| OrtAlreadyInitialized::new_err(e.to_string()))
}
