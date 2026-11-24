// SPDX-FileCopyrightText: 2026 NVIDIA
//
// SPDX-License-Identifier: Apache-2.0

use std::{io::IsTerminal, sync::Once};

use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

/// Enable RustNN's existing log output only when the embedding application
/// explicitly configures `RUST_LOG`. A library must not install a global logger
/// during normal imports because Python applications may own that logger.
pub(crate) fn init_rust_logging_if_requested() {
    static LOGGER_INIT: Once = Once::new();
    if std::env::var_os("RUST_LOG").is_some() {
        LOGGER_INIT.call_once(|| {
            let env_filter =
                tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                    tracing_subscriber::EnvFilter::builder()
                        .with_default_directive(tracing::level_filters::LevelFilter::WARN.into())
                        .parse("")
                        .unwrap()
                });
            let fmt = {
                let this = tracing_subscriber::fmt::layer()
                    .compact()
                    .without_time()
                    .with_writer(std::io::stderr)
                    .with_ansi(std::io::stderr().is_terminal());
                tracing_subscriber::filter::Filtered::new(this, env_filter)
            };
            tracing_subscriber::registry()
                //.with(nvtx::tracing::NvtxLayer::default()) // if using https://github.com/NVIDIA/NVTX/tree/dev/rust
                .with(fmt)
                .init();
        });
    }
}
