.PHONY: help setup build test clean dev install lint fmt check all run-all-demo run-all-demos setup-demos minilm-demo-hub mobilenet-demo-hub smollm-demo-hub

VENV_DIR = .venv

# Invoke the virtual-environment interpreter directly.  This avoids activating
# the environment through shell-specific syntax, so `make test` works with both
# cmd.exe on Windows and POSIX shells on Linux/macOS.
ifeq ($(OS),Windows_NT)
PYTHON ?= python
VENV_PYTHON = $(VENV_DIR)/Scripts/python.exe
VENV_PIP = $(VENV_DIR)/Scripts/pip.exe
VENV_ACTIVATE = $(VENV_DIR)/Scripts/activate
else
PYTHON ?= python3
VENV_PYTHON = $(VENV_DIR)/bin/python
VENV_PIP = $(VENV_DIR)/bin/pip
VENV_ACTIVATE = $(VENV_DIR)/bin/activate
endif

help:
	@echo "pywebnn - Python bindings for W3C WebNN"
	@echo ""
	@echo "Available targets:"
	@echo "  setup       - Create virtual environment and install dependencies"
	@echo "  dev         - Install package in development mode (requires setup first)"
	@echo "  build       - Build wheel package"
	@echo "  install     - Install from built wheel"
	@echo "  test        - Run all Python tests"
	@echo ""
	@echo "Demo targets:"
	@echo "  run-all-demo         - Run all end-to-end demos for CI verification"
	@echo "  run-all-demos        - Alias for run-all-demo"
	@echo "  lint        - Run linting checks (cargo fmt, clippy, black, mypy)"
	@echo "  fmt         - Format Rust and Python code"
	@echo "  check       - Run cargo check"
	@echo "  clean       - Remove build artifacts and virtual environment"
	@echo "  all         - Setup, build, and test"

setup:
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV_DIR)
	@echo "Installing development dependencies..."
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install maturin pytest numpy onnxruntime
	@echo "Virtual environment ready at $(VENV_DIR)"

dev: setup
	@echo "Building and installing pywebnn in development mode..."
	@if [ "$$(uname)" = "Darwin" ]; then \
		. $(VENV_ACTIVATE) && maturin develop --features onnx-runtime,coreml-runtime; \
	else \
		. $(VENV_ACTIVATE) && maturin develop --features onnx-runtime; \
	fi
	@echo "Development installation complete"

build:
	@echo "Building release wheel..."
	maturin build --release
	@echo "Wheel built in target/wheels/"

install: build
	@echo "Installing from wheel..."
	pip install --force-reinstall target/wheels/pywebnn-*.whl
	@echo "Installation complete"

test:
	@echo "Running Python tests..."
	@$(VENV_PYTHON) tools/run_tests.py

lint:
	@echo "Running Rust linting..."
	cargo fmt --check
	cargo clippy -- -D warnings
	@if [ -d "$(VENV_DIR)" ]; then \
		echo "Running Python linting..."; \
		. $(VENV_ACTIVATE) && python -m black --check python/ tests/ examples/ || true; \
		. $(VENV_ACTIVATE) && python -m mypy python/ || true; \
	fi

fmt:
	@echo "Formatting Rust code..."
	cargo fmt
	@if [ -d "$(VENV_DIR)" ]; then \
		echo "Formatting Python code..."; \
		. $(VENV_ACTIVATE) && python -m black python/ tests/ examples/ || true; \
	fi

check:
	@echo "Running cargo check..."
	cargo check

clean:
	@echo "Cleaning build artifacts..."
	cargo clean
	rm -rf target/
	rm -rf $(VENV_DIR)
	rm -rf python/webnn/__pycache__
	rm -rf tests/__pycache__
	rm -rf .pytest_cache
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.so" -delete
	@echo "Clean complete"

run-all-demo run-all-demos: dev
	$(VENV_PYTHON) examples/python_simple.py
	$(VENV_PYTHON) examples/python_matmul.py
	$(VENV_PYTHON) examples/dispatch_example.py
	$(VENV_PYTHON) examples/test_quantization_roundtrip.py
	$(VENV_PYTHON) examples/kv_cache_device_tensors.py
	$(VENV_PYTHON) examples/text_generation_gpt.py --tokens 3 --temperature 0
	$(VENV_PYTHON) examples/text_generation_enhanced.py --use-kv-cache --tokens 3 --temperature 0

setup-demos: setup
	$(VENV_PIP) install pillow huggingface_hub transformers torch tokenizers

minilm-demo-hub: setup-demos dev
	$(VENV_PYTHON) examples/minilm_embeddings.py

mobilenet-demo-hub: setup-demos dev
	$(VENV_PYTHON) examples/mobilenetv2_from_hub.py examples/images/test.jpg --backend cpu

smollm-demo-hub: setup-demos dev
	$(VENV_PYTHON) examples/smollm_from_hub.py --backend cpu --max-new-tokens 15 --compare-transformers

all: setup dev test
	@echo "All tasks complete"
