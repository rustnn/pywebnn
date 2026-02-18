.PHONY: help setup setup-demos build test clean dev install lint fmt check all \
	minilm-demo-hub mobilenet-demo-hub smollm-demo-hub run-all-demo run-all-demos

# Python version to use (defaults to python3 in PATH)
PYTHON ?= python3.10
VENV_DIR = .venv
VENV_ACTIVATE = $(VENV_DIR)/bin/activate

help:
	@echo "pywebnn - Python bindings for W3C WebNN"
	@echo ""
	@echo "Available targets:"
	@echo "  setup       - Create virtual environment and install dependencies"
	@echo "  setup-demos - Install demo dependencies (transformers, torch, Pillow)"
	@echo "  dev         - Install package in development mode (requires setup first)"
	@echo "  build       - Build wheel package"
	@echo "  install     - Install from built wheel"
	@echo "  test        - Run all Python tests"
	@echo ""
	@echo "Demo targets:"
	@echo "  minilm-demo-hub      - Run MiniLM embeddings demo (Hugging Face Hub)"
	@echo "  mobilenet-demo-hub   - Run MobileNetV2 classification demo (Hugging Face Hub)"
	@echo "  smollm-demo-hub      - Run SmolLM text generation demo (Hugging Face Hub)"
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
	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/pip install maturin pytest pytest-asyncio numpy onnxruntime
	@echo "Virtual environment ready at $(VENV_DIR)"

setup-demos: setup
	@echo "Installing demo dependencies..."
	$(VENV_DIR)/bin/pip install transformers torch Pillow requests --extra-index-url https://download.pytorch.org/whl/cpu
	@echo "Demo dependencies installed"

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
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Virtual environment not found. Run 'make setup' first."; \
		exit 1; \
	fi
	@echo "Running Python tests..."
	. $(VENV_ACTIVATE) && pytest tests/ -v

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

# ==============================================================================
# Demo Targets - End-to-end verification
# ==============================================================================

RUN_ALL_DEMOS_LEVELS ?= none

minilm-demo-hub: setup-demos dev
	@echo "========================================================================"
	@echo "Running all-MiniLM-L6-v2 demo (Hugging Face Hub)"
	@echo "========================================================================"
	@echo ""
	@echo "Downloading model from Hugging Face Hub: tarekziade/all-MiniLM-L6-v2-webnn"
	@echo "------------------------------------------------------------------------"
	MINILM_MODEL_ID=tarekziade/all-MiniLM-L6-v2-webnn $(VENV_DIR)/bin/python examples/minilm_embeddings.py
	@echo ""
	@echo "========================================================================"
	@echo "Demo completed successfully!"
	@echo "========================================================================"

mobilenet-demo-hub: setup-demos dev
	@echo "========================================================================"
	@echo "Running MobileNetV2 (Hugging Face Hub)"
	@echo "========================================================================"
	@echo ""
	@echo "Downloading model from Hugging Face Hub: tarekziade/mobilenet-webnn"
	@echo "------------------------------------------------------------------------"
	$(VENV_DIR)/bin/python examples/mobilenetv2_from_hub.py examples/images/test.jpg --backend cpu
	@echo ""
	@echo "========================================================================"
	@echo "Demo completed successfully!"
	@echo "========================================================================"

smollm-demo-hub: setup-demos dev
	@echo "========================================================================"
	@echo "Running SmolLM-135M generation demo (Hugging Face Hub)"
	@echo "========================================================================"
	@echo ""
	@echo "Downloading model from Hugging Face Hub: tarekziade/SmolLM-135M-webnn"
	@echo "------------------------------------------------------------------------"
	$(VENV_DIR)/bin/python examples/smollm_from_hub.py --backend cpu --max-new-tokens 15 --compare-transformers
	@echo ""
	@echo "========================================================================"
	@echo "Demo completed successfully!"
	@echo "========================================================================"

run-all-demo: setup-demos dev
	@echo "========================================================================"
	@echo "Running All Demos (quantization=$(RUN_ALL_DEMOS_LEVELS))"
	@echo "========================================================================"
	@echo ""
	@echo "Demo 1/5: Quantization Round-Trip Test"
	@echo "------------------------------------------------------------------------"
	RUN_ALL_DEMOS_LEVELS='$(RUN_ALL_DEMOS_LEVELS)' $(VENV_DIR)/bin/python examples/test_quantization_roundtrip.py
	@echo ""
	@echo "Demo 2/5: MiniLM Embeddings (Hugging Face Hub)"
	@echo "------------------------------------------------------------------------"
	@RUN_ALL_DEMOS_LEVELS='$(RUN_ALL_DEMOS_LEVELS)' $(MAKE) minilm-demo-hub
	@echo ""
	@echo "Demo 3/5: MobileNetV2 Image Classification (Hugging Face Hub)"
	@echo "------------------------------------------------------------------------"
	@RUN_ALL_DEMOS_LEVELS='$(RUN_ALL_DEMOS_LEVELS)' $(MAKE) mobilenet-demo-hub
	@echo ""
	@echo "Demo 4/5: SmolLM-135M Text Generation (Hugging Face Hub)"
	@echo "------------------------------------------------------------------------"
	@RUN_ALL_DEMOS_LEVELS='$(RUN_ALL_DEMOS_LEVELS)' $(MAKE) smollm-demo-hub
	@echo ""
	@echo "Demo 5/5: KV Cache with Device Tensors"
	@echo "------------------------------------------------------------------------"
	RUN_ALL_DEMOS_LEVELS='$(RUN_ALL_DEMOS_LEVELS)' $(VENV_DIR)/bin/python examples/kv_cache_device_tensors.py
	@echo ""
	@echo "========================================================================"
	@echo "All demos completed successfully!"
	@echo "========================================================================"

run-all-demos: run-all-demo

all: setup dev test
	@echo "All tasks complete"
