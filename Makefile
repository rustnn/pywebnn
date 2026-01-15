.PHONY: help setup build test clean dev install lint fmt check all

# Python version to use (defaults to python3 in PATH)
PYTHON ?= python3
VENV_DIR = .venv
VENV_ACTIVATE = $(VENV_DIR)/bin/activate

help:
	@echo "pywebnn - Python bindings for W3C WebNN"
	@echo ""
	@echo "Available targets:"
	@echo "  setup       - Create virtual environment and install dependencies"
	@echo "  dev         - Install package in development mode (requires setup first)"
	@echo "  build       - Build wheel package"
	@echo "  install     - Install from built wheel"
	@echo "  test        - Run all Python tests"
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

all: setup dev test
	@echo "All tasks complete"
