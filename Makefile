PYTHON ?= python
PIP ?= pip
VENV ?= .venv

.PHONY: help venv install lint test test-full bench-transformer bench-resnet bench-gpt2 bench-long-context bench-unet verify clean

help:
	@echo "Targets:"
	@echo "  make venv                - create virtual environment ($(VENV))"
	@echo "  make install             - install dev dependencies into active env"
	@echo "  make lint                - run ruff lint checks"
	@echo "  make test                - run fast unit tests"
	@echo "  make test-full           - run full pytest suite (unit + integration)"
	@echo "  make bench-transformer   - run transformer benchmark"
	@echo "  make bench-resnet        - run resnet benchmark"
	@echo "  make bench-gpt2          - run GPT-style benchmark"
	@echo "  make bench-long-context  - run long-context experiment sweep"
	@echo "  make bench-unet          - run diffusion U-Net benchmark"

venv:
	$(PYTHON) -m venv $(VENV)

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-dev.txt

lint:
	ruff check .

test:
	pytest tests/unit

test-full:
	pytest

bench-transformer:
	$(PYTHON) -m benchmarks.mem_vs_time_transformer --modes naive checkpoint tiny --trials 3 --export results/transformer.json

bench-resnet:
	$(PYTHON) -m benchmarks.mem_vs_time_resnet --modes naive tiny --trials 3 --export results/resnet.json

bench-gpt2:
	$(PYTHON) experiments/transformers/gpt2_mem_bench.py --modes naive checkpoint tiny --trials 2 --export results/gpt2.json

bench-long-context:
	$(PYTHON) experiments/transformers/long_context.py --seq-lens 256 512 1024 --trials 2 --export results/long_context.json

bench-unet:
	$(PYTHON) experiments/diffusion/unet_mem_bench.py --trials 2 --export results/unet.json

verify:
	$(PYTHON) scripts/run_full_test.py

clean:
	rm -rf $(VENV) .pytest_cache .ruff_cache results/*.json
