.PHONY: build check test test-python fault bench demo fmt lint

build:
	cargo build --workspace

check:
	cargo check --workspace

fmt:
	cargo fmt --all -- --check

lint:
	cargo clippy --workspace --all-targets -- -D warnings

test:
	cargo test --workspace

test-python:
	PYTHONPATH=python:. python3 -m unittest discover -s tests -p 'test_*.py' -v

fault:
	PYTHONPATH=python python3 tests/fault_injection.py

bench:
	python3 benchmarks/horizonbench/run.py --output-dir results/horizonbench

demo:
	cargo run -p horizon-cli -- --db horizon-demo.db demo
