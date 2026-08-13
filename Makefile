.PHONY: build check test test-python fault bench demo fmt lint

build:
	cargo build --workspace --all-features

check:
	cargo check --workspace --all-features

fmt:
	cargo fmt --all -- --check

lint:
	cargo clippy --workspace --all-targets --all-features -- -D warnings

test:
	cargo test --workspace --all-features

test-python:
	PYTHONPATH=python:. python3 -m unittest discover -s tests -p 'test_*.py' -v

fault:
	PYTHONPATH=python python3 tests/fault_injection.py

bench:
	python3 benchmarks/horizonbench/run.py --output-dir results/horizonbench

demo:
	cargo run -p horizon-cli -- --db horizon-demo.db demo
