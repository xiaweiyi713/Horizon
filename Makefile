.PHONY: build check test test-python fault bench bench-trace bench-artifact-workspace bench-preflight demo fmt lint

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

bench-trace:
	PYTHONPATH=python:. python3 -m unittest tests/test_durable_trace_matrix_integration.py -v

bench-artifact-workspace:
	PYTHONPATH=python:. python3 -m unittest tests/test_artifact_workspace_matrix_integration.py -v

bench-preflight:
	PYTHONPATH=python:. python3 -m unittest tests/test_horizonbench_preflight.py -v

demo:
	cargo run -p horizon-cli -- --db horizon-demo.db demo
