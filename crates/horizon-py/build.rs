fn main() {
    // Cargo tests and local binaries link against macOS's Python framework.
    // The framework path is not in dyld's default search list, so preserve the
    // rpath supplied by the interpreter configuration. Maturin builds extension
    // modules with dynamic lookup instead, where this helper is a no-op.
    pyo3_build_config::add_python_framework_link_args();
}
