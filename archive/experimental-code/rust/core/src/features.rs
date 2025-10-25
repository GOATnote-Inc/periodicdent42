pub fn enabled(name: &str) -> bool {
    match name {
        "instrument_control" => cfg!(feature = "instrument_control"),
        "gpu" => cfg!(feature = "gpu"),
        "wasm" => cfg!(feature = "wasm"),
        "pybind" => cfg!(feature = "pybind"),
        "postgres" => cfg!(feature = "postgres"),
        "arrow" => cfg!(feature = "arrow"),
        "polars" => cfg!(feature = "polars"),
        _ => false,
    }
}
