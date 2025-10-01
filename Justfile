set shell := ["bash", "-cu"]

setup:
curl https://sh.rustup.rs -sSf | sh -s -- -y
rustup target add wasm32-unknown-unknown
cargo install cargo-watch cargo-audit cargo-deny maturin wasm-bindgen-cli wasm-pack --locked

run:
cd rust && cargo run -p service

test:
cd rust && cargo test --all --all-features

pywheel:
cd rust/pycore && maturin build --release

wasm:
cd rust/wasm-demo && wasm-pack build --target web --out-dir pkg

service:
cd rust/service && cargo run
