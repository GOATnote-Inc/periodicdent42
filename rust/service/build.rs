fn main() {
    let proto_dir = std::path::Path::new("proto");
    tonic_build::configure()
        .build_server(true)
        .compile_well_known_types(true)
        .out_dir(std::env::var("OUT_DIR").unwrap())
        .compile(&["proto/periodic/experiment.proto"], &[proto_dir])
        .expect("compile protos");
}
