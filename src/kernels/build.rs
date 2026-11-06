use std::{fs, path::PathBuf};

pub(crate) fn kernel_sources(directory: &str, extension: &str) -> Vec<PathBuf> {
    fn collect(directory: &std::path::Path, extension: &str, sources: &mut Vec<PathBuf>) {
        for entry in fs::read_dir(directory).expect("failed to read kernel directory") {
            let path = entry.expect("failed to read kernel entry").path();
            if path.is_dir() {
                collect(&path, extension, sources);
            } else if path.extension().is_some_and(|ext| ext == extension) {
                sources.push(path);
            }
        }
    }
    println!("cargo:rerun-if-changed={directory}");
    let mut sources = Vec::new();
    collect(std::path::Path::new(directory), extension, &mut sources);
    sources.sort();
    assert!(!sources.is_empty(), "no .{extension} kernels found");
    for source in &sources {
        println!("cargo:rerun-if-changed={}", source.display());
    }
    sources
}
