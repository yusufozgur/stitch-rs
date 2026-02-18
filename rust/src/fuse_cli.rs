use std::fs;
use std::path::Path;

use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{
    fuse::fuse_2d,
    image::{Image2D, read_image_2d},
    read_config_file,
    stitch2d::Stitch2DResult,
};

pub fn fuse(config_path: &Path, fuse_config_path: &Path) -> Image2D {
    let stitched_result: Stitch2DResult = load_stitch2d_result(&fuse_config_path);

    if stitched_result.subgraphs.len() != 1 {
        panic!(
            "Expected 1 subgraph, but found {}. Check your correlation_threshold or overlap.",
            stitched_result.subgraphs.len()
        );
    }

    let stitch_config = read_config_file(config_path);

    let images = stitch_config
        .tile_paths
        .into_par_iter()
        .map(|path| {
            if !path.exists() {
                panic!("File does not exist: {:?}", path);
            }
            read_image_2d(&path)
        })
        .collect::<Vec<_>>();

    let fused_image = fuse_2d(
        &images, // I can get this out of the stitch config
        &stitched_result.subgraphs[0],
        &stitched_result.offsets[0],
        crate::fuse::FuseMode::Linear,
    );
    return fused_image;
}

fn load_stitch2d_result(config_path: &Path) -> Stitch2DResult {
    let config = fs::read_to_string(config_path).unwrap();
    // 2. Parse the JSON string back into the Rust Option<Stitch2DResult>
    let stitch2d_result: Stitch2DResult = serde_json::from_str(&config).unwrap();
    return stitch2d_result;
}
