// Most important functions/imports are
// - read_config_file
// - stitch2d_no_fusing
// - fuse_cli::fuse
// - StitchConfig
// with these, you should be able to seperately perform stitching and fusing

pub use fuse::{fuse_2d, fuse_3d_float, FuseMode};
pub use image::{
    read_dcm, read_dcm_headers, read_image_2d, read_tiff, read_tiff_headers, save_as_dcm_8,
    save_as_tiff_float, save_image_2d, Image3D,
};
use rayon::prelude::*;
use serde_json::*;
pub use std::path::{Path, PathBuf};
use stitch2d::IBox2D;
use stitch3d::IBox3D;
use transpose::transpose_inplace;

mod fuse;
pub mod image;
mod stitch2d;
mod stitch3d;
pub mod fuse_cli;

pub use fuse_cli::fuse;
use stitch2d::Stitch2DResult;

use crate::image::Image2D;

#[derive(PartialEq, Clone, Copy, Debug)]
pub enum StitchMode {
    TwoD,
    ThreeD,
}

#[derive(Clone)]
pub struct StitchConfig {
    pub version: String,
    pub mode: StitchMode,
    pub overlap_ratio: (f32, f32, f32),
    pub correlation_threshold: f32,
    pub relative_error_threshold: f32,
    pub absolute_error_threshold: f32,
    pub check_peaks: usize,
    pub dimension_mask: (bool, bool, bool),
    pub fuse_mode: FuseMode,
    pub no_fuse: bool,
    pub output_path: PathBuf,
    pub alignment_file: Option<PathBuf>,
    pub tile_paths: Vec<PathBuf>,
    pub tile_layout: Vec<IBox3D>,
    pub copy_files: bool,
    pub use_phase_correlation: bool,
    pub use_prior: bool,
    pub prior_sigmas: (f32, f32, f32),
    pub merge_subgraphs: bool,
    pub save_float: bool,
}

impl StitchConfig {
    pub fn new() -> Self {
        StitchConfig {
            version: "1.0".to_string(),
            mode: StitchMode::TwoD,
            save_float: false,
            overlap_ratio: (0.2, 0.2, 0.2),
            correlation_threshold: 0.3,
            relative_error_threshold: 2.5,
            absolute_error_threshold: 3.5,
            check_peaks: 5,
            dimension_mask: (true, true, true),
            fuse_mode: FuseMode::Linear,
            no_fuse: false,
            output_path: PathBuf::new(),
            alignment_file: None,
            tile_paths: vec![],
            tile_layout: vec![],
            copy_files: false,
            use_phase_correlation: true,
            use_prior: false,
            prior_sigmas: (10.0, 10.0, 10.0),
            merge_subgraphs: true,
        }
    }
}

pub fn stitch2d_no_fusing(config: StitchConfig) -> (Vec<Image2D>, Stitch2DResult) {
println!("Reading files...");
    let start = std::time::Instant::now();
    let images = config
        .tile_paths
        .into_par_iter()
        .map(|path| {
            if !path.exists() {
                panic!("File does not exist: {:?}", path);
            }
            read_image_2d(&path)
        })
        .collect::<Vec<_>>();

    println!("Time to read files: {:?}", start.elapsed());

    let mut stitched_result = None;

    if !config.alignment_file.is_none() && config.alignment_file.clone().unwrap().exists() {
        let alignment_file = config.alignment_file.unwrap();
        let json_str = std::fs::read_to_string(alignment_file).unwrap();
        let res = serde_json::from_str(&json_str).unwrap();
        println!("Alignment file loaded");
        stitched_result = Some(res);
    }

    if stitched_result.is_none() {
        println!("Aligning images...");
        let start = std::time::Instant::now();
        let dim_mask = (config.dimension_mask.0, config.dimension_mask.1);
        let tile_layout = config
            .tile_layout
            .iter()
            .map(|layout| IBox2D::new(layout.x, layout.y, layout.width, layout.height))
            .collect::<Vec<_>>();
        let result = stitch2d::stitch(
            &images,
            &tile_layout,
            (config.overlap_ratio.0, config.overlap_ratio.1),
            config.check_peaks,
            config.correlation_threshold,
            config.relative_error_threshold,
            config.absolute_error_threshold,
            dim_mask,
            config.use_phase_correlation,
            config.use_prior,
            (config.prior_sigmas.0, config.prior_sigmas.1),
            config.merge_subgraphs,
        );
        println!("Time to find alignment: {:?}", start.elapsed());
        stitched_result = Some(result);

        let json = json!(stitched_result);
        let json_str = json.to_string();
        let json_path = config.output_path.join("align_values.json");
        std::fs::write(json_path.clone(), json_str).unwrap();
        println!("Alignment values saved to: {:?}", json_path);
    }

    let stitched_result = stitched_result.unwrap();


    return (images, stitched_result)
}

pub fn read_config_file(path: &Path) -> StitchConfig {
    let base_path = path.parent();
    let json_str = std::fs::read_to_string(path).unwrap();

    let mut config = StitchConfig::new();

    let json: Value = serde_json::from_str(&json_str).unwrap();

    // Check version
    if !json.get("version").is_none() {
        config.version = json["version"].as_str().unwrap().to_string();
        println!("Version: {}", config.version);
    }

    // Get 3d or 2d
    if json.get("mode").is_none() {
        panic!("No mode specified");
    }

    let mode = json["mode"].as_str().unwrap();
    match mode {
        "2d" => {
            config.mode = StitchMode::TwoD;
        }
        "3d" => {
            config.mode = StitchMode::ThreeD;
        }
        _ => {
            panic!("Invalid mode");
        }
    }

    println!("Mode: {:?}", config.mode);

    // Check overlap ratio
    if !json.get("overlap_ratio").is_none() {
        // Check if single value or array
        if json["overlap_ratio"].is_array() {
            let arr = json["overlap_ratio"].as_array().unwrap();
            if arr.len() == 2 {
                config.overlap_ratio.0 = arr[0].as_f64().unwrap() as f32;
                config.overlap_ratio.1 = arr[1].as_f64().unwrap() as f32;
            } else if arr.len() == 3 {
                config.overlap_ratio.0 = arr[0].as_f64().unwrap() as f32;
                config.overlap_ratio.1 = arr[1].as_f64().unwrap() as f32;
                config.overlap_ratio.2 = arr[2].as_f64().unwrap() as f32;
            } else {
                panic!("Invalid overlap ratio");
            }
        } else {
            let val = json["overlap_ratio"].as_f64().unwrap() as f32;
            config.overlap_ratio = (val, val, val);
        }

        println!("Overlap ratio: {:?}", config.overlap_ratio);
    }

    // Check correlation threshold
    if !json.get("correlation_threshold").is_none() {
        config.correlation_threshold = json["correlation_threshold"].as_f64().unwrap() as f32;
        println!("Correlation threshold: {}", config.correlation_threshold);
    }

    // Check check peaks
    if !json.get("check_peaks").is_none() {
        config.check_peaks = json["check_peaks"].as_u64().unwrap() as usize;
        println!("Check peaks: {}", config.check_peaks);
    }

    // Check save float
    if !json.get("save_float").is_none() {
        config.save_float = json["save_float"].as_bool().unwrap();
        println!("Save float: {}", config.save_float);
    }

    // Check dimension mask
    if !json.get("dimension_mask").is_none() {
        let mask = json["dimension_mask"].as_array().unwrap();
        let mut x = true;
        let mut y = true;
        let mut z = true;
        if mask.len() == 2 {
            x = mask[0].as_bool().unwrap();
            y = mask[1].as_bool().unwrap();
        } else if mask.len() == 3 {
            x = mask[0].as_bool().unwrap();
            y = mask[1].as_bool().unwrap();
            z = mask[2].as_bool().unwrap();
        } else {
            println!("Invalid dimension mask... using default");
        }

        config.dimension_mask = (x, y, z);

        println!("Dimension mask: {:?}", config.dimension_mask);
    }

    // Check fuse mode
    if !json.get("fuse_mode").is_none() {
        let mode = json["fuse_mode"].as_str().unwrap();
        match mode {
            "average" => {
                config.fuse_mode = FuseMode::Average;
            }
            "max" => {
                config.fuse_mode = FuseMode::Max;
            }
            "min" => {
                config.fuse_mode = FuseMode::Min;
            }
            "overwrite" => {
                config.fuse_mode = FuseMode::Overwrite;
            }
            "linear" => {
                config.fuse_mode = FuseMode::Linear;
            }
            "overwrite-prioritize-center" => {
                config.fuse_mode = FuseMode::OverwritePrioritizeCenter;
            }
            _ => {
                panic!("Invalid fuse mode");
            }
        }

        println!("Fuse mode: {:?}", config.fuse_mode);
    }

    if !json.get("use_phase_correlation").is_none() {
        config.use_phase_correlation = json["use_phase_correlation"].as_bool().unwrap();
        println!("Use phase correlation: {}", config.use_phase_correlation);
    }

    // Check no fuse
    if !json.get("no_fuse").is_none() {
        config.no_fuse = json["no_fuse"].as_bool().unwrap();

        println!("No fuse: {}", config.no_fuse);
    }

    // Check prior
    if !json.get("use_prior").is_none() {
        config.use_prior = json["use_prior"].as_bool().unwrap();
        println!("Use prior: {}", config.use_prior);
    }

    // Check merge
    if !json.get("merge_subgraphs").is_none() {
        config.merge_subgraphs = json["merge_subgraphs"].as_bool().unwrap();
        println!("Merge subgraphs: {}", config.merge_subgraphs);
    }

    // Check prior sigmas
    if !json.get("prior_sigma").is_none() {
        if json["prior_sigma"].is_array() {
            let sigmas = json["prior_sigma"].as_array().unwrap();
            if sigmas.len() == 2 {
                config.prior_sigmas.0 = sigmas[0].as_f64().unwrap() as f32;
                config.prior_sigmas.1 = sigmas[1].as_f64().unwrap() as f32;
            } else if sigmas.len() == 3 {
                config.prior_sigmas.0 = sigmas[0].as_f64().unwrap() as f32;
                config.prior_sigmas.1 = sigmas[1].as_f64().unwrap() as f32;
                config.prior_sigmas.2 = sigmas[2].as_f64().unwrap() as f32;
            } else {
                panic!("Invalid prior sigmas");
            }
        } else {
            let val = json["prior_sigma"].as_f64().unwrap() as f32;
            config.prior_sigmas = (val, val, val);
        }
        println!("Prior sigmas: {:?}", config.prior_sigmas);
    }

    // Check absolute error threshold
    if !json.get("absolute_error_threshold").is_none() {
        config.absolute_error_threshold = json["absolute_error_threshold"].as_f64().unwrap() as f32;
        println!(
            "Absolute error threshold: {}",
            config.absolute_error_threshold
        );
    }

    // Check relative error threshold
    if !json.get("relative_error_threshold").is_none() {
        config.relative_error_threshold = json["relative_error_threshold"].as_f64().unwrap() as f32;
        println!(
            "Relative error threshold: {}",
            config.relative_error_threshold
        );
    }

    // Check output path
    if !json.get("output_path").is_none() {
        config.output_path = PathBuf::from(json["output_path"].as_str().unwrap());

        if !config.output_path.is_absolute() {
            if base_path.is_none() {
                panic!("Invalid base path");
            }

            config.output_path = base_path.unwrap().join(&config.output_path);
        }
    } else {
        if base_path.is_none() {
            panic!("Invalid base path");
        }
        config.output_path = base_path.unwrap().join("output");
    }

    // Check alignment file
    if !json.get("alignment_file").is_none() {
        let mut path = PathBuf::from(json["alignment_file"].as_str().unwrap());
        if !path.is_absolute() {
            if base_path.is_none() {
                panic!("Invalid base path");
            }

            path = base_path.unwrap().join(&path);
        }
        config.alignment_file = Some(path);
    }

    // Check tile paths
    if json.get("tile_paths").is_none() {
        panic!("No tile paths specified");
    }

    if !json.get("tiles").is_none() {
        let tiles = json["tiles"].as_array().unwrap();
        for tile in tiles {
            let tile = tile.as_object().unwrap();
            let mut path = PathBuf::from(tile["path"].as_str().unwrap());
            if !path.is_absolute() {
                if base_path.is_none() {
                    panic!("Invalid base path");
                }

                path = base_path.unwrap().join(&path);
            }
            let mut temp = IBox3D::new(0, 0, 0, 1, 1, 1);

            if !tile.get("box").is_none() {
                let box_arr = tile["box"].as_array().unwrap();
                if box_arr.len() == 2 {
                    temp.x = box_arr[0].as_i64().unwrap();
                    temp.y = box_arr[1].as_i64().unwrap();
                } else if box_arr.len() == 3 {
                    temp.x = box_arr[0].as_i64().unwrap();
                    temp.y = box_arr[1].as_i64().unwrap();
                    temp.z = box_arr[2].as_i64().unwrap();
                } else if box_arr.len() == 4 {
                    temp.x = box_arr[0].as_i64().unwrap();
                    temp.y = box_arr[1].as_i64().unwrap();
                    temp.width = box_arr[2].as_i64().unwrap();
                    temp.height = box_arr[3].as_i64().unwrap();
                } else if box_arr.len() == 6 {
                    temp.x = box_arr[0].as_i64().unwrap();
                    temp.y = box_arr[1].as_i64().unwrap();
                    temp.z = box_arr[2].as_i64().unwrap();
                    temp.width = box_arr[3].as_i64().unwrap();
                    temp.height = box_arr[4].as_i64().unwrap();
                    temp.depth = box_arr[5].as_i64().unwrap();
                } else {
                    panic!("Invalid tile layout");
                }
            } else {
                let x = tile.get("x").unwrap().as_i64().unwrap();
                let y = tile.get("y").unwrap().as_i64().unwrap();
                if !tile.get("z").is_none() {
                    let z = tile.get("z").unwrap().as_i64().unwrap();
                    temp.x = x;
                    temp.y = y;
                    temp.z = z;
                } else {
                    temp.x = x;
                    temp.y = y;
                }

                if !tile.get("width").is_none() {
                    temp.width = tile.get("width").unwrap().as_i64().unwrap();
                }

                if !tile.get("height").is_none() {
                    temp.height = tile.get("height").unwrap().as_i64().unwrap();
                }

                if !tile.get("depth").is_none() {
                    temp.depth = tile.get("depth").unwrap().as_i64().unwrap();
                }
            }

            config.tile_paths.push(path);
            config.tile_layout.push(temp);
        }
    } else {
        let tile_paths = json["tile_paths"].as_array().unwrap();
        for tile_path in tile_paths {
            let mut path = PathBuf::from(tile_path.as_str().unwrap());
            if !path.is_absolute() {
                if base_path.is_none() {
                    panic!("Invalid base path");
                }

                path = base_path.unwrap().join(&path);
            }
            config.tile_paths.push(path);
        }

        // Check tile layout
        if json.get("tile_layout").is_none() {
            panic!("No tile layout specified");
        }

        let tile_layout = json["tile_layout"].as_array().unwrap();
        for layout in tile_layout {
            let arr = layout.as_array().unwrap();
            let mut temp = IBox3D::new(0, 0, 0, 1, 1, 1);
            if arr.len() == 2 {
                temp.x = arr[0].as_i64().unwrap();
                temp.y = arr[1].as_i64().unwrap();
            } else if arr.len() == 3 {
                temp.x = arr[0].as_i64().unwrap();
                temp.y = arr[1].as_i64().unwrap();
                temp.z = arr[2].as_i64().unwrap();
            } else if arr.len() == 4 {
                temp.x = arr[0].as_i64().unwrap();
                temp.y = arr[1].as_i64().unwrap();
                temp.width = arr[2].as_i64().unwrap();
                temp.height = arr[3].as_i64().unwrap();
            } else if arr.len() == 6 {
                temp.x = arr[0].as_i64().unwrap();
                temp.y = arr[1].as_i64().unwrap();
                temp.z = arr[2].as_i64().unwrap();
                temp.width = arr[3].as_i64().unwrap();
                temp.height = arr[4].as_i64().unwrap();
                temp.depth = arr[5].as_i64().unwrap();
            } else {
                panic!("Invalid tile layout");
            }

            config.tile_layout.push(temp);
        }

        if config.tile_paths.len() != config.tile_layout.len() {
            panic!("Tile paths and layout do not match length!");
        }
    }

    config
}

pub fn stitch_3d(config: StitchConfig) {
    let mut tile_paths = config.tile_paths.clone();
    let mut temp_dir = PathBuf::new();
    if config.copy_files {
        println!("Copying files to temp directory...");
        temp_dir = std::env::temp_dir();
        temp_dir = temp_dir.join("stitch3d");
        let random: u32 = rand::random();
        temp_dir = temp_dir.join(format!("temp_{}", random));
        if !temp_dir.exists() {
            std::fs::create_dir_all(&temp_dir).unwrap();
        }

        tile_paths = tile_paths
            .iter()
            .enumerate()
            .map(|(i, path)| {
                let file_name = path.file_name().unwrap();
                let temp_path = temp_dir.join(file_name);
                std::fs::copy(path, temp_path.clone()).unwrap();
                println!(
                    "[{}/{}] Copied file: {:?} to {:?}",
                    i + 1,
                    tile_paths.len(),
                    path,
                    temp_path
                );
                temp_path
            })
            .collect::<Vec<_>>();
    }
    println!("Reading files for size information...");
    let start = std::time::Instant::now();
    let images = tile_paths
        .into_par_iter()
        .map(|path| {
            // Check if it exists
            if !path.exists() {
                panic!("File does not exist: {:?}", path);
            }
            if path.extension().unwrap() == "dcm" {
                read_dcm_headers(&path)
            } else {
                read_tiff_headers(&path)
            }
        })
        .collect::<Vec<_>>();

    // Print file sizes
    images.iter().for_each(|image| {
        println!(
            "Width: {}, Height: {}, Depth: {} Min: {} Max: {}",
            image.width, image.height, image.depth, image.min, image.max
        );
    });

    println!("Time to read files: {:?}", start.elapsed());

    let mut stitched_result = None;

    if !config.alignment_file.is_none() && config.alignment_file.clone().unwrap().exists() {
        let alignment_file = config.alignment_file.unwrap();
        let json_str = std::fs::read_to_string(alignment_file).unwrap();
        let res = serde_json::from_str(&json_str).unwrap();
        println!("Alignment file loaded");
        stitched_result = Some(res);
    }

    if stitched_result.is_none() {
        println!("Aligning images...");
        let start = std::time::Instant::now();
        let result = stitch3d::stitch(
            &images,
            &config.tile_layout,
            config.overlap_ratio,
            config.check_peaks,
            config.correlation_threshold,
            config.relative_error_threshold,
            config.absolute_error_threshold,
            config.dimension_mask,
            config.use_phase_correlation,
            config.use_prior,
            config.prior_sigmas,
            config.merge_subgraphs,
        );
        println!("Time to find alignment: {:?}", start.elapsed());
        stitched_result = Some(result);

        let json = json!(stitched_result);
        let json_str = json.to_string();
        let json_path = config.output_path.join("align_values.json");
        std::fs::write(json_path.clone(), json_str).unwrap();
        println!("Alignment values saved to: {:?}", json_path);
    }

    let stitched_result = stitched_result.unwrap();

    if config.no_fuse {
        return;
    }

    println!("Fusing images...");
    let start = std::time::Instant::now();

    stitched_result
        .offsets
        .iter()
        .enumerate()
        .for_each(|(i, offset)| {
            let fused_image = fuse_3d_float(
                &images,
                &stitched_result.subgraphs[i],
                offset,
                config.fuse_mode,
            );

            let output_file = format!("fused_{}.tiff", i);
            let buf = config.output_path.join(output_file);
            save_as_tiff_float(&buf, &fused_image);
        });

    println!("Time to fuse images: {:?}", start.elapsed());

    if config.copy_files {
        // Delete temp directory
        std::fs::remove_dir_all(temp_dir).unwrap();
    }
}

pub fn stitch_2d(config: StitchConfig) {
    
    let (images, stitched_result) = stitch2d_no_fusing(config.clone());

    if config.no_fuse {
        return;
    }

    println!("Fusing images...");
    let start = std::time::Instant::now();

    stitched_result
        .offsets
        .iter()
        .enumerate()
        .for_each(|(i, offset)| {
            let fused_image = fuse_2d(
                &images,
                &stitched_result.subgraphs[i],
                offset,
                config.fuse_mode,
            );
            let output_file = format!("fused_{}.tiff", i);
            let buf = config.output_path.join(output_file);
            save_image_2d(&buf, &fused_image);
        });

    println!("Time to fuse images: {:?}", start.elapsed());
}

pub fn normalize_brightness(config: &StitchConfig) {
    let tile_paths = config.tile_paths.clone();
    // New directory /normalized
    let normalized_path = config.output_path.join("normalized");
    if !normalized_path.exists() {
        std::fs::create_dir_all(&normalized_path).unwrap();
    }

    // Get mean brightness
    let mean_brightness = 1.0;

    println!("Mean brightness: {}", mean_brightness);

    // Normalize brightness
    tile_paths.iter().enumerate().for_each(|(_i, path)| {
        let image = if path.extension().unwrap() == "dcm" {
            read_dcm(&path)
        } else {
            read_tiff(&path)
        };

        // let brightness = image.data.par_chunks(image.width as usize * image.height as usize).map(|chunk| {
        //     chunk.iter().map(|x| if x.is_finite() { x } else { &0.0 }).sum::<f32>()
        // }).sum::<f32>() / image.data.len() as f32;

        let brightness = otsu_threshold(&image.data);

        let normalized = image
            .data
            .iter()
            .map(|x| x * mean_brightness / brightness)
            .collect::<Vec<_>>();
        let normalized_image = Image3D {
            width: image.width,
            height: image.height,
            depth: image.depth,
            data: normalized,
            min: 0.0,
            max: 0.0,
        };

        let output_file = normalized_path.join(path.file_name().unwrap());

        save_as_tiff_float(&output_file, &normalized_image);

        println!("Normalized file saved to: {:?}", output_file);
    });
}

pub fn otsu_threshold(data: &[f32]) -> f32 {
    let mut data = data.to_vec();
    data.retain(|&x| x.is_finite());

    if data.len() == 0 {
        return f32::NEG_INFINITY;
    }

    let max_value: f32 = data.iter().fold(data[0], |acc, &x| acc.max(x)) + 1.0;
    let min_value: f32 = data.iter().fold(data[0], |acc, &x| acc.min(x));

    let diff = max_value - min_value;

    let hist_size = 512;
    let mut hist = vec![0; hist_size];
    for i in 0..data.len() {
        let index = ((data[i] - min_value) / diff * hist_size as f32)
            .floor()
            .clamp(0.0, (hist_size - 1) as f32) as usize;
        hist[index] += 1;
    }

    // Sum histogram, weighted
    let mut sum = 0.0;
    for i in 0..hist_size {
        sum += (hist[i] as f32) * ((i as f32 / hist_size as f32) * diff + min_value);
    }

    let mut background_sum = 0.0;
    let mut background_weight = 0.0;

    let mut max_variance = 0.0;
    let mut best_threshold = 0.0;

    for i in 0..hist_size {
        let threshold = (i as f32 / hist_size as f32 + 0.0 / hist_size as f32) * diff + min_value;
        background_weight += hist[i] as f32;
        if background_weight == 0.0 {
            continue;
        }

        let foreground_weight = data.len() as f32 - background_weight;
        if foreground_weight == 0.0 {
            break;
        }

        background_sum += threshold * hist[i] as f32;
        let foreground_sum = sum - background_sum;

        //  println!("Background Sum: {}, Background Weight: {}, Foreground Sum: {}, Foreground Weight: {}, hist: {}", background_sum, background_weight, foreground_sum, foreground_weight, hist[i]);

        let background_mean = background_sum / background_weight;
        let foreground_mean = foreground_sum / foreground_weight;

        let mean_diff_sq = (background_mean - foreground_mean).powi(2);
        let variance = background_weight * foreground_weight * mean_diff_sq;

        if variance > max_variance {
            max_variance = variance;
            best_threshold = threshold;
        }
    }

    best_threshold
}

pub fn normalize2(path: &Path, dim_mask: (bool, bool, bool)) {
    let blurred_file_path = path.with_extension("blurred.tif");
    // check if exists
    let blurred_file = if !blurred_file_path.exists() {
        println!("Creating blurred file");
        let mut image = read_tiff(path);
        let kw = 10;
        let kh = kw;
        let kd = kw;

        let kernel = generate_guassian_kernel_3d(kw, 1, 1, 20.0);

        for i in 0..5 {
            if dim_mask.0 {
                // Process x
                image
                    .data
                    .par_chunks_exact_mut(image.width)
                    .for_each(|chunk| {
                        let copy = chunk.to_vec();
                        for x in 0..image.width {
                            let mut sum = 0.0;
                            let mut weight_sum = 0.0;
                            for kx in 0..kw {
                                let x2 = x as i32 + kx as i32 - kw as i32 / 2;
                                if x2 >= 0
                                    && x2 < image.width as i32
                                    && copy[x2 as usize].is_finite()
                                {
                                    let val = copy[x2 as usize];
                                    let weight = kernel[kx];
                                    sum += val * weight;
                                    weight_sum += weight;
                                }
                            }
                            chunk[x] = sum / weight_sum;
                        }
                    });

                println!("Processed x");
            }

            if dim_mask.1 {
                // Process y
                image
                    .data
                    .par_chunks_exact_mut(image.width * image.height)
                    .for_each(|chunk| {
                        let mut copy = vec![0.0; chunk.len()];
                        transpose::transpose(chunk, &mut copy, image.width, image.height);
                        for x in 0..image.width {
                            for y in 0..image.height {
                                let mut sum = 0.0;
                                let mut weight_sum = 0.0;
                                for ky in 0..kh {
                                    let y2 = y as i32 + ky as i32 - kh as i32 / 2;
                                    if y2 >= 0
                                        && y2 < image.height as i32
                                        && copy[x * image.height + y2 as usize].is_finite()
                                    {
                                        let val = copy[x * image.height + y2 as usize];
                                        let weight = kernel[ky];
                                        sum += val * weight;
                                        weight_sum += weight;
                                    }
                                }
                                chunk[y * image.width + x] = sum / weight_sum;
                            }
                        }
                    });

                println!("Processed y");
            }

            // Process z
            if dim_mask.2 {
                let mut scratch = vec![0.0; image.depth];

                for x in 0..image.width {
                    for y in 0..image.height {
                        for z in 0..image.depth {
                            scratch[z] =
                                image.data[z * image.width * image.height + y * image.width + x];
                        }

                        for z in 0..image.depth {
                            let mut sum = 0.0;
                            let mut weight_sum = 0.0;
                            for kz in 0..kd {
                                let z2 = z as i32 + kz as i32 - kd as i32 / 2;
                                if z2 >= 0
                                    && z2 < image.depth as i32
                                    && scratch[z2 as usize].is_finite()
                                {
                                    let val = scratch[z2 as usize];
                                    let weight = kernel[kz];
                                    sum += val * weight;
                                    weight_sum += weight;
                                }
                            }
                            image.data[z * image.width * image.height + y * image.width + x] =
                                sum / weight_sum;
                        }
                    }
                }
                println!("Processed z");
            }

            println!("Iteration: {}", i);
        }
        save_as_tiff_float(&blurred_file_path, &image);

        image
    } else {
        println!("Reading blurred file");
        read_tiff(&blurred_file_path)
    };

    //let blurred_file_path2 = path.with_extension("blurred2.tif");

    let mut max: f32 = 0.0;
    blurred_file.data.iter().for_each(|x| {
        if x.is_finite() {
            max = max.max(*x);
        }
    });
    // Threshold
    //let threshold = otsu_threshold(&blurred_file.data);
    //println!("Threshold: {}", threshold);

    // Normalize
    let mut image = read_tiff(path);

    image
        .data
        .iter_mut()
        .zip(blurred_file.data.iter())
        .for_each(|(x, y)| {
            if y.is_finite() && *y > f32::EPSILON {
                *x = *x * max / y;
            }
        });

    let output_file = path.with_extension("normalized.tif");
    save_as_tiff_float(&output_file, &image);
}

pub fn generate_guassian_kernel_3d(
    width: usize,
    height: usize,
    depth: usize,
    sigma: f32,
) -> Vec<f32> {
    let mut kernel = vec![0.0; width * height * depth];
    let sigma_sqr = sigma * sigma;
    let half_width = width / 2;
    let half_height = height / 2;
    let half_depth = depth / 2;
    let mut sum = 0.0;
    for zi in 0..depth {
        for yi in 0..height {
            for xi in 0..width {
                let x = xi as f32 - half_width as f32;
                let y = yi as f32 - half_height as f32;
                let z = zi as f32 - half_depth as f32;
                let value = (-(x * x + y * y + z * z) / (2.0 * sigma_sqr)).exp();
                kernel[zi * width * height + yi * width + xi] = value;
                sum += value;
            }
        }
    }
    for i in 0..kernel.len() {
        kernel[i] /= sum;
    }
    kernel
}
