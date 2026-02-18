use stitch::*;
fn main() {
    let args = std::env::args().collect::<Vec<_>>();

    if args.len() < 2 {
        println!("Usage: cmd config_file.json or cmd normalize file.tiff or cmd fuse fuse_config.json");
        return;
    }

    if args[1] == "fuse" {
        if args.len() < 5 {
            println!("Usage: cmd fuse fuse_config.json config_file.json out.tiff");
            return;
        }
        let config_path = Path::new(&args[3]);
        if !config_path.exists() {
            println!("Config file does not exist");
            return;
        }

        let fuse_config_path = Path::new(&args[2]);
        if !fuse_config_path.exists() {
            println!("Fuse config file does not exist");
            return;
        }

        let out_path = Path::new(&args[4]).to_path_buf();

        let fused_image = fuse_cli::fuse(&config_path,&fuse_config_path);
        
        save_image_2d(&out_path, &fused_image);
        return;
    }

    if args[1] == "normalize" {
        if args.len() < 3 {
            println!("Usage: cmd normalize file.tiff");
            return;
        }
        let mut dim_mask = (true, true, true);
        if args.len() > 4 {
            dim_mask.0 = args[3].parse::<bool>().unwrap();
            dim_mask.1 = args[4].parse::<bool>().unwrap();
            dim_mask.2 = args[5].parse::<bool>().unwrap();
        } else if args.len() > 3 {
            dim_mask.0 = args[3].parse::<bool>().unwrap();
            dim_mask.1 = args[4].parse::<bool>().unwrap();
        }

        normalize2(Path::new(&args[2]), dim_mask);
        return;
    }

    let config_path = Path::new(&args[1]);
    if !config_path.exists() {
        println!("Config file does not exist");
        return;
    }
    let mut config = read_config_file(config_path);
    let mut i = 2;
    let mut normalize = false;
    while i < args.len() {
        match args[i].as_str() {
            "-o" => {
                i += 1;
                config.output_path = PathBuf::from(&args[i]);
            }
            "--save-float" => {
                config.save_float = true;
            }
            "--no-fuse" => {
                config.no_fuse = true;
            }
            "--copy" => {
                config.copy_files = true;
            }
            "--normalize" => {
                normalize = true;
            }
            "--fuse-mode" => {
                i += 1;
                match args[i].as_str() {
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
                        println!("Invalid fuse mode: {}", args[i]);
                        return;
                    }
                }
            }
            _ => {
                println!("Invalid argument: {}", args[i]);
                return;
            }
        }
        i += 1;
    }

    // Make output directory
    if !config.output_path.exists() {
        std::fs::create_dir_all(&config.output_path).unwrap();
    }

    if normalize {
        normalize_brightness(&config);
        return;
    }

    match config.mode {
        StitchMode::TwoD => {
            stitch_2d(config);
        }
        StitchMode::ThreeD => {
            stitch_3d(config);
        }
    }
}
