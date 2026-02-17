## Image stitching with restricted tile offsets

![sdfsfsfd drawio](https://github.com/user-attachments/assets/e699baf7-4909-498f-ac2a-55436051d302)

Stitches images using a modified version of the algorithm described in [Globally optimal stitching of tiled 3D microscopic image acquisitions](https://doi.org/10.1093%2Fbioinformatics%2Fbtp184).

The algorithm is modified to include a system to prioritize tile offset values within a distribution values to improve robustness for images containing periodic, or minimal features.

## Installation

1. Install [Rust](https://www.rust-lang.org/)
2. Open command line, change working directory to this repository. `cd stitching`
3. Run `cd rust`
4. Run `cargo install --path .`
5. The stitching program is now installed and can be invoked with the `stitch` command

## Basic Usage

Set up a `stitch_config.json` file in the directory containing your images. The basic format is as follows:

```json
{
  "mode": "3d",
  "overlap_ratio": [0.2, 0.2, 0.2],
  "tiles": [
    {
      "path": "tile1.dcm",
      "x": 0,
      "y": 0,
      "z": 0,
      "width": 100,
      "height": 100,
      "depth": 100
    },
    {
      "path": "tile2.dcm",
      "x": 100,
      "y": 0,
      "z": 0,
      "width": 100,
      "height": 100,
      "depth": 100
    },
    {
      "path": "tile2.dcm",
      "x": 200,
      "y": 0,
      "z": 0,
      "width": 100,
      "height": 100,
      "depth": 100
    }
  ]
}
```
You can then run `stitch <path_to_stitch_config_file>` to run stitching. The results will be output in the `output` folder.

## Stitch Config Generator UI

Additionally, you can go to https://webstitch.app/ to easily generate configuration files for both this program and ImageJ.

## Using Docker Container

build
```
docker build -t stitch-rs .
```

run
```
docker run --rm -v ./workdir:/workdir stitch-rs /workdir/config_file.json
```