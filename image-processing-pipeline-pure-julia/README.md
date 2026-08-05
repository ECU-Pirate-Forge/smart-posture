# Pure Julia Image Pipeline

This is a Julia-native version of the GTK image-processing pipeline editor in
`../image-processing-pipeline-julia`. It preserves the draggable node graph,
built-in images, previews, and TOML save/load format without importing OpenCV
or linking to OpenCV binaries.

The operations are implemented in `PureImageOperations.jl` with Julia arrays:

- grayscale conversion and thresholding
- summed-area box blur
- Canny edge detection with Sobel gradients, non-maximum suppression, and hysteresis
- bilinear resize
- rotation and negative
- separable erosion and dilation

## Setup

From this directory:

```powershell
julia --project=. -e "using Pkg; Pkg.instantiate(); Pkg.precompile()"
```

## Run

```powershell
julia --project=. image_processing_pipeline.jl
```

## Test

```powershell
julia --project=. test\runtests.jl
```

The pipeline file format and operation parameters are compatible with the
reference implementation. PNG previews are written through ImageIO rather than
OpenCV.
