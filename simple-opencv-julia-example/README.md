# SimpleOpenCV

`SimpleOpenCV.jl` is a simple OpenCV script written in Julia that uses OpenCV, ImageCore, TestImages, and TOML packges. When ran it simply converts a colored mandril image to a grayscale image.

## Setup

From the project root:

```bash
cd smart-posture/simple-opencv-julia-example/
julia --project=. -e "using Pkg; Pkg.instantiate(); Pkg.precompile()"
```

The project currently pins `OpenCV_jll` to `4.10.0+0`, which resolves `Qt6Base_jll` to a version that loads correctly on this Windows setup. Keep that pin unless a newer OpenCV/Qt artifact has been verified locally.

To check that the application imports are ready:

```bash
julia --project=. -e "using OpenCV; using ImageCore; using TestImages; using TOML; println(\"imports ok\")"
```

## Run

Start the app from the repository root:

```bash
julia --project=. SimpleOpenCV.jl
```

Two windows should open.  One with the original color image of mandril and the modified grayscale image.
