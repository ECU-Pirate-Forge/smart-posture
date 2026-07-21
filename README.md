# smart-posture

A workspace of posture and computer-vision experiments using Julia, Python, OpenCV, and MediaPipe.

## Repository Layout

- `image-processing-pipeline-julia/`
	- GTK desktop app in Julia for building an image-processing graph from draggable nodes.
	- Uses OpenCV operations such as blur, threshold, canny, rotate, resize, and morphology.
- `simple-opencv-julia-example/`
	- Minimal Julia + OpenCV script that shows original and grayscale images.
- `mediapipe_julia_landmarks_using_python_packages/`
	- Julia app that can annotate posture points and integrates with MediaPipe via Python packages.
	- Includes Python MediaPipe scripts and model files for landmark/object demos.
- `mediapipe_python_examples/`
	- Pure Python MediaPipe landmark demo scripts.
- `posture_analysis_python_example/`
	- Python posture analysis utilities/classes for shoulder/hip/spine metrics.

## Prerequisites

- Windows, macOS, or Linux
- Julia 1.10+
- Python 3.10+ (recommended)
- Internet access for first-time package install

## Quick Start (Windows PowerShell)

From the repository root:

```powershell
Set-Location c:\code\pirate-forge\smart-posture
```

### 1) Julia Image Pipeline App

```powershell
Set-Location .\image-processing-pipeline-julia
julia --project=. -e "using Pkg; Pkg.instantiate(); Pkg.precompile()"
julia --project=. .\image_processing_pipeline.jl
```

### 2) Julia OpenCV Minimal Example

```powershell
Set-Location ..\simple-opencv-julia-example
julia --project=. -e "using Pkg; Pkg.instantiate(); Pkg.precompile()"
julia --project=. .\SimpleOpenCV.jl
```

### 3) MediaPipe from Python (inside Julia MediaPipe folder)

```powershell
Set-Location ..\mediapipe_julia_landmarks_using_python_packages
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python .\mediapipe-landmark-example.py
```

### 4) Posture Analysis Python Example

```powershell
Set-Location ..\posture_analysis_python_example
python posture_analysis.py
```

## Notes

- The Julia OpenCV projects currently pin `OpenCV_jll` to `4.10.0+0` in their local project environments for a known working Windows setup.
- In `mediapipe_julia_landmarks_using_python_packages/`, model assets are expected under `mediapipe-models/`.
- If `image.jpg` is missing for MediaPipe demos, download one using the URLs in `sample-images.md`.

## Helpful Commands

Verify Julia imports in a Julia project folder:

```powershell
julia --project=. -e "using OpenCV; using ImageCore; using TestImages; using TOML; println(\"imports ok\")"
```

Verify Python dependencies:

```powershell
python -c "import mediapipe, cv2; print('python deps ok')"
```

## Troubleshooting

- Julia OpenCV import fails on Windows:
	- Use a regular PowerShell/Command Prompt instead of a Conda-activated shell.
	- Re-run package setup with `Pkg.instantiate()` and `Pkg.precompile()`.
- MediaPipe Python import fails:
	- Confirm your virtual environment is activated.
	- Reinstall with `python -m pip install -r requirements.txt`.
- GUI windows do not appear:
	- Run from a desktop session with display access.