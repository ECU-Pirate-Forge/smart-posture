# Julia OpenCV Image Pipeline

`image_processing_pipeline.jl` is a desktop image-processing pipeline editor built with Julia, Gtk4, OpenCV, ImageCore, TestImages, and TOML. It opens a GTK window where you can add draggable processing nodes, chain node outputs together, preview results, and save or load the graph as a TOML pipeline file.

## Prerequisites

- Windows, macOS, or Linux with a desktop session capable of opening GTK4 windows.
- Julia 1.10 or newer. The checked-in manifest was generated with Julia 1.10.0.
- Internet access during first setup so Julia can download packages and binary artifacts.
- A terminal opened at the repository root.



## Setup

Using precompiled OpenCV.jl bindings.

From the repository root:
julia --project=.   # to open current folder as a project using Project.toml
]                   # to open package manager
instantiate         # download and install packages listed in Project.toml
<press the back button> to exit package manager
exit()              # exits the julia prompt

julia --project=. image_processing_pipeline.jl # runs the julia app





From the repository root:

```bash
cd OpenCvDemo
julia --project=. -e "using Pkg; Pkg.instantiate(); Pkg.precompile()"
```

The project currently pins `OpenCV_jll` to `4.10.0+0`, which resolves `Qt6Base_jll` to a version that loads correctly on this Windows setup. Keep that pin unless a newer OpenCV/Qt artifact has been verified locally.

To check that the application imports are ready:

```powershell
julia --project=. -e "using Gtk4; using OpenCV; using ImageCore; using TestImages; using TOML; println(\"imports ok\")"
```

## Run

Start the app from the repository root:

```powershell
julia --project=. image_processing_pipeline.jl
```

The application opens a window titled `Julia OpenCV Image Processing Pipeline`.

## What You Can Do

- Choose a built-in source image: `Mandrill` or `Lighthouse`.
- Add draggable processing nodes to a scrollable canvas.
- Select each node input from `App Source` or another node output.
- Apply OpenCV operations and preview each node output.
- Recompute the graph after changes.
- Save and load pipeline layouts as TOML files. The default path is `pipeline.toml`.

## Operations

| Operation | Parameter |
| --- | --- |
| `Identity` | No parameter |
| `Gray` | No parameter |
| `Blur` | Kernel size |
| `Canny` | Low threshold; high threshold is `low * 2` |
| `Threshold` | Threshold level, scaled as `parameter * 4` |
| `Rotate` | `1` = 90 degrees clockwise, `2` = 180 degrees, `3` = 90 degrees counter-clockwise |
| `Resize` | Scale factor as `parameter / 10` |
| `Negative` | No parameter |
| `Erode` | Kernel size |
| `Dilation` | Kernel size |

## Basic Workflow

1. Choose the `App Source` image.
2. Click `Add Component` to add a node.
3. Drag nodes by the `Drag here` label.
4. In each node, choose a `Source` and an `Operation`.
5. Adjust the parameter control when the selected operation exposes one.
6. Click `Recompute Graph` to force a full graph refresh.
7. Enter a config file path, such as `pipeline.toml`.
8. Click `Save Pipeline` to write the current graph.
9. Click `Load Pipeline` to restore a saved graph.

## Pipeline File Format

Pipeline files are TOML documents with an app source and a list of nodes. Each node stores its ID, canvas position, selected source, selected operation, and parameter value.

Example:

```toml
app_source = "Mandrill"
version = 1

[[nodes]]
source = "App Source"
param = 9
id = 1
x = 30.0
operation = "Identity"
y = 30.0

[[nodes]]
source = "Node 1"
param = 3
id = 2
x = 330.0
operation = "Blur"
y = 30.0
```

Node sources can reference `App Source` or another node by name, such as `Node 1`. Cycles are detected during graph recomputation and reported in the affected node status label.

## Troubleshooting

### `using OpenCV` or precompilation fails

Run the import check:

```powershell
julia --project=. -e "using OpenCV; println(\"OpenCV ok\")"
```

If it fails with a Qt DLL error, confirm that `OpenCV_jll` is still pinned in the project:

```powershell
julia --project=. -e "using Pkg; Pkg.status([\"OpenCV\", \"OpenCV_jll\", \"Qt6Base_jll\"]; mode=Pkg.PKGMODE_MANIFEST)"
```

The known-working stack for this project is `OpenCV 4.6.1`, pinned `OpenCV_jll 4.10.0+0`, and `Qt6Base_jll 6.7.1+1`.

### The GTK window does not open

- Confirm you are running from a desktop session with display access.
- Re-run setup with `julia --project=. -e "using Pkg; Pkg.instantiate(); Pkg.precompile()"`.
- On Windows, start from a regular PowerShell or Command Prompt instead of a Conda-activated shell.

### A saved pipeline does not load as expected

- Confirm the config path points to an existing TOML file.
- Confirm node sources reference either `App Source` or existing nodes such as `Node 1`.
- If the graph contains a cycle, the app reports it in the affected node status label.
