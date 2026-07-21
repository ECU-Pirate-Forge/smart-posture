if Sys.iswindows()
	ENV["GTK_USE_PORTAL"] = "0"
	ENV["GIO_USE_VFS"] = "local"
	ENV["GSETTINGS_BACKEND"] = "memory"
end

using Gtk4
using OpenCV

const PYTHONCALL_REF = Ref{Any}(nothing)
const PYTHONCALL_AVAILABLE = let
	ok = true
	try
		@eval using PythonCall
		PYTHONCALL_REF[] = PythonCall
	catch
		ok = false
	end
	ok
end

const MEDIAPIPE_LANDMARKS = [
	("Nose", 0),
	("Left Eye Inner", 1),
	("Left Eye", 2),
	("Left Eye Outer", 3),
	("Right Eye Inner", 4),
	("Right Eye", 5),
	("Right Eye Outer", 6),
	("Left Ear", 7),
	("Right Ear", 8),
	("Mouth Left", 9),
	("Mouth Right", 10),
	("Left Shoulder", 11),
	("Right Shoulder", 12),
	("Left Elbow", 13),
	("Right Elbow", 14),
	("Left Wrist", 15),
	("Right Wrist", 16),
	("Left Pinky", 17),
	("Right Pinky", 18),
	("Left Index", 19),
	("Right Index", 20),
	("Left Thumb", 21),
	("Right Thumb", 22),
	("Left Hip", 23),
	("Right Hip", 24),
	("Left Knee", 25),
	("Right Knee", 26),
	("Left Ankle", 27),
	("Right Ankle", 28),
	("Left Heel", 29),
	("Right Heel", 30),
	("Left Foot Index", 31),
	("Right Foot Index", 32),
]

const JOINTS = vcat(["Head", "Neck", "Upper Back", "Mid Back", "Lower Back"], [name for (name, _) in MEDIAPIPE_LANDMARKS])
const SEGMENTS = [(1, 2), (2, 3), (3, 4), (4, 5)]

mutable struct PostureState
	image_path::String
	original_bgr::Any
	display_bgr::Any
	display_scale::Float64
	points::Dict{String, Tuple{Float64, Float64}}
	preview_file::String
	image_widget::Any
	image_host_widget::Any
	click_handler::Any
	status_label::Any
	angle_label::Any
	joint_combo::Any
end

function require_pythoncall()
	if PYTHONCALL_AVAILABLE && PYTHONCALL_REF[] !== nothing
		return PYTHONCALL_REF[]
	else
		error("PythonCall.jl is required for auto-detect. Install with: julia -e 'using Pkg; Pkg.add(\"PythonCall\")'")
	end
end

function py_is_none(PC, obj)
	if hasproperty(PC, :pyisnone)
		return PC.pyisnone(obj)
	elseif hasproperty(PC, :pyisnull)
		return PC.pyisnull(obj)
	else
		try
			return Bool(PC.pyconvert(Bool, PC.pyeq(obj, PC.pybuiltins.None)))
		catch
			return false
		end
	end
end

function py_import_safe(PC, module_name::AbstractString)
	name = strip(String(module_name))
	if isempty(name)
		error("Python module name is empty")
	end
	try
		return PC.pyimport(name)
	catch err
		pyexe = try
			String(PC.python_executable_path())
		catch
			"<unknown>"
		end
		error("Failed to import Python module '$(name)' from $(pyexe). Error: $(err)")
	end
end

function py_has_attr(PC, obj, name::AbstractString)
	try
		return Bool(PC.pyconvert(Bool, PC.pyhasattr(obj, String(name))))
	catch
		try
			PC.pygetattr(obj, String(name))
			return true
		catch
			return false
		end
	end
end

function py_to_float64(PC, value; default = nothing)
	try
		return Float64(PC.pyconvert(Float64, value))
	catch
		try
			return Float64(value)
		catch
			if default !== nothing
				return default
			end
			error("Failed to convert Python numeric value to Float64")
		end
	end
end

function print_python_runtime_banner()
	if PYTHONCALL_AVAILABLE && PYTHONCALL_REF[] !== nothing
		PC = PYTHONCALL_REF[]
		pyexe = "<unknown>"
		try
			raw = PC.python_executable_path()
			if raw !== missing
				pyexe = String(raw)
			end
		catch
		end
		pyver = try
			sys = py_import_safe(PC, "sys")
			if pyexe == "<unknown>"
				pyexe = String(PC.pyconvert(String, PC.pygetattr(sys, "executable")))
			end
			String(PC.pyconvert(String, PC.pygetattr(sys, "version")))
		catch err
			"<unknown> (" * sprint(showerror, err) * ")"
		end

		println("[find_posture] Python executable: $(pyexe)")
		println("[find_posture] Python version: $(pyver)")
	else
		println("[find_posture] PythonCall unavailable; auto-detect is disabled.")
	end
end

function get_mediapipe_pose_landmarker(PC)
	model_path = joinpath(@__DIR__, "mediapipe-models", "pose_landmarker_heavy.task")
	if !isfile(model_path)
		error("MediaPipe pose landmarker model not found: $(model_path)")
	end

	mp = py_import_safe(PC, "mediapipe")
	tasks_python = py_import_safe(PC, "mediapipe.tasks.python")
	vision = py_import_safe(PC, "mediapipe.tasks.python.vision")

	base_options_type = PC.pygetattr(tasks_python, "BaseOptions")
	options_type = PC.pygetattr(vision, "PoseLandmarkerOptions")
	landmarker_type = PC.pygetattr(vision, "PoseLandmarker")

	base_options = base_options_type(model_asset_path = model_path)
	options = options_type(base_options = base_options, output_segmentation_masks = false)
	detector = landmarker_type.create_from_options(options)

	return mp, detector
end

function safe_combo_text(combo, default::String)
	try
		txt = Gtk4.active_text(combo)
		return txt === nothing ? default : String(txt)
	catch
		return default
	end
end

function mat_dims(img)
	d = size(img)
	if length(d) == 3 && d[1] in (1, 3, 4)
		return Int(d[3]), Int(d[2])
	elseif length(d) >= 2
		return Int(d[1]), Int(d[2])
	end
	return 0, 0
end

function render_canvas!(state::PostureState)
	if state.display_bgr === nothing
		return
	end

	canvas = copy(state.display_bgr)

	for joint in JOINTS
		if haskey(state.points, joint)
			ox, oy = state.points[joint]
			dx = Int(round(ox * state.display_scale))
			dy = Int(round(oy * state.display_scale))
			p = OpenCV.Point(Int32(dx), Int32(dy))
			OpenCV.circle(canvas, p, 6, (0, 255, 255); thickness = 2)
			OpenCV.putText(
				canvas,
				joint,
				OpenCV.Point(Int32(dx + 8), Int32(max(14, dy - 8))),
				OpenCV.FONT_HERSHEY_SIMPLEX,
				0.5,
				(255, 255, 0);
				thickness = 1
			)
		end
	end

	for (a, b) in SEGMENTS
		ja = JOINTS[a]
		jb = JOINTS[b]
		if haskey(state.points, ja) && haskey(state.points, jb)
			ax, ay = state.points[ja]
			bx, by = state.points[jb]
			p1 = OpenCV.Point(Int32(round(ax * state.display_scale)), Int32(round(ay * state.display_scale)))
			p2 = OpenCV.Point(Int32(round(bx * state.display_scale)), Int32(round(by * state.display_scale)))
			OpenCV.line(canvas, p1, p2, (0, 255, 0); thickness = 2)
		end
	end

	out_file = tempname() * ".png"
	ok = OpenCV.imwrite(out_file, canvas)
	if ok
		dh, dw = mat_dims(canvas)
		image = GtkPicture(out_file)
		set_gtk_property!(image, :halign, Gtk4.Align_START)
		set_gtk_property!(image, :valign, Gtk4.Align_START)
		set_gtk_property!(image, :can_shrink, false)
		set_gtk_property!(image, :width_request, dw)
		set_gtk_property!(image, :height_request, dh)
		state.image_widget = image
		state.image_host_widget[] = image
		if state.click_handler !== nothing
			state.click_handler(image)
		end
		if !isempty(state.preview_file)
			try
				rm(state.preview_file)
			catch
			end
		end
		state.preview_file = out_file
	end
end

function update_angle_text!(state::PostureState)
	lines = ["Segment Angles (degrees, deviation from vertical)"]

	for (a, b) in SEGMENTS
		ja = JOINTS[a]
		jb = JOINTS[b]
		if haskey(state.points, ja) && haskey(state.points, jb)
			x1, y1 = state.points[ja]
			x2, y2 = state.points[jb]
			dx = x2 - x1
			dy = y2 - y1
			ang = atan(abs(dx), abs(dy)) * 180.0 / pi
			push!(lines, "$(ja) -> $(jb): $(round(ang, digits = 1))")
		else
			push!(lines, "$(ja) -> $(jb): n/a")
		end
	end

	set_gtk_property!(state.angle_label, :label, join(lines, "\n"))
end

function set_status!(state::PostureState, text::String)
	set_gtk_property!(state.status_label, :label, text)
end

function default_image_folder()
	data_dir = joinpath(@__DIR__, "data")
	return isdir(data_dir) ? data_dir : @__DIR__
end

function load_image!(state::PostureState, path::String)
	img = OpenCV.imread(path)
	h, w = mat_dims(img)
	if h == 0 || w == 0
		error("Could not read image: $(path)")
	end

	max_w = 1200
	max_h = 1050
	scale = min(max_w / w, max_h / h)
	scale = min(scale, 1.0)

	display = if scale < 0.999
		nw = max(1, Int(round(w * scale)))
		nh = max(1, Int(round(h * scale)))
		OpenCV.resize(img, OpenCV.Size(Int32(nw), Int32(nh)); interpolation = OpenCV.INTER_AREA)
	else
		copy(img)
	end

	state.image_path = path
	state.original_bgr = img
	state.display_bgr = display
	state.display_scale = scale
	empty!(state.points)

	render_canvas!(state)
	update_angle_text!(state)
	set_status!(state, "Loaded image: $(path)")
end

function next_joint!(state::PostureState)
	idx = Int(get_gtk_property(state.joint_combo, :active, Int32)) + 1
	idx = min(length(JOINTS), idx + 1)
	set_gtk_property!(state.joint_combo, :active, idx - 1)
end

function click_to_original_coords(state::PostureState, x::Float64, y::Float64)
	dh, dw = mat_dims(state.display_bgr)
	if dh == 0 || dw == 0
		return nothing
	end

	widget_w = max(1.0, Float64(width(state.image_widget)))
	widget_h = max(1.0, Float64(height(state.image_widget)))

	if x < 0 || y < 0 || x > widget_w || y > widget_h
		return nothing
	end

	render_scale = min(widget_w / dw, widget_h / dh)
	rendered_w = dw * render_scale
	rendered_h = dh * render_scale
	pad_x = max(0.0, (widget_w - rendered_w) / 2)
	pad_y = max(0.0, (widget_h - rendered_h) / 2)

	if x < pad_x || y < pad_y || x > pad_x + rendered_w || y > pad_y + rendered_h
		return nothing
	end

	display_x = (x - pad_x) / render_scale
	display_y = (y - pad_y) / render_scale

	ox = display_x / state.display_scale
	oy = display_y / state.display_scale

	h, w = mat_dims(state.original_bgr)
	ox = clamp(ox, 0.0, max(0.0, w - 1.0))
	oy = clamp(oy, 0.0, max(0.0, h - 1.0))

	return ox, oy
end

function auto_detect_joints!(state::PostureState)
	if state.original_bgr === nothing || isempty(state.image_path)
		error("Load an image first")
	end

	PC = require_pythoncall()
	mp, detector = get_mediapipe_pose_landmarker(PC)
	image_type = PC.pygetattr(mp, "Image")
	image = image_type.create_from_file(state.image_path)
	result = detector.detect(image)

	pose_landmarks_list = PC.pyconvert(Vector{Any}, result.pose_landmarks)
	if isempty(pose_landmarks_list)
		error("No human pose detected")
	end

	landmarks = PC.pyconvert(Vector{Any}, pose_landmarks_list[1])

	function lm_xy(idx::Int; min_vis::Float64 = 0.5)
		if idx + 1 > length(landmarks)
			return nothing
		end
		lm = landmarks[idx + 1]
		vis = py_to_float64(PC, lm.visibility; default = 1.0)
		if vis < min_vis
			return nothing
		end
		x = py_to_float64(PC, lm.x)
		y = py_to_float64(PC, lm.y)
		return x, y
	end

	h, w = mat_dims(state.original_bgr)
	for joint in JOINTS
		pop!(state.points, joint, nothing)
	end

	function to_px(nx::Float64, ny::Float64)
		x = clamp(nx * w, 0.0, max(0.0, w - 1.0))
		y = clamp(ny * h, 0.0, max(0.0, h - 1.0))
		return x, y
	end

	normalized_points = Dict{String, Tuple{Float64, Float64}}()
	detected_count = 0
	for (joint_name, idx) in MEDIAPIPE_LANDMARKS
		p = lm_xy(idx; min_vis = 0.0)
		if p === nothing
			continue
		end
		normalized_points[joint_name] = p
		state.points[joint_name] = to_px(p[1], p[2])
		detected_count += 1
	end

	nose = get(normalized_points, "Nose", nothing)
	ls = get(normalized_points, "Left Shoulder", nothing)
	rs = get(normalized_points, "Right Shoulder", nothing)
	lh = get(normalized_points, "Left Hip", nothing)
	rh = get(normalized_points, "Right Hip", nothing)

	custom_added = false
	if nose !== nothing && ls !== nothing && rs !== nothing && lh !== nothing && rh !== nothing
		neck_x = (ls[1] + rs[1]) / 2
		neck_y = (ls[2] + rs[2]) / 2
		hip_x = (lh[1] + rh[1]) / 2
		hip_y = (lh[2] + rh[2]) / 2

		head = to_px(nose[1], nose[2])
		neck = to_px(neck_x, neck_y)
		upper_back = to_px(neck_x * 0.65 + hip_x * 0.35, neck_y * 0.65 + hip_y * 0.35)
		mid_back = to_px(neck_x * 0.35 + hip_x * 0.65, neck_y * 0.35 + hip_y * 0.65)
		lower_back = to_px(hip_x, hip_y)

		state.points["Head"] = head
		state.points["Neck"] = neck
		state.points["Upper Back"] = upper_back
		state.points["Mid Back"] = mid_back
		state.points["Lower Back"] = lower_back
		custom_added = true
	end

	render_canvas!(state)
	update_angle_text!(state)
	status = "Auto-detected $(detected_count)/$(length(MEDIAPIPE_LANDMARKS)) MediaPipe landmarks"
	if custom_added
		status *= " + custom joints"
	else
		status *= "; custom joints skipped (need nose, shoulders, hips)"
	end
	set_status!(state, status)
end

function build_ui(app)
	win = GtkApplicationWindow(app, "Posture Mapper")
	set_gtk_property!(win, :default_width, 1600)
	set_gtk_property!(win, :default_height, 1200)

	root = GtkBox(:v, 8)
	set_gtk_property!(root, :margin_top, 10)
	set_gtk_property!(root, :margin_bottom, 10)
	set_gtk_property!(root, :margin_start, 10)
	set_gtk_property!(root, :margin_end, 10)

	toolbar = GtkBox(:h, 8)
	path_entry = GtkEntry()
	set_gtk_property!(path_entry, :hexpand, true)
	set_gtk_property!(path_entry, :text, "")
	browse_btn = GtkButton("Browse...")
	load_btn = GtkButton("Load Image")
	auto_btn = GtkButton("Auto Detect")
	clear_btn = GtkButton("Clear Points")

	joint_combo = GtkComboBoxText()
	for j in JOINTS
		push!(joint_combo, j)
	end
	set_gtk_property!(joint_combo, :active, 0)

	auto_advance = GtkCheckButton("Auto Advance")
	set_gtk_property!(auto_advance, :active, true)

	push!(toolbar, GtkLabel("Image Path"))
	push!(toolbar, path_entry)
	push!(toolbar, browse_btn)
	push!(toolbar, load_btn)
	push!(toolbar, auto_btn)
	push!(toolbar, GtkLabel("Joint"))
	push!(toolbar, joint_combo)
	push!(toolbar, auto_advance)
	push!(toolbar, clear_btn)

	body = GtkBox(:h, 10)

	image = GtkBox(:v)
	scroller = GtkScrolledWindow()
	set_gtk_property!(scroller, :hexpand, true)
	set_gtk_property!(scroller, :vexpand, true)
	scroller[] = image

	side = GtkBox(:v, 8)
	set_gtk_property!(side, :width_request, 300)

	help_text = GtkLabel("Click image to place selected joint manually.\nOr use Auto Detect to estimate joints from pose landmarks.")
	set_gtk_property!(help_text, :xalign, 0.0)

	angle_label = GtkLabel("Segment Angles (degrees, from +x axis)")
	set_gtk_property!(angle_label, :xalign, 0.0)
	set_gtk_property!(angle_label, :yalign, 0.0)

	status_label = GtkLabel("Load an image to start")
	set_gtk_property!(status_label, :xalign, 0.0)

	push!(side, help_text)
	push!(side, angle_label)
	push!(side, status_label)

	push!(body, scroller)
	push!(body, side)

	push!(root, toolbar)
	push!(root, body)
	win[] = root

	state = PostureState(
		"",
		nothing,
		nothing,
		1.0,
		Dict{String, Tuple{Float64, Float64}}(),
		"",
		image,
		scroller,
		nothing,
		status_label,
		angle_label,
		joint_combo
	)

	function install_click_handler!(target)
		click = GtkGestureClick(target)
		signal_connect(click, "pressed") do _, _, x, y
			if state.original_bgr === nothing
				set_status!(state, "Load an image first")
				return
			end

			joint = safe_combo_text(state.joint_combo, JOINTS[1])
			mapped = click_to_original_coords(state, Float64(x), Float64(y))
			if mapped === nothing
				set_status!(state, "Click inside the displayed image area")
				return
			end
			ox, oy = mapped

			state.points[joint] = (ox, oy)
			render_canvas!(state)
			update_angle_text!(state)
			set_status!(state, "Mapped $(joint) at ($(Int(round(ox))), $(Int(round(oy))))")

			if get_gtk_property(auto_advance, :active, Bool)
				next_joint!(state)
			end
		end
	end
	state.click_handler = install_click_handler!

	signal_connect(browse_btn, "clicked") do _
		open_dialog(
			"Select image",
			win,
			["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"];
			start_folder = default_image_folder()
		) do filename
			if filename isa AbstractString && !isempty(filename)
				set_gtk_property!(path_entry, :text, String(filename))
				set_status!(state, "Selected image: $(filename)")
			end
		end
	end

	signal_connect(load_btn, "clicked") do _
		path = String(get_gtk_property(path_entry, :text, String))
		if isempty(strip(path))
			set_status!(state, "Image path is empty")
			return
		end
		if !isfile(path)
			set_status!(state, "File not found: $(path)")
			return
		end

		try
			load_image!(state, path)
		catch err
			set_status!(state, "Load failed: $(err)")
		end
	end

	signal_connect(clear_btn, "clicked") do _
		empty!(state.points)
		render_canvas!(state)
		update_angle_text!(state)
		set_status!(state, "Cleared all mapped joints")
	end

	signal_connect(auto_btn, "clicked") do _
		try
			auto_detect_joints!(state)
		catch err
			set_status!(state, "Auto detect failed: $(err)")
		end
	end

	show(win)
end

print_python_runtime_banner()

app = GtkApplication("com.example.posturemapper")
signal_connect(app, "activate") do a
	build_ui(a)
end
Gtk4.run(app)
