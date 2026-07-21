if Sys.iswindows()
	ENV["GTK_USE_PORTAL"] = "0"
	ENV["GIO_USE_VFS"] = "local"
	ENV["GSETTINGS_BACKEND"] = "memory"
end

using Gtk4
using OpenCV
using ImageCore
using TestImages
using TOML

const OPERATIONS = ["Identity", "Gray", "Blur", "Canny", "Threshold", "Rotate", "Resize", "Negative", "Erode", "Dilation"]

mutable struct PipelineNode
	id::Int
	frame::Any
	source_combo::Any
	operation_combo::Any
	param_label::Any
	param_spin::Any
	image_widget::Any
	status_label::Any
	x::Float64
	y::Float64
	output::Any
	preview_file::String
end

mutable struct AppState
	canvas::Any
	nodes::Dict{Int, PipelineNode}
	next_id::Int
	source_image::Any
	app_status::Any
end

function combo_text(combo, default::String)
	try
		text = Gtk4.active_text(combo)
		return text === nothing ? default : String(text)
	catch
		return default
	end
end

function source_options_for_node(state::AppState, node_id::Int)
	options = ["App Source"]
	for id in sort(collect(keys(state.nodes)))
		if id != node_id
			push!(options, "Node $(id)")
		end
	end
	return options
end

function set_app_source!(state::AppState, source_picker, selected::String)
	if selected == "Lighthouse"
		set_gtk_property!(source_picker, :active, 1)
	else
		selected = "Mandrill"
		set_gtk_property!(source_picker, :active, 0)
	end
	state.source_image = load_test_source(selected)
	set_gtk_property!(state.app_status, :label, "Source: $(selected)")
end

function clear_nodes!(state::AppState)
	for node in values(state.nodes)
		if !isempty(node.preview_file)
			try
				rm(node.preview_file)
			catch
			end
		end
		Gtk4.G_.remove(state.canvas, node.frame)
	end
	empty!(state.nodes)
	state.next_id = 1
end

function save_pipeline!(state::AppState, path::String, app_source::String)
	nodes_data = Any[]
	for node_id in sort(collect(keys(state.nodes)))
		node = state.nodes[node_id]
		push!(nodes_data, Dict(
			"id" => node.id,
			"x" => node.x,
			"y" => node.y,
			"source" => combo_text(node.source_combo, "App Source"),
			"operation" => combo_text(node.operation_combo, "Identity"),
			"param" => Int(round(get_gtk_property(node.param_spin, :value, Float64)))
		))
	end

	config = Dict(
		"version" => 1,
		"app_source" => app_source,
		"nodes" => nodes_data
	)

	open(path, "w") do io
		TOML.print(io, config)
	end
end

function load_pipeline!(state::AppState, path::String, source_picker)
	config = TOML.parsefile(path)
	nodes_data = get(config, "nodes", Any[])
	app_source = String(get(config, "app_source", "Mandrill"))

	clear_nodes!(state)
	set_app_source!(state, source_picker, app_source)

	sorted_nodes = sort(collect(nodes_data), by = n -> Int(get(n, "id", 0)))
	for node_cfg in sorted_nodes
		x = Float64(get(node_cfg, "x", 30.0))
		y = Float64(get(node_cfg, "y", 30.0))
		make_node_component!(state, x, y)
	end

	refresh_source_options!(state)
	for node_cfg in sorted_nodes
		node_id = Int(get(node_cfg, "id", 0))
		if !haskey(state.nodes, node_id)
			continue
		end

		node = state.nodes[node_id]
		op = String(get(node_cfg, "operation", "Identity"))
		op_index = findfirst(x -> x == op, OPERATIONS)
		set_gtk_property!(node.operation_combo, :active, (op_index === nothing ? 1 : op_index) - 1)

		param = Float64(get(node_cfg, "param", 9))
		set_gtk_property!(node.param_spin, :value, param)

		source = String(get(node_cfg, "source", "App Source"))
		options = source_options_for_node(state, node.id)
		source_index = findfirst(x -> x == source, options)
		set_gtk_property!(node.source_combo, :active, (source_index === nothing ? 1 : source_index) - 1)

		update_param_ui!(node)
	end

	recompute_all!(state)
end

function load_test_source(name::String)
	image_name = name == "Lighthouse" ? "lighthouse.png" : "mandrill.tiff"
	img = testimage(image_name)
	return collect(rawview(channelview(img)))
end

function ensure_gray(img)
	try
		return OpenCV.cvtColor(img, OpenCV.COLOR_RGB2GRAY)
	catch
		try
			return OpenCV.cvtColor(img, OpenCV.COLOR_BGR2GRAY)
		catch
			return img
		end
	end
end

function morphology_kernel(param_value::Int)
	k = max(1, param_value)
	if iseven(k)
		k += 1
	end
	return ones(UInt8, k, k)
end

function apply_operation(img, op::String, param_value::Int)
	if op == "Identity"
		return img
	elseif op == "Gray"
		return ensure_gray(img)
	elseif op == "Blur"
		k = max(1, param_value)
		if iseven(k)
			k += 1
		end
		return OpenCV.blur(img, OpenCV.Size(Int32(k), Int32(k)))
	elseif op == "Canny"
		gray = ensure_gray(img)
		low = Float64(max(1, param_value))
		high = low * 2.0
		return OpenCV.Canny(gray, low, high)
	elseif op == "Threshold"
		gray = ensure_gray(img)
		level = Float64(clamp(param_value * 4, 0, 255))
		_, dst = OpenCV.threshold(gray, level, 255.0, OpenCV.THRESH_BINARY)
		return dst
	elseif op == "Rotate"
		mode = mod(param_value - 1, 3) + 1
		if mode == 1
			return OpenCV.rotate(img, OpenCV.ROTATE_90_CLOCKWISE)
		elseif mode == 2
			return OpenCV.rotate(img, OpenCV.ROTATE_180)
		else
			return OpenCV.rotate(img, OpenCV.ROTATE_90_COUNTERCLOCKWISE)
		end
	elseif op == "Resize"
		scale = clamp(Float64(param_value) / 10.0, 0.1, 5.0)
		return OpenCV.resize(
			img,
			OpenCV.Size(Int32(0), Int32(0));
			fx = scale,
			fy = scale,
			interpolation = OpenCV.INTER_LINEAR
		)
	elseif op == "Negative"
		return OpenCV.bitwise_not(img)
	elseif op == "Erode"
		kernel = morphology_kernel(param_value)
		return OpenCV.erode(img, kernel)
	elseif op == "Dilation"
		kernel = morphology_kernel(param_value)
		return OpenCV.dilate(img, kernel)
	else
		return img
	end
end

function update_param_ui!(node::PipelineNode)
	op = String(Gtk4.active_text(node.operation_combo))
	has_param = false

	if op == "Blur"
		has_param = true
		set_gtk_property!(node.param_label, :label, "Kernel Size")
		set_gtk_property!(node.param_spin, :value, max(1.0, get_gtk_property(node.param_spin, :value, Float64)))
	elseif op == "Canny"
		has_param = true
		set_gtk_property!(node.param_label, :label, "Canny Low Threshold")
		set_gtk_property!(node.param_spin, :value, max(1.0, get_gtk_property(node.param_spin, :value, Float64)))
	elseif op == "Threshold"
		has_param = true
		set_gtk_property!(node.param_label, :label, "Threshold Level (x4)")
		set_gtk_property!(node.param_spin, :value, max(0.0, get_gtk_property(node.param_spin, :value, Float64)))
	elseif op == "Rotate"
		has_param = true
		set_gtk_property!(node.param_label, :label, "Rotate Mode (1:CW90 2:180 3:CCW90)")
		v = Int(round(get_gtk_property(node.param_spin, :value, Float64)))
		if v < 1 || v > 3
			set_gtk_property!(node.param_spin, :value, 1.0)
		end
	elseif op == "Resize"
		has_param = true
		set_gtk_property!(node.param_label, :label, "Scale (x/10)")
		v = get_gtk_property(node.param_spin, :value, Float64)
		set_gtk_property!(node.param_spin, :value, clamp(v, 1.0, 50.0))
	elseif op == "Erode"
		has_param = true
		set_gtk_property!(node.param_label, :label, "Kernel Size")
		set_gtk_property!(node.param_spin, :value, max(1.0, get_gtk_property(node.param_spin, :value, Float64)))
	elseif op == "Dilation"
		has_param = true
		set_gtk_property!(node.param_label, :label, "Kernel Size")
		set_gtk_property!(node.param_spin, :value, max(1.0, get_gtk_property(node.param_spin, :value, Float64)))
	elseif op == "Gray"
		set_gtk_property!(node.param_label, :label, "No Parameter")
	else
		set_gtk_property!(node.param_label, :label, "No Parameter")
	end

	visible(node.param_label, has_param)
	visible(node.param_spin, has_param)
end

function render_preview!(node::PipelineNode, out_img)
	out_file = tempname() * ".png"
	ok = OpenCV.imwrite(out_file, out_img)
	if !ok
		set_gtk_property!(node.status_label, :label, "Preview write failed")
		return
	end

	set_gtk_property!(node.image_widget, :file, out_file)

	if !isempty(node.preview_file)
		try
			rm(node.preview_file)
		catch
		end
	end
	node.preview_file = out_file
end

function parse_source_node_id(source_text::String)
	if !startswith(source_text, "Node ")
		return nothing
	end
	parts = split(source_text)
	if length(parts) != 2
		return nothing
	end
	try
		return parse(Int, parts[2])
	catch
		return nothing
	end
end

function resolve_node_output!(state::AppState, node_id::Int, visiting::Set{Int}, cache::Dict{Int, Any})
	if haskey(cache, node_id)
		return cache[node_id]
	end
	if node_id in visiting
		error("Cycle detected at Node $(node_id)")
	end
	if !haskey(state.nodes, node_id)
		error("Node $(node_id) does not exist")
	end

	node = state.nodes[node_id]
	push!(visiting, node_id)

	selected_source = String(Gtk4.active_text(node.source_combo))
	source_node_id = parse_source_node_id(selected_source)
	input_image = state.source_image

	if source_node_id !== nothing
		if source_node_id == node.id
			error("Node $(node.id) cannot use itself as input")
		end
		input_image = resolve_node_output!(state, source_node_id, visiting, cache)
	end

	selected_operation = String(Gtk4.active_text(node.operation_combo))
	param_value = Int(round(get_gtk_property(node.param_spin, :value, Float64)))
	output_image = apply_operation(input_image, selected_operation, param_value)

	cache[node_id] = output_image
	delete!(visiting, node_id)
	return output_image
end

function recompute_all!(state::AppState)
	if isempty(state.nodes)
		return
	end

	cache = Dict{Int, Any}()
	visiting = Set{Int}()

	for node_id in sort(collect(keys(state.nodes)))
		node = state.nodes[node_id]
		try
			out_img = resolve_node_output!(state, node_id, visiting, cache)
			node.output = out_img
			render_preview!(node, out_img)
			set_gtk_property!(node.status_label, :label, "Rendered")
		catch err
			set_gtk_property!(node.status_label, :label, "Error: $(err)")
		end
	end
end

function refresh_source_options!(state::AppState)
	all_ids = sort(collect(keys(state.nodes)))

	for node in values(state.nodes)
		existing = try
			String(Gtk4.active_text(node.source_combo))
		catch
			"App Source"
		end

		options = ["App Source"]
		for id in all_ids
			if id != node.id
				push!(options, "Node $(id)")
			end
		end

		Gtk4.G_.remove_all(node.source_combo)
		for opt in options
			push!(node.source_combo, opt)
		end

		selected_index = findfirst(x -> x == existing, options)
		if selected_index === nothing
			selected_index = 1
		end
		set_gtk_property!(node.source_combo, :active, selected_index - 1)
	end
end

function make_node_component!(state::AppState, start_x::Float64, start_y::Float64)
	node_id = state.next_id
	state.next_id += 1

	frame = GtkFrame("Node $(node_id)")
	set_gtk_property!(frame, :width_request, 290)

	content = GtkBox(:v, 6)
	set_gtk_property!(content, :margin_top, 8)
	set_gtk_property!(content, :margin_bottom, 8)
	set_gtk_property!(content, :margin_start, 8)
	set_gtk_property!(content, :margin_end, 8)

	drag_title = GtkLabel("Drag here")
	set_gtk_property!(drag_title, :xalign, 0.0)

	source_label = GtkLabel("Source")
	set_gtk_property!(source_label, :xalign, 0.0)
	source_combo = GtkComboBoxText()
	push!(source_combo, "App Source")
	set_gtk_property!(source_combo, :active, 0)

	op_label = GtkLabel("Operation")
	set_gtk_property!(op_label, :xalign, 0.0)
	operation_combo = GtkComboBoxText()
	for op in OPERATIONS
		push!(operation_combo, op)
	end
	set_gtk_property!(operation_combo, :active, 0)

	param_label = GtkLabel("Param")
	set_gtk_property!(param_label, :xalign, 0.0)
	param_spin = GtkSpinButton(1.0, 51.0, 1.0)
	set_gtk_property!(param_spin, :value, 9.0)

	preview = GtkImage()
	set_gtk_property!(preview, :hexpand, true)
	set_gtk_property!(preview, :vexpand, true)
	preview_frame = GtkFrame("Output")
	set_gtk_property!(preview_frame, :height_request, 170)
	preview_frame[] = preview

	node_status = GtkLabel("Idle")
	set_gtk_property!(node_status, :xalign, 0.0)

	push!(content, drag_title)
	push!(content, source_label)
	push!(content, source_combo)
	push!(content, op_label)
	push!(content, operation_combo)
	push!(content, param_label)
	push!(content, param_spin)
	push!(content, preview_frame)
	push!(content, node_status)
	frame[] = content

	Gtk4.G_.put(state.canvas, frame, start_x, start_y)

	node = PipelineNode(
		node_id,
		frame,
		source_combo,
		operation_combo,
		param_label,
		param_spin,
		preview,
		node_status,
		start_x,
		start_y,
		nothing,
		""
	)
	state.nodes[node_id] = node

	drag = GtkGestureDrag(drag_title)
	drag_base_x = Ref(start_x)
	drag_base_y = Ref(start_y)

	signal_connect(drag, "drag-begin") do _, _, _
		drag_base_x[] = node.x
		drag_base_y[] = node.y
	end

	signal_connect(drag, "drag-update") do _, dx, dy
		node.x = max(0.0, drag_base_x[] + Float64(dx))
		node.y = max(0.0, drag_base_y[] + Float64(dy))
		Gtk4.G_.move(state.canvas, node.frame, node.x, node.y)
	end

	signal_connect(source_combo, "changed") do _
		recompute_all!(state)
	end
	signal_connect(operation_combo, "changed") do _
		update_param_ui!(node)
		recompute_all!(state)
	end
	signal_connect(param_spin, "value-changed") do _
		recompute_all!(state)
	end

	update_param_ui!(node)

	refresh_source_options!(state)
	recompute_all!(state)
end

function build_ui(app)
	win = GtkApplicationWindow(app, "Julia OpenCV Image Processing Pipeline")
	set_gtk_property!(win, :default_width, 1180)
	set_gtk_property!(win, :default_height, 760)

	root = GtkBox(:v, 10)
	set_gtk_property!(root, :margin_top, 12)
	set_gtk_property!(root, :margin_bottom, 12)
	set_gtk_property!(root, :margin_start, 12)
	set_gtk_property!(root, :margin_end, 12)

	toolbar = GtkBox(:h, 8)
	add_node_btn = GtkButton("Add Component")
	render_btn = GtkButton("Recompute Graph")
	save_btn = GtkButton("Save Pipeline")
	load_btn = GtkButton("Load Pipeline")
	config_path_entry = GtkEntry()
	set_gtk_property!(config_path_entry, :text, "pipeline.toml")
	set_gtk_property!(config_path_entry, :hexpand, true)
	source_picker = GtkComboBoxText()
	push!(source_picker, "Mandrill")
	push!(source_picker, "Lighthouse")
	set_gtk_property!(source_picker, :active, 0)

	toolbar_status = GtkLabel("Source: Mandrill")
	set_gtk_property!(toolbar_status, :xalign, 0.0)

	push!(toolbar, add_node_btn)
	push!(toolbar, render_btn)
	push!(toolbar, save_btn)
	push!(toolbar, load_btn)
	push!(toolbar, GtkLabel("Config File"))
	push!(toolbar, config_path_entry)
	push!(toolbar, GtkLabel("App Source"))
	push!(toolbar, source_picker)
	push!(toolbar, toolbar_status)

	canvas = GtkFixed()
	set_gtk_property!(canvas, :width_request, 2800)
	set_gtk_property!(canvas, :height_request, 1600)

	scroller = GtkScrolledWindow()
	set_gtk_property!(scroller, :hexpand, true)
	set_gtk_property!(scroller, :vexpand, true)
	scroller[] = canvas

	state = AppState(canvas, Dict{Int, PipelineNode}(), 1, load_test_source("Mandrill"), toolbar_status)

	signal_connect(add_node_btn, "clicked") do _
		slot = state.next_id - 1
		x = 30.0 + (slot % 4) * 300.0
		y = 30.0 + floor(slot / 4) * 320.0
		make_node_component!(state, x, y)
	end

	signal_connect(render_btn, "clicked") do _
		recompute_all!(state)
		set_gtk_property!(state.app_status, :label, "Graph recomputed")
	end

	signal_connect(save_btn, "clicked") do _
		path = String(get_gtk_property(config_path_entry, :text, String))
		if isempty(strip(path))
			set_gtk_property!(state.app_status, :label, "Save failed: empty config path")
			return
		end

		try
			save_pipeline!(state, path, combo_text(source_picker, "Mandrill"))
			set_gtk_property!(state.app_status, :label, "Saved pipeline: $(path)")
		catch err
			set_gtk_property!(state.app_status, :label, "Save failed: $(err)")
		end
	end

	signal_connect(load_btn, "clicked") do _
		path = String(get_gtk_property(config_path_entry, :text, String))
		if isempty(strip(path))
			set_gtk_property!(state.app_status, :label, "Load failed: empty config path")
			return
		end
		if !isfile(path)
			set_gtk_property!(state.app_status, :label, "Load failed: file not found")
			return
		end

		try
			load_pipeline!(state, path, source_picker)
			set_gtk_property!(state.app_status, :label, "Loaded pipeline: $(path)")
		catch err
			set_gtk_property!(state.app_status, :label, "Load failed: $(err)")
		end
	end

	signal_connect(source_picker, "changed") do picker
		selected = combo_text(picker, "Mandrill")
		state.source_image = load_test_source(selected)
		set_gtk_property!(state.app_status, :label, "Source: $(selected)")
		recompute_all!(state)
	end

	push!(root, toolbar)
	push!(root, scroller)
	win[] = root
	show(win)

	make_node_component!(state, 30.0, 30.0)
end

app = GtkApplication("com.dietrick.julia-image-processing-pipeline")
signal_connect(app, "activate") do a
	build_ui(a)
end
Gtk4.run(app)