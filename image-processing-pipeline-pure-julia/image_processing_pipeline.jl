if Sys.iswindows()
    ENV["GTK_USE_PORTAL"] = "0"
    ENV["GIO_USE_VFS"] = "local"
    ENV["GSETTINGS_BACKEND"] = "memory"
end

using FileIO
using Gtk4
using TestImages
using TOML

include("PureImageOperations.jl")
using .PureImageOperations

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
    text = Gtk4.active_text(combo)
    return text === nothing ? default : String(text)
end

function source_options_for_node(state::AppState, node_id::Int)
    options = ["App Source"]
    append!(options, ["Node $id" for id in sort!(collect(keys(state.nodes))) if id != node_id])
    return options
end

load_test_source(name::String) =
    testimage(name == "Lighthouse" ? "lighthouse.png" : "mandrill.tiff")

function set_app_source!(state::AppState, source_picker, selected::String)
    source_name = selected == "Lighthouse" ? "Lighthouse" : "Mandrill"
    set_gtk_property!(source_picker, :active, source_name == "Lighthouse" ? 1 : 0)
    state.source_image = load_test_source(source_name)
    set_gtk_property!(state.app_status, :label, "Source: $source_name")
end

function remove_preview!(node::PipelineNode)
    if !isempty(node.preview_file) && isfile(node.preview_file)
        rm(node.preview_file)
    end
    node.preview_file = ""
end

function clear_nodes!(state::AppState)
    for node in values(state.nodes)
        remove_preview!(node)
        Gtk4.G_.remove(state.canvas, node.frame)
    end
    empty!(state.nodes)
    state.next_id = 1
end

function save_pipeline!(state::AppState, path::String, app_source::String)
    nodes = [
        Dict(
            "id" => node.id,
            "x" => node.x,
            "y" => node.y,
            "source" => combo_text(node.source_combo, "App Source"),
            "operation" => combo_text(node.operation_combo, "Identity"),
            "param" => Int(round(get_gtk_property(node.param_spin, :value, Float64))),
        )
        for node in (state.nodes[id] for id in sort!(collect(keys(state.nodes))))
    ]
    open(path, "w") do io
        TOML.print(io, Dict("version" => 1, "app_source" => app_source, "nodes" => nodes))
    end
end

function load_pipeline!(state::AppState, path::String, source_picker)
    config = TOML.parsefile(path)
    node_configs = sort(collect(get(config, "nodes", Any[])); by=n -> Int(get(n, "id", 0)))
    clear_nodes!(state)
    set_app_source!(state, source_picker, String(get(config, "app_source", "Mandrill")))

    for config in node_configs
        make_node_component!(
            state,
            Float64(get(config, "x", 30.0)),
            Float64(get(config, "y", 30.0)),
        )
    end

    refresh_source_options!(state)
    for config in node_configs
        node_id = Int(get(config, "id", 0))
        haskey(state.nodes, node_id) || continue
        node = state.nodes[node_id]

        operation = String(get(config, "operation", "Identity"))
        operation_index = something(findfirst(==(operation), OPERATIONS), 1)
        set_gtk_property!(node.operation_combo, :active, operation_index - 1)
        set_gtk_property!(node.param_spin, :value, Float64(get(config, "param", 9)))

        source = String(get(config, "source", "App Source"))
        options = source_options_for_node(state, node.id)
        source_index = something(findfirst(==(source), options), 1)
        set_gtk_property!(node.source_combo, :active, source_index - 1)
        update_param_ui!(node)
    end
    recompute_all!(state)
end

function update_param_ui!(node::PipelineNode)
    operation = combo_text(node.operation_combo, "Identity")
    labels = Dict(
        "Blur" => "Kernel Size",
        "Canny" => "Canny Low Threshold",
        "Threshold" => "Threshold Level (x4)",
        "Rotate" => "Rotate Mode (1:CW90 2:180 3:CCW90)",
        "Resize" => "Scale (x/10)",
        "Erode" => "Kernel Size",
        "Dilation" => "Kernel Size",
    )
    has_parameter = haskey(labels, operation)
    set_gtk_property!(node.param_label, :label, get(labels, operation, "No Parameter"))

    value = get_gtk_property(node.param_spin, :value, Float64)
    if operation == "Rotate"
        set_gtk_property!(node.param_spin, :value, 1 <= value <= 3 ? value : 1.0)
    elseif operation == "Resize"
        set_gtk_property!(node.param_spin, :value, clamp(value, 1.0, 50.0))
    elseif operation != "Threshold"
        set_gtk_property!(node.param_spin, :value, max(1.0, value))
    end
    visible(node.param_label, has_parameter)
    visible(node.param_spin, has_parameter)
end

function render_preview!(node::PipelineNode, image)
    output_path = tempname() * ".png"
    save(output_path, image)
    set_gtk_property!(node.image_widget, :file, output_path)
    remove_preview!(node)
    node.preview_file = output_path
end

function parse_source_node_id(source::String)
    match_result = match(r"^Node (\d+)$", source)
    return match_result === nothing ? nothing : parse(Int, match_result.captures[1])
end

function resolve_node_output!(
    state::AppState,
    node_id::Int,
    visiting::Set{Int},
    cache::Dict{Int, Any},
)
    haskey(cache, node_id) && return cache[node_id]
    node_id in visiting && error("Cycle detected at Node $node_id")
    haskey(state.nodes, node_id) || error("Node $node_id does not exist")

    node = state.nodes[node_id]
    push!(visiting, node_id)
    try
        source_id = parse_source_node_id(combo_text(node.source_combo, "App Source"))
        input = source_id === nothing ?
            state.source_image :
            resolve_node_output!(state, source_id, visiting, cache)
        output = apply_operation(
            input,
            combo_text(node.operation_combo, "Identity"),
            Int(round(get_gtk_property(node.param_spin, :value, Float64))),
        )
        cache[node_id] = output
        return output
    finally
        delete!(visiting, node_id)
    end
end

function recompute_all!(state::AppState)
    cache = Dict{Int, Any}()
    for node_id in sort!(collect(keys(state.nodes)))
        node = state.nodes[node_id]
        try
            output = resolve_node_output!(state, node_id, Set{Int}(), cache)
            node.output = output
            render_preview!(node, output)
            set_gtk_property!(node.status_label, :label, "Rendered")
        catch error
            set_gtk_property!(node.status_label, :label, "Error: $(sprint(showerror, error))")
        end
    end
end

function refresh_source_options!(state::AppState)
    for node in values(state.nodes)
        existing = combo_text(node.source_combo, "App Source")
        options = source_options_for_node(state, node.id)
        Gtk4.G_.remove_all(node.source_combo)
        foreach(option -> push!(node.source_combo, option), options)
        set_gtk_property!(
            node.source_combo,
            :active,
            something(findfirst(==(existing), options), 1) - 1,
        )
    end
end

function make_node_component!(state::AppState, start_x::Float64, start_y::Float64)
    node_id = state.next_id
    state.next_id += 1

    frame = GtkFrame("Node $node_id")
    set_gtk_property!(frame, :width_request, 290)
    content = GtkBox(:v, 6)
    for margin in (:margin_top, :margin_bottom, :margin_start, :margin_end)
        set_gtk_property!(content, margin, 8)
    end

    drag_title = GtkLabel("Drag here")
    source_label = GtkLabel("Source")
    operation_label = GtkLabel("Operation")
    foreach(label -> set_gtk_property!(label, :xalign, 0.0), (drag_title, source_label, operation_label))

    source_combo = GtkComboBoxText()
    push!(source_combo, "App Source")
    set_gtk_property!(source_combo, :active, 0)

    operation_combo = GtkComboBoxText()
    foreach(operation -> push!(operation_combo, operation), OPERATIONS)
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

    status = GtkLabel("Idle")
    set_gtk_property!(status, :xalign, 0.0)
    foreach(
        widget -> push!(content, widget),
        (
            drag_title,
            source_label,
            source_combo,
            operation_label,
            operation_combo,
            param_label,
            param_spin,
            preview_frame,
            status,
        ),
    )
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
        status,
        start_x,
        start_y,
        nothing,
        "",
    )
    state.nodes[node_id] = node

    drag = GtkGestureDrag(drag_title)
    drag_origin = Ref((start_x, start_y))
    signal_connect(drag, "drag-begin") do _, _, _
        drag_origin[] = (node.x, node.y)
    end
    signal_connect(drag, "drag-update") do _, dx, dy
        node.x = max(0.0, drag_origin[][1] + Float64(dx))
        node.y = max(0.0, drag_origin[][2] + Float64(dy))
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
    window = GtkApplicationWindow(app, "Pure Julia Image Processing Pipeline")
    set_gtk_property!(window, :default_width, 1180)
    set_gtk_property!(window, :default_height, 760)

    root = GtkBox(:v, 10)
    for margin in (:margin_top, :margin_bottom, :margin_start, :margin_end)
        set_gtk_property!(root, margin, 12)
    end

    toolbar = GtkBox(:h, 8)
    add_button = GtkButton("Add Component")
    render_button = GtkButton("Recompute Graph")
    save_button = GtkButton("Save Pipeline")
    load_button = GtkButton("Load Pipeline")
    config_path = GtkEntry()
    set_gtk_property!(config_path, :text, "pipeline.toml")
    set_gtk_property!(config_path, :hexpand, true)
    source_picker = GtkComboBoxText()
    foreach(source -> push!(source_picker, source), ("Mandrill", "Lighthouse"))
    set_gtk_property!(source_picker, :active, 0)
    status = GtkLabel("Source: Mandrill")
    set_gtk_property!(status, :xalign, 0.0)

    foreach(
        widget -> push!(toolbar, widget),
        (
            add_button,
            render_button,
            save_button,
            load_button,
            GtkLabel("Config File"),
            config_path,
            GtkLabel("App Source"),
            source_picker,
            status,
        ),
    )

    canvas = GtkFixed()
    set_gtk_property!(canvas, :width_request, 2800)
    set_gtk_property!(canvas, :height_request, 1600)
    scroller = GtkScrolledWindow()
    set_gtk_property!(scroller, :hexpand, true)
    set_gtk_property!(scroller, :vexpand, true)
    scroller[] = canvas

    state = AppState(canvas, Dict{Int, PipelineNode}(), 1, load_test_source("Mandrill"), status)

    signal_connect(add_button, "clicked") do _
        slot = state.next_id - 1
        make_node_component!(
            state,
            30.0 + (slot % 4) * 300.0,
            30.0 + floor(slot / 4) * 320.0,
        )
    end
    signal_connect(render_button, "clicked") do _
        recompute_all!(state)
        set_gtk_property!(status, :label, "Graph recomputed")
    end
    signal_connect(save_button, "clicked") do _
        path = strip(String(get_gtk_property(config_path, :text, String)))
        if isempty(path)
            set_gtk_property!(status, :label, "Save failed: empty config path")
            return
        end
        try
            save_pipeline!(state, path, combo_text(source_picker, "Mandrill"))
            set_gtk_property!(status, :label, "Saved pipeline: $path")
        catch error
            set_gtk_property!(status, :label, "Save failed: $(sprint(showerror, error))")
        end
    end
    signal_connect(load_button, "clicked") do _
        path = strip(String(get_gtk_property(config_path, :text, String)))
        if !isfile(path)
            set_gtk_property!(status, :label, "Load failed: file not found")
            return
        end
        try
            load_pipeline!(state, path, source_picker)
            set_gtk_property!(status, :label, "Loaded pipeline: $path")
        catch error
            set_gtk_property!(status, :label, "Load failed: $(sprint(showerror, error))")
        end
    end
    signal_connect(source_picker, "changed") do picker
        set_app_source!(state, picker, combo_text(picker, "Mandrill"))
        recompute_all!(state)
    end

    push!(root, toolbar)
    push!(root, scroller)
    window[] = root
    show(window)
    make_node_component!(state, 30.0, 30.0)
end

function main()
    app = GtkApplication("com.dietrick.pure-julia-image-processing-pipeline")
    signal_connect(build_ui, app, "activate")
    Gtk4.run(app)
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
