module PureImageOperations

using ImageCore

export OPERATIONS, apply_operation, ensure_gray

const OPERATIONS = [
    "Identity",
    "Gray",
    "Blur",
    "Canny",
    "Threshold",
    "Rotate",
    "Resize",
    "Negative",
    "Erode",
    "Dilation",
]

function image_channels(img::AbstractMatrix{<:Colorant})
    viewed = channelview(img)
    if ndims(viewed) == 2
        return reshape(Float64.(viewed), 1, size(viewed)...)
    end
    return Float64.(viewed)
end

image_channels(img::AbstractMatrix{<:Real}) =
    reshape(Float64.(img), 1, size(img)...)

function channels_image(channels::AbstractArray{<:Real, 3})
    count, height, width = size(channels)
    if count == 1
        return [Gray(clamp(channels[1, y, x], 0.0, 1.0)) for y in 1:height, x in 1:width]
    end

    return [
        RGB(
            clamp(channels[1, y, x], 0.0, 1.0),
            clamp(channels[2, y, x], 0.0, 1.0),
            clamp(channels[3, y, x], 0.0, 1.0),
        )
        for y in 1:height, x in 1:width
    ]
end

function gray_values(img)
    channels = image_channels(img)
    if size(channels, 1) == 1
        return dropdims(channels; dims=1)
    end

    return @views 0.299 .* channels[1, :, :] .+
                  0.587 .* channels[2, :, :] .+
                  0.114 .* channels[3, :, :]
end

ensure_gray(img) = Gray.(gray_values(img))

function box_blur_channel(channel::AbstractMatrix{<:Real}, kernel_size::Int)
    height, width = size(channel)
    radius = kernel_size ÷ 2
    integral = zeros(Float64, height + 1, width + 1)
    integral[2:end, 2:end] .= cumsum(cumsum(channel; dims=1); dims=2)
    output = Matrix{Float64}(undef, height, width)

    for x in 1:width, y in 1:height
        top = max(1, y - radius)
        bottom = min(height, y + radius)
        left = max(1, x - radius)
        right = min(width, x + radius)
        total = integral[bottom + 1, right + 1] -
                integral[top, right + 1] -
                integral[bottom + 1, left] +
                integral[top, left]
        output[y, x] = total / ((bottom - top + 1) * (right - left + 1))
    end
    return output
end

function box_blur(img, parameter::Int)
    kernel_size = max(1, parameter)
    kernel_size += iseven(kernel_size)
    channels = image_channels(img)
    output = similar(channels)
    for channel in axes(channels, 1)
        output[channel, :, :] .= box_blur_channel(
            @view(channels[channel, :, :]),
            kernel_size,
        )
    end
    return channels_image(output)
end

function sobel_gradients(gray::AbstractMatrix{<:Real})
    height, width = size(gray)
    magnitude = zeros(Float64, height, width)
    direction = zeros(Float64, height, width)

    for x in 2:(width - 1), y in 2:(height - 1)
        gx = -gray[y - 1, x - 1] + gray[y - 1, x + 1] -
             2gray[y, x - 1] + 2gray[y, x + 1] -
             gray[y + 1, x - 1] + gray[y + 1, x + 1]
        gy = gray[y - 1, x - 1] + 2gray[y - 1, x] + gray[y - 1, x + 1] -
             gray[y + 1, x - 1] - 2gray[y + 1, x] - gray[y + 1, x + 1]
        magnitude[y, x] = hypot(gx, gy)
        direction[y, x] = mod(rad2deg(atan(gy, gx)), 180)
    end
    return magnitude, direction
end

function suppress_nonmaxima(magnitude, direction)
    height, width = size(magnitude)
    output = zeros(Float64, height, width)

    for x in 2:(width - 1), y in 2:(height - 1)
        angle = direction[y, x]
        if angle < 22.5 || angle >= 157.5
            before, after = magnitude[y, x - 1], magnitude[y, x + 1]
        elseif angle < 67.5
            before, after = magnitude[y - 1, x + 1], magnitude[y + 1, x - 1]
        elseif angle < 112.5
            before, after = magnitude[y - 1, x], magnitude[y + 1, x]
        else
            before, after = magnitude[y - 1, x - 1], magnitude[y + 1, x + 1]
        end
        value = magnitude[y, x]
        output[y, x] = value >= before && value >= after ? value : 0.0
    end
    return output
end

function hysteresis_edges(magnitude, low::Float64, high::Float64)
    height, width = size(magnitude)
    edges = falses(height, width)
    queued = falses(height, width)
    queue = Tuple{Int, Int}[]

    for x in 1:width, y in 1:height
        if magnitude[y, x] >= high
            push!(queue, (y, x))
            queued[y, x] = true
        end
    end

    index = 1
    while index <= length(queue)
        y, x = queue[index]
        index += 1
        edges[y, x] = true
        for nx in max(1, x - 1):min(width, x + 1),
            ny in max(1, y - 1):min(height, y + 1)
            if !queued[ny, nx] && magnitude[ny, nx] >= low
                queued[ny, nx] = true
                push!(queue, (ny, nx))
            end
        end
    end
    return Gray.(edges)
end

function canny_edges(img, parameter::Int)
    smoothed = box_blur_channel(gray_values(img), 5)
    magnitude, direction = sobel_gradients(smoothed)
    thinned = suppress_nonmaxima(magnitude, direction)
    low = max(1, parameter) / 255
    return hysteresis_edges(thinned, low, 2low)
end

function threshold_image(img, parameter::Int)
    level = clamp(parameter * 4, 0, 255) / 255
    return Gray.(gray_values(img) .> level)
end

function resize_bilinear(img, parameter::Int)
    scale = clamp(parameter / 10, 0.1, 5.0)
    input = image_channels(img)
    count, old_height, old_width = size(input)
    new_height = max(1, round(Int, old_height * scale))
    new_width = max(1, round(Int, old_width * scale))
    output = Array{Float64}(undef, count, new_height, new_width)

    for x in 1:new_width, y in 1:new_height
        source_y = clamp((y - 0.5) / scale + 0.5, 1.0, old_height)
        source_x = clamp((x - 0.5) / scale + 0.5, 1.0, old_width)
        y1, x1 = floor(Int, source_y), floor(Int, source_x)
        y2, x2 = min(y1 + 1, old_height), min(x1 + 1, old_width)
        wy, wx = source_y - y1, source_x - x1
        for channel in 1:count
            output[channel, y, x] =
                (1 - wy) * ((1 - wx) * input[channel, y1, x1] + wx * input[channel, y1, x2]) +
                wy * ((1 - wx) * input[channel, y2, x1] + wx * input[channel, y2, x2])
        end
    end
    return channels_image(output)
end

function extrema_pass(input::AbstractMatrix{<:Real}, radius::Int, dim::Int, take_min::Bool)
    height, width = size(input)
    output = similar(input, Float64)
    for x in 1:width, y in 1:height
        range = dim == 1 ?
            @view(input[max(1, y - radius):min(height, y + radius), x]) :
            @view(input[y, max(1, x - radius):min(width, x + radius)])
        output[y, x] = take_min ? minimum(range) : maximum(range)
    end
    return output
end

function morphology(img, parameter::Int, take_min::Bool)
    kernel_size = max(1, parameter)
    kernel_size += iseven(kernel_size)
    radius = kernel_size ÷ 2
    input = image_channels(img)
    output = similar(input)
    for channel in axes(input, 1)
        horizontal = extrema_pass(@view(input[channel, :, :]), radius, 2, take_min)
        output[channel, :, :] .= extrema_pass(horizontal, radius, 1, take_min)
    end
    return channels_image(output)
end

function negative(img)
    channels = image_channels(img)
    return channels_image(1 .- channels)
end

function apply_operation(img, operation::String, parameter::Int)
    operation == "Identity" && return img
    operation == "Gray" && return ensure_gray(img)
    operation == "Blur" && return box_blur(img, parameter)
    operation == "Canny" && return canny_edges(img, parameter)
    operation == "Threshold" && return threshold_image(img, parameter)
    operation == "Negative" && return negative(img)
    operation == "Erode" && return morphology(img, parameter, true)
    operation == "Dilation" && return morphology(img, parameter, false)
    operation == "Resize" && return resize_bilinear(img, parameter)

    if operation == "Rotate"
        mode = mod(parameter - 1, 3) + 1
        return mode == 1 ? rotr90(img) : mode == 2 ? rot180(img) : rotl90(img)
    end
    return img
end

end
