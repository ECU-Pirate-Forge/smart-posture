using Test
using ImageCore
using TestImages

include(joinpath(@__DIR__, "..", "PureImageOperations.jl"))
using .PureImageOperations

@testset "Pure Julia image operations" begin
    image = testimage("mandrill.tiff")

    @test apply_operation(image, "Identity", 1) === image
    @test eltype(apply_operation(image, "Gray", 1)) <: Gray
    @test size(apply_operation(image, "Blur", 3)) == size(image)
    @test size(apply_operation(image, "Canny", 30)) == size(image)
    @test eltype(apply_operation(image, "Threshold", 30)) <: Gray
    @test size(apply_operation(image, "Rotate", 1)) == reverse(size(image))
    @test size(apply_operation(image, "Resize", 5)) == (256, 256)
    @test size(apply_operation(image, "Negative", 1)) == size(image)
    @test size(apply_operation(image, "Erode", 3)) == size(image)
    @test size(apply_operation(image, "Dilation", 3)) == size(image)

    black = fill(Gray(0.0), 5, 5)
    black[3, 3] = Gray(1.0)
    @test Gray(0.0) == apply_operation(black, "Erode", 3)[3, 3]
    @test Gray(1.0) == apply_operation(black, "Dilation", 3)[2, 2]
    @test Gray(1.0) == apply_operation(black, "Negative", 1)[1, 1]
end
