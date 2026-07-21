module SimpleOpenCV

using OpenCV
using ImageCore, TestImages

img_orig = testimage("mandrill.tiff")

img_raw = collect(rawview(channelview(img_orig)))

img_gray = OpenCV.cvtColor(img_raw, OpenCV.COLOR_RGB2GRAY)

img_blur = OpenCV.blur(img_gray, OpenCV.Size(Int32(9), Int32(9)))

img_bgr = OpenCV.cvtColor(img_raw, OpenCV.COLOR_RGB2BGR)

OpenCV.namedWindow("Original", OpenCV.WINDOW_AUTOSIZE)
OpenCV.namedWindow("Blurred Gray", OpenCV.WINDOW_AUTOSIZE)

OpenCV.imshow("Original", img_bgr)
OpenCV.imshow("Blurred Gray", img_blur)

println("Click an image window, then press any key to close...")
OpenCV.waitKey(0)
OpenCV.destroyAllWindows()


end # module SimpleOpenCV
