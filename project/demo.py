import image_scratch

image_scratch.image_client("PAI", "images/*.png", "output/image")
image_scratch.image_server("PAI")

image_scratch.video_client("PAI", "/home/dell/noise.mp4", "output/video.mp4")
image_scratch.video_server("PAI")

image_scratch.image_predict("images/*.png", "output")
image_scratch.video_predict("/home/dell/noise.mp4", "output/predict.mp4")

