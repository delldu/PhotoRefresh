import detect_scratch

detect_scratch.image_client("PAI", "images/*.png", "output/image")
detect_scratch.image_server("PAI")

detect_scratch.video_client("PAI", "/home/dell/noise.mp4", "output/video.mp4")
detect_scratch.video_server("PAI")

detect_scratch.image_predict("images/*.png", "output")
detect_scratch.video_predict("/home/dell/noise.mp4", "output/predict.mp4")

