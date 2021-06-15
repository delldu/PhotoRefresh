python detection.py \
	--test_path ../test_images/old_w_scratch \
	--output_dir output/masks \
	--input_size full_size \
	--GPU 0

convert output/masks/input/a.png output/masks/mask/a.png +append /tmp/a.png
convert output/masks/input/b.png output/masks/mask/b.png +append /tmp/b.png
convert output/masks/input/c.png output/masks/mask/c.png +append /tmp/c.png
convert output/masks/input/d.png output/masks/mask/d.png +append /tmp/d.png

# python test.py \
# 	--Scratch_and_Quality_restore \
# 	--test_input output/masks/input \
# 	--test_mask output/masks/mask \
# 	--outputs_dir output \
# 	--gpu_ids -1
