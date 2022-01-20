## README

image scratch package 1.0.0

### 1. How to use ?
demo.py

### 2. Reference
```
import todos
import image_scratch

device = todos.get_device()

d = image_scratch.detector(device)
gray_tensor = image_scratch.load_gray_tensor(input_file)
output_tensor = image_scratch.model_forward(d, device, gray_tensor)
todos.save_tensor(output_tensor, output_file)
```


