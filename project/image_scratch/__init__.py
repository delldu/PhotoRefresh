"""Detect Scratch Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 12月 14日 星期二 00:22:28 CST
# ***
# ************************************************************************************/
#

__version__ = "1.0.0"

import os
from tqdm import tqdm
from PIL import Image

import torch
import redos
import todos
from . import unet
import torchvision.transforms as T
import pdb


# INPUT_IMAGE_TIMES = 16


def get_model():
    """Create model."""

    model = unet.UNet(
        in_channels=1,
        out_channels=1,
        depth=4,
        conv_num=2,
        wf=6,
        padding=True,
    )

    cdir = os.path.dirname(__file__)
    model_path = "models/image_scratch.pth"
    checkpoint = model_path if cdir == "" else cdir + "/" + model_path

    device = todos.model.get_device()
    todos.model.load(model, checkpoint)
    model = model.to(device)
    model.eval()

    print(f"Running on {device} ...")
    model = torch.jit.script(model)

    todos.data.mkdir("output")
    if not os.path.exists("output/image_scratch.torch"):
        model.save("output/image_scratch.torch")

    return model, device


def load_tensor(input_file):
    image = Image.open(input_file).convert("L")
    return T.ToTensor()(image).unsqueeze(0)


# def model_forward(model, device, gray_input_tensor):
#     # zeropad for model

#     # convert tensor from 1x4xHxW to 1x1xHxW
#     H, W = gray_input_tensor.size(2), gray_input_tensor.size(3)
#     if H % INPUT_IMAGE_TIMES == 0 and W % INPUT_IMAGE_TIMES == 0:
#         # output_tensor.size() -- [1, 1, 1024, 1024]
#         return todos.model.forward(model, device, gray_input_tensor)

#     # else
#     gray_input_tensor = todos.data.zeropad_tensor(gray_input_tensor, times=INPUT_IMAGE_TIMES)
#     output_tensor = todos.model.forward(model, device, gray_input_tensor)
#     return output_tensor[:, :, 0:H, 0:W]


def model_forward(model, device, input_tensor, multi_times=16):
    # zeropad for model
    H, W = input_tensor.size(2), input_tensor.size(3)
    if H % multi_times != 0 or W % multi_times != 0:
        input_tensor = todos.data.zeropad_tensor(input_tensor, times=multi_times)

    torch.cuda.synchronize()
    with torch.jit.optimized_execution(False):
        output_tensor = todos.model.forward(model, device, input_tensor)
    torch.cuda.synchronize()

    return output_tensor[:, :, 0:H, 0:W]


def image_client(name, input_files, output_dir):
    redo = redos.Redos(name)
    cmd = redos.image.Command()
    image_filenames = todos.data.load_files(input_files)
    for filename in image_filenames:
        output_file = f"{output_dir}/{os.path.basename(filename)}"
        context = cmd.scratch(filename, output_file)
        redo.set_queue_task(context)
    print(f"Created {len(image_filenames)} tasks for {name}.")
    print(redo)


def image_server(name, HOST="localhost", port=6379):
    # load model
    model, device = get_model()

    def do_service(input_file, output_file, targ):
        print(f"  detect {input_file} ...")
        try:
            input_tensor = todos.data.load_tensor(input_file)
            gray_tensor = load_tensor(input_file)
            output_tensor = model_forward(model, device, gray_tensor)
            blend_tensor = torch.cat([input_tensor, output_tensor.cpu()], dim=1)
            todos.data.save_tensor(blend_tensor, output_file)
            return True
        except Exception as e:
            print("Error: ", e)
            return False

    return redos.image.service(name, "image_scratch", do_service, HOST, port)


def image_predict(input_files, output_dir):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_model()

    def do_service(input_file, output_file, targ):
        try:
            input_tensor = todos.data.load_tensor(filename)
            gray_tensor = load_tensor(filename)
            output_tensor = model_forward(model, device, gray_tensor)
            blend_tensor = torch.cat([input_tensor, output_tensor.cpu()], dim=1)
            todos.data.save_tensor(blend_tensor, output_file)

            return True
        except Exception as e:
            print("Error: ", e)
            return False

    # load files
    image_filenames = todos.data.load_files(input_files)

    # start predict
    progress_bar = tqdm(total=len(image_filenames))
    for filename in image_filenames:
        progress_bar.update(1)
        output_file = f"{output_dir}/{os.path.basename(filename)}"
        do_service(filename, output_file, None)


def video_service(input_file, output_file, targ):
    # load video
    video = redos.video.Reader(input_file)
    if video.n_frames < 1:
        print(f"Read video {input_file} error.")
        return False

    # Create directory to store result
    output_dir = output_file[0 : output_file.rfind(".")]
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_model()

    print(f"  detect {input_file}, save to {output_file} ...")
    progress_bar = tqdm(total=video.n_frames)

    def clean_video_frame(no, data):
        # print(f"frame: {no} -- {data.shape}")
        progress_bar.update(1)

        input_tensor = todos.data.frame_totensor(data)
        input_tensor = input_tensor[:, 0:3, :, :]
        gray_tensor = input_tensor.mean(dim=1, keepdim=True)
        temp_output_file = "{}/{:06d}.png".format(output_dir, no)
        output_tensor = model_forward(model, device, gray_tensor)
        blend_tensor = torch.cat([input_tensor, output_tensor.cpu()], dim=1)
        todos.data.save_tensor(blend_tensor, temp_output_file)

    video.forward(callback=clean_video_frame)

    redos.video.encode(output_dir, output_file)

    # delete temp files
    for i in range(video.n_frames):
        temp_output_file = "{}/{:06d}.png".format(output_dir, i)
        os.remove(temp_output_file)

    return True


def video_client(name, input_file, output_file):
    cmd = redos.video.Command()
    context = cmd.scratch(input_file, output_file)
    redo = redos.Redos(name)
    redo.set_queue_task(context)
    print(f"Created 1 video tasks for {name}.")


def video_server(name, HOST="localhost", port=6379):
    return redos.video.service(name, "video_scratch", video_service, HOST, port)


def video_predict(input_file, output_file):
    return video_service(input_file, output_file, None)
