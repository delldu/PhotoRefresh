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

import torch
import todos
from . import unet
import pdb


def get_scratch_model():
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


def image_predict(input_files, output_dir):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_scratch_model()

    def do_service(input_file, output_file, targ):
        try:
            input_tensor = todos.data.load_tensor(filename)
            output_tensor = todos.model.forward(model, device, input_tensor)
            todos.data.save_tensor(output_tensor, output_file)

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
