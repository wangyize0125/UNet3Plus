#! python3
# encoding: utf-8
"""
@author:    Yize Wang
@contact:    wangyize@hust.edu.cn
@file:    predict.py
@time:   2022/3/17 18:06
@description:    
"""

import os
import argparse
import torch
import logging
import numpy as np

from unet import UNet, UNetV3
from PIL import Image
from utils.dataset import BasicDataset
from torchvision import transforms


def get_args():
    parser = argparse.ArgumentParser(description="Predict from input images")

    parser.add_argument("-g", "--gpu_id", metavar="G", type=int, default=0, help="Number of GPU")
    parser.add_argument("-u", "--unet_type", metavar="U", default="v1", help="UNet type v1/v2/v3")
    parser.add_argument("-m", "--model", default="MODEL.pth", metavar="FILE", help="Specify the model file")
    parser.add_argument("-i", "--input", metavar="INPUT", help="filenames of input images", required=True)
    parser.add_argument("-o", "--output", metavar="OUTPUT", help="filenames of output images")
    parser.add_argument("-n", "--no_save", action="store_true", help="Do not save the output", default=False)
    parser.add_argument("-t", "--threshold", type=float, help="Minimum probability value to consider a mask white", default=0.5)
    parser.add_argument("-s", "--scale", type=float, help="image scale factor when preprocessing", default=1.0)

    return parser.parse_args()


def predict_img(unet_type, net, full_img, device, scale_factor=1, out_threshold=0.5):
    net.eval()

    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        output = output.squeeze(0)

        tf = transforms.Compose([transforms.ToPILImage(), transforms.Resize(full_img.size[1]), transforms.ToTensor()])
        probs = tf(output.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return (full_mask > out_threshold).astype(np.float32).astype(np.uint8)


def mask_to_image(mask):
    return Image.fromarray((mask * 255))


if __name__ == "__main__":
    args = get_args()

    gpu_id = args.gpu_id
    unet_type = args.unet_type
    in_file_folder = args.input
    out_file_folder = args.output

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if unet_type == "v2":
        pass
    elif unet_type =="v3":
        net = UNetV3(n_channels=3, n_classes=1)
    else:
        net = UNet(n_channels=3, n_classes=1)

    logging.info("Loading model {}".format(args.model))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info("Using device {}".format(device))
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))
    logging.info("Model loaded!")

    for root, dirs, files in os.walk(args.input):
        for file in files:
            if file.endswith(".png"):
                logging.info("\nPredicting image {} ...".format(file))
                img = Image.open(os.path.join(args.input, file))
                mask = predict_img(unet_type=unet_type, net=net, full_img=img, scale_factor=args.scale, out_threshold=args.threshold, device=device)

                if not args.no_save:
                    result = mask_to_image(mask)
                    result.save(os.path.join(args.output, file))
                    logging.info("Mask saved to {}".format(os.path.join(args.output, file)))
