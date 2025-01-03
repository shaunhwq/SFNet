import os
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from Image_dehazing.models.SFNet import SFNet


def pre_process(image: np.array, device: str, factor: int = 8) -> torch.Tensor:
    """
    :param image: Input image to transform to the model input
    :param device: Device to send input to
    :returns: Tensor input to model, in the shape [b, c, h, w]
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb = np.ascontiguousarray(image_rgb.astype(np.float32)) / 255.0
    image_tensor = torch.from_numpy(image_rgb.transpose(2, 0, 1))
    image_tensor = image_tensor.unsqueeze(0).to(device)

    _, _, h, w = image_tensor.shape
    H, W = ((h + factor) // factor) * factor, ((w + factor) // factor * factor)
    padh = H - h if h % factor != 0 else 0
    padw = W - w if w % factor != 0 else 0
    image_tensor = F.pad(image_tensor, (0, padw, 0, padh), 'reflect')

    return image_tensor


def post_process(model_output: torch.Tensor, input_hw: Tuple[int, int]) -> np.array:
    """
    :param model_output: Output tensor produced by the model [b, c, h, w]
    :param input_hw: Tuple containing input image height and width
    :returns: Output image which can be displayed by OpenCV
    """
    h, w = input_hw
    model_output = model_output[:,:,:h,:w]
    image_rgb = model_output.squeeze(0).cpu().permute(1, 2, 0).numpy()
    image_rgb = (image_rgb * 255).clip(0, 255).astype(np.uint8)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return image_bgr


if __name__ == "__main__":
    video_path = "/Users/shaun/datasets/image_enhancement/dehaze/DVD/DrivingHazy/31_hazy_video.mp4"
    device = "cpu"
    weights_path = "weights/SOTS_Indoor.pkl"

    # SOTS Indoor (620, 480), Outdoor (550x483)
    model = SFNet(mode=["test", "Indoor"])
    weights = torch.load(weights_path, map_location="cpu")["model"]
    model.load_state_dict(weights)
    model.to(device)
    model.eval()

    cap = cv2.VideoCapture(video_path)

    while True:
        frame_no = cap.get(cv2.CAP_PROP_POS_FRAMES)
        ret, frame = cap.read()

        frame = cv2.resize(frame, (620, 480))

        in_tensor = pre_process(frame, device)
        with torch.no_grad():
            model_outputs = model(in_tensor)
        out_image = post_process(model_outputs[2], frame.shape[:2])

        display_image = np.vstack([frame, out_image])

        cv2.imshow("output", display_image)
        key = cv2.waitKey(1)
        if key & 255 == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
