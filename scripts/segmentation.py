import sys
import argparse
import numpy as np
from cellpose import models, core
import cv2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Take img path, returns cellpose segm")
    parser.add_argument("--input_path", required=True, help="Path to inpit image, npy format")
    parser.add_argument("--output_path", required=True, help="Path to output mask, png format")
    args = parser.parse_args()

    output_path = args.output_path

    img = np.load(args.input_path)
    # img = np.load(sys.stdin.buffer) # load image from input buffer

    #Check if GPU access
    if core.use_gpu()==False:
        raise ImportError("No GPU access, change your runtime")

    model = models.CellposeModel(gpu=True)


    # parameter description: https://cellpose.readthedocs.io/en/latest/settings.html
    masks, flows, styles = model.eval(
        img,
        diameter=210,
        flow_threshold=0.5,
        cellprob_threshold=0.5,
        channels=[0, 0]
    )
    masks = np.astype(masks, np.uint8)

    cv2.imwrite(output_path, masks)



