import numpy as np
from aicspylibczi import CziFile
from pathlib import Path
import sys
import argparse
import cv2


def load_czi_img(sci_file_path: Path) -> np.ndarray: # sci/czi file loader
    sci_file = CziFile(sci_file_path)
    img_block, dims_list = sci_file.read_image(S=0, Z=0)  
    img_squeezed = np.squeeze(img_block)

    x_spacing = float(sci_file.meta.find("Metadata/Scaling/Items/Distance[@Id='X']/Value").text)
    y_spacing = float(sci_file.meta.find("Metadata/Scaling/Items/Distance[@Id='Y']/Value").text)
    voxel_spacing = [y_spacing, x_spacing] # meters to millimeters

    return img_squeezed, voxel_spacing


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read image and save it as numpy")
    parser.add_argument("--input", required=True, help="Path to input image, czi format")
    parser.add_argument("--output", required=True, help="Path to output, npy format")
    args = parser.parse_args()

    img, spacing = load_czi_img(args.input)

    np.save(args.output, img)
    # np.save(sys.stdout.buffer, img) # save numpy array into stdout buffer

    # _, encoded = cv2.imencode('.png', img)
    # sys.stdout.buffer.write(encoded.tobytes())

