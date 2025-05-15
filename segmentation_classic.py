
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from pathlib import Path




def segmentate(img):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    # blurred = cv.medianBlur(gray,11)
    blurred = cv.bilateralFilter(gray,9,75,75)
    edges = cv.Canny(blurred, threshold1=30, threshold2=100)

    kernel = np.ones((9,9),np.uint8)
    edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)

    contours, hierarchy = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv.contourArea)

    empty = np.zeros(img.shape, np.uint8)
    edge = cv.cvtColor(cv.drawContours(empty, [largest_contour], 0, (255,255,255), thickness=cv.FILLED),cv.COLOR_BGR2GRAY)

    return edge



if __name__ == "__main__":
    data_path = Path("D:/Olympus_magMag234.85_2025-02-26/DiOC6")


    file_paths = sorted(file_path for file_path in data_path.iterdir()
            if
                file_path.suffix == ".tif" and
                "scale" not in file_path.name and
                file_path.is_file()
            )


    ID = 0
    dataset_range = len(file_paths) // 2
    assert ID >= 0 and ID < dataset_range, "invalid scan index"


    if ID % 2 == 1:
        light_on_id, light_off_id = 2*ID, 2*ID + 1
    else:
        light_on_id, light_off_id = 2*ID + 1, 2*ID


    light_on = cv.imread(str(file_paths[light_on_id]))
    light_off = cv.imread(str(file_paths[light_off_id]))
    assert light_on is not None, f"file {file_paths[light_on_id]} could not be read"
    assert light_off is not None, f"file {file_paths[light_off_id]} could not be read"




    ###################################################################################

    img = light_on
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    # blurred = cv.medianBlur(gray,11)
    blurred = cv.bilateralFilter(gray,9,75,75)
    edges = cv.Canny(blurred, threshold1=30, threshold2=100)

    kernel = np.ones((9,9),np.uint8)
    edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)

    contours, hierarchy = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    empty = np.zeros(img.shape, np.uint8)
    edge = cv.cvtColor(cv.drawContours(empty, contours, 1, (255,255,255), thickness=cv.FILLED),cv.COLOR_BGR2GRAY)

    # M = cv.moments(contours[1])
    # if M["m00"] != 0:  # Avoid division by zero
    #     cx = int(M["m10"] / M["m00"])
    #     cy = int(M["m01"] / M["m00"])
    #     cv.circle(edge, (cx, cy), 5, (0, 0, 255), -1)

    # cv.imshow("edge with center", edge)
    # cv.waitKey(1)

    mask = edge

    x, y, w, h = cv.boundingRect(contours[1])
    roi_mask = mask[y:y+h, x:x+w]
    roi = light_off.copy()[y:y+h, x:x+w]
    roi[roi_mask != 255] = [0, 0, 0]
    roi = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)




    ###################################################################################








    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    axes[0, 0].imshow(light_on, cmap='gray')
    axes[0, 0].set_title(f'{file_paths[light_off_id].name}')

    axes[0, 1].imshow(light_off)
    axes[0, 1].set_title(f'{file_paths[light_off_id].name}')

    axes[1, 0].imshow(mask, cmap='gray')
    axes[1, 0].set_title('mask')

    axes[1, 1].imshow(roi, cmap='gray')
    axes[1, 1].set_title('roi')

    plt.tight_layout()
    plt.show()





