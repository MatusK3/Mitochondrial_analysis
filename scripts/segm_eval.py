import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import json




segmentation_methode = "xenia"
GT_DIR_PATH = "data/mito/mask/"       
PRED_DIR_PATH = f"results/segmentation/{segmentation_methode}/"
IOU_THRESHOLDS = [0.5, 0.75]
EVAL_OUTPUT = f"results/segmentation/{segmentation_methode}_eval.csv"






# def load_segm_instances(image_path):
#     if not os.path.exists(image_path):
#         return []
    
#     png_masks = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # Read as grayscale
#     if png_masks is None:
#         return []

#     n_channels = 1 + np.max(png_masks)
#     channels = np.eye(n_channels, dtype=bool)[png_masks]
#     channels = np.rollaxis(channels, 2)
#     masks = channels[1:] # ignore background
    
#     return masks

# def compute_iou_matrix(gt_masks, pred_masks):
#     """
#     Computes IoU between every GT mask and every Predicted mask.
#     Returns: Matrix of shape (len(gt), len(pred))
#     """
#     if len(gt_masks) == 0 or len(pred_masks) == 0:
#         return np.zeros((len(gt_masks), len(pred_masks)))

#     iou_matrix = np.zeros((len(gt_masks), len(pred_masks)))

#     for i, gt in enumerate(gt_masks):
#         gt_area = np.sum(gt)
#         for j, pred in enumerate(pred_masks):
#             pred_area = np.sum(pred)
            
#             # Intersection
#             intersection = np.sum(gt & pred)
            
#             # Union
#             union = gt_area + pred_area - intersection
            
#             if union > 0:
#                 iou_matrix[i, j] = intersection / union
#             else:
#                 iou_matrix[i, j] = 0.0
                
#     return iou_matrix



def load_segm_instances(image_path):
    """
    Optimized mask loading. 
    Instead of one-hot encoding the whole range, we only process unique IDs present.
    Returns: Boolean array of shape (N_instances, Height * Width)
    """
    if not os.path.exists(image_path):
        return None

    # Load as grayscale (uint16 is safer for ID masks if IDs > 255, but uint8 is common)
    mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if mask is None:
        return None
    
    # Handle potentially 3 channel images by taking one channel
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    # Get unique instance IDs, ignoring background (0)
    instance_ids = np.unique(mask)
    instance_ids = instance_ids[instance_ids != 0]

    if len(instance_ids) == 0:
        return np.array([])

    # Vectorized mask creation: (N, H, W) -> Flatten to (N, H*W) for fast dot product later
    # This uses broadcasting to create boolean masks for all IDs at once
    flat_mask = mask.flatten()
    binary_masks = (flat_mask == instance_ids[:, None])
    
    return binary_masks


def compute_iou_matrix(gt_masks, pred_masks):
    # Check if either mask set is empty using shape
    n_gt = gt_masks.shape[0] if gt_masks.size > 0 else 0
    n_pred = pred_masks.shape[0] if pred_masks.size > 0 else 0

    if n_gt == 0 or n_pred == 0:
        return np.zeros((n_gt, n_pred))

    # Promote to int64 during dot product to handle large pixel counts
    # intersection shape: (N_gt, N_pred)
    intersection = np.dot(gt_masks.astype(np.int64), pred_masks.astype(np.int64).T)

    # Compute areas
    gt_areas = np.sum(gt_masks, axis=1)    
    pred_areas = np.sum(pred_masks, axis=1) 

    # Union = Area_A + Area_B - Intersection
    union = gt_areas[:, None] + pred_areas[None, :] - intersection

    # Calculate IoU
    iou_matrix = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union!=0)
    
    return iou_matrix


def match_instances(iou_matrix, threshold):
    """
    Matches GT to Pred using Greedy strategy based on max IoU.
    Returns: TP, FP, FN counts
    """
    if iou_matrix.size == 0:
        # Handle edge cases where one list is empty
        # If matrix is (N, 0) -> N False Negatives
        # If matrix is (0, M) -> M False Positives
        num_gt = iou_matrix.shape[0]
        num_pred = iou_matrix.shape[1]
        return 0, num_pred, num_gt

    # Work on a copy so we can mask out matched indices
    matrix = iou_matrix.copy()
    num_gt, num_pred = matrix.shape
    
    tp = 0
    # Greedy matching: Find max IoU, match if > threshold, remove, repeat
    while True:
        # Find max value in matrix
        max_idx = np.unravel_index(np.argmax(matrix, axis=None), matrix.shape)
        max_iou = matrix[max_idx]

        if max_iou < threshold:
            break

        # Record match
        tp += 1
        gt_idx, pred_idx = max_idx
        
        # "Remove" this row and col by setting to -1
        matrix[gt_idx, :] = -1
        matrix[:, pred_idx] = -1

    fp = num_pred - tp
    fn = num_gt - tp
    return tp, fp, fn


def compute_pixel_metrics(gt_masks, pred_masks):
    """
    Computes standard semantic metrics regardless of object count.
    gt_mask_full: The original grayscale image from dir A
    pred_mask_full: The original grayscale image from dir B
    """
    # Convert to binary (any value > 0 is "Object")
    gt_binary = np.all(gt_masks, axis=0)#(gt_mask_full > 0).astype(np.uint8)
    pred_binary = np.all(pred_masks, axis=0)#(pred_mask_full > 0).astype(np.uint8)

    intersection = np.logical_and(gt_binary, pred_binary).sum()
    union = np.logical_or(gt_binary, pred_binary).sum()
    
    pixel_iou = intersection / union if union > 0 else 1.0
    return pixel_iou


def main():
    # load class names
    DATSET_PATH = "data/mito"
    config_path = f"{DATSET_PATH}/config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    classes = config["classes"]



    print(f"Starting Evaluation...")
    print(f"GT Directory: {GT_DIR_PATH}")
    print(f"Pred Directory: {PRED_DIR_PATH}")
    
    # Store results: class -> threshold -> {TP, FP, FN}
    results = {}
    pixel_ious = {}

    for class_name in classes:
        class_gt_path = os.path.join(GT_DIR_PATH, class_name)
        class_pred_path = os.path.join(PRED_DIR_PATH, class_name)
        
        # Initialize stats for this class
        if class_name not in results:
            results[class_name] = {t: {'tp': 0, 'fp': 0, 'fn': 0} for t in IOU_THRESHOLDS}
            pixel_ious[class_name] = []

        gt_files = [f for f in os.listdir(class_gt_path)]# if f.lower().endswith('.png')]
        
        print(f"\nProcessing class: {class_name} ({len(gt_files)} images)")

        for filename in tqdm(gt_files):
            file_gt_path = os.path.join(class_gt_path, filename)
            file_pred_path = os.path.join(class_pred_path, filename)

            # Load Masks
            gt_masks = load_segm_instances(file_gt_path)
            pred_masks = load_segm_instances(file_pred_path) # Returns empty if file missing

            if pred_masks is None: # skip if prediction was not jet created
                continue
            
            pixel_ious[class_name].append(compute_pixel_metrics(gt_masks, pred_masks))


            # Compute IoU Matrix once for the image pair
            iou_mat = compute_iou_matrix(gt_masks, pred_masks)

            # Evaluate for each threshold
            for t in IOU_THRESHOLDS:
                tp, fp, fn = match_instances(iou_mat, t)
                results[class_name][t]['tp'] += tp
                results[class_name][t]['fp'] += fp
                results[class_name][t]['fn'] += fn


    # ================= REPORTING =================
    print("\n" + "="*60)
    print("FINAL EVALUATION REPORT")
    print("="*60)

    overall_stats = {t: {'tp': 0, 'fp': 0, 'fn': 0} for t in IOU_THRESHOLDS}
    
    # Create a list for DataFrame
    data_rows = []

    for class_name, metrics in results.items():
        for t in IOU_THRESHOLDS:
            tp = metrics[t]['tp']
            fp = metrics[t]['fp']
            fn = metrics[t]['fn']
            
            # Aggregate for overall
            overall_stats[t]['tp'] += tp
            overall_stats[t]['fp'] += fp
            overall_stats[t]['fn'] += fn
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            mean_pixel_iou = np.mean(pixel_ious[class_name])

            data_rows.append({
                "Class": class_name,
                "Mean_Pixel_IoU": round(mean_pixel_iou, 4),
                "IoU Threshold": t,
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
                "F1 Score": round(f1, 4),
                "TP": tp, "FP": fp, "FN": fn,
            })

    # Add Overall Rows
    for t in IOU_THRESHOLDS:
        tp = overall_stats[t]['tp']
        fp = overall_stats[t]['fp']
        fn = overall_stats[t]['fn']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        mean_pixel_iou = np.mean(np.concatenate(list(pixel_ious.values())))

        data_rows.append({
            "Class": "___OVERALL___",
            "Mean_Pixel_IoU": round(mean_pixel_iou, 4),
            "IoU Threshold": t,
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1 Score": round(f1, 4),
            "TP": tp, "FP": fp, "FN": fn,
        })

    df = pd.DataFrame(data_rows)
    # Sort for nicer viewing
    df = df.sort_values(by=["IoU Threshold", "Class"])
    
    print(df.to_string(index=False))
    
    # Save to CSV optionally
    df.to_csv(EVAL_OUTPUT, index=False)

if __name__ == "__main__":
    main()








# -CELLPOSE-----------------------------------------------------------------------------
#         Class  Mean_Pixel_IoU  IoU Threshold  Precision  Recall  F1 Score   TP  FP  FN
# Acetate_DAY_1          0.9630           0.50     0.8457  0.9580    0.8984  137  25   6
# Acetate_DAY_3          0.9663           0.50     0.9800  0.9932    0.9866  147   3   1
#      SD_DAY_1          0.9652           0.50     0.9528  0.9603    0.9565  121   6   5
#      SD_DAY_3          0.8018           0.50     0.9401  0.9515    0.9458  157  10   8
#           YPD          0.9593           0.50     0.9624  0.9343    0.9481  128   5   9
#     YPD_DAY_1          0.9688           0.50     0.9516  0.9516    0.9516  118   6   6
#     YPD_DAY_3          0.8131           0.50     0.7387  0.8817    0.8039   82  29  11
#         YPGal          0.9994           0.50     0.9219  0.9267    0.9243  177  15  14
#         YPGly          0.9985           0.50     0.9650  0.9020    0.9324  138   5  15
# ___OVERALL___          0.9395           0.50     0.9206  0.9414    0.9309 1205 104  75
#
# Acetate_DAY_1          0.9630           0.75     0.8210  0.9301    0.8721  133  29  10
# Acetate_DAY_3          0.9663           0.75     0.9800  0.9932    0.9866  147   3   1
#      SD_DAY_1          0.9652           0.75     0.9213  0.9286    0.9249  117  10   9
#      SD_DAY_3          0.8018           0.75     0.6826  0.6909    0.6867  114  53  51
#           YPD          0.9593           0.75     0.9549  0.9270    0.9407  127   6  10
#     YPD_DAY_1          0.9688           0.75     0.9435  0.9435    0.9435  117   7   7
#     YPD_DAY_3          0.8131           0.75     0.6216  0.7419    0.6765   69  42  24
#         YPGal          0.9994           0.75     0.9219  0.9267    0.9243  177  15  14
#         YPGly          0.9985           0.75     0.8951  0.8366    0.8649  128  15  25
# ___OVERALL___          0.9395           0.75     0.8625  0.8820    0.8722 1129 180 151



# -Xenia-------------------------------------------------------------------------------
#         Class  Mean_Pixel_IoU  IoU Threshold  Precision  Recall  F1 Score  TP  FP  FN
# Acetate_DAY_1             NaN           0.50     0.0000  0.0000    0.0000   0   0   0
# Acetate_DAY_3             NaN           0.50     0.0000  0.0000    0.0000   0   0   0
#      SD_DAY_1             NaN           0.50     0.0000  0.0000    0.0000   0   0   0
#      SD_DAY_3             NaN           0.50     0.0000  0.0000    0.0000   0   0   0
#           YPD             NaN           0.50     0.0000  0.0000    0.0000   0   0   0
#     YPD_DAY_1             NaN           0.50     0.0000  0.0000    0.0000   0   0   0
#     YPD_DAY_3             NaN           0.50     0.0000  0.0000    0.0000   0   0   0
#         YPGal          0.9960           0.50     0.8493  0.7470    0.7949  62  11  21
#         YPGly          0.9913           0.50     0.9194  0.8143    0.8636  57   5  13
# ___OVERALL___          0.9937           0.50     0.8815  0.7778    0.8264 119  16  34
#
# Acetate_DAY_1             NaN           0.75     0.0000  0.0000    0.0000   0   0   0
# Acetate_DAY_3             NaN           0.75     0.0000  0.0000    0.0000   0   0   0
#      SD_DAY_1             NaN           0.75     0.0000  0.0000    0.0000   0   0   0
#      SD_DAY_3             NaN           0.75     0.0000  0.0000    0.0000   0   0   0
#           YPD             NaN           0.75     0.0000  0.0000    0.0000   0   0   0
#     YPD_DAY_1             NaN           0.75     0.0000  0.0000    0.0000   0   0   0
#     YPD_DAY_3             NaN           0.75     0.0000  0.0000    0.0000   0   0   0
#         YPGal          0.9960           0.75     0.8219  0.7229    0.7692  60  13  23
#         YPGly          0.9913           0.75     0.8548  0.7571    0.8030  53   9  17
# ___OVERALL___          0.9937           0.75     0.8370  0.7386    0.7847 113  22  40