import cv2
import os
import numpy as np
import pandas as pd
import motmetrics as mm


def calculate_iou(box1, box2):
    """
    Calculate IoU between two boxes.
    box format: [x_center, y_center, width, height] (normalized).
    """
    x1_min, y1_min = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2
    x1_max, y1_max = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2

    x2_min, y2_min = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2
    x2_max, y2_max = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    iou = inter_area / (box1_area + box2_area - inter_area + 1e-6)
    return iou

def centroid_distances(box1,box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def convert_yolo_to_mota(image_folder, label_folder, output_file, iou_threshold=0.5, cent_threshold=0, stop_at_frame=None):
    """
    Convert YOLO labels to MOTA-compatible labels. Uses cent_threshold to determine if two objects are the same.
    VERY rudimentary and should only be used as a starting point for labeling. Manual work is required afterwards
    
    Args:
        image_folder: Path to the folder containing images.
        label_folder: Path to the folder containing YOLO labels.
        output_file: Path to save the MOTA-compatible labels.
        iou_threshold: IoU threshold for assigning the same ID.
    """
    files = sorted([f for f in os.listdir(label_folder) if f.endswith(".txt")])
    prev_objects = []
    object_id = 0
    results = []

    for frame_num, file in enumerate(files):
        if stop_at_frame is not None and frame_num >= stop_at_frame:
            break
        file_path = os.path.join(label_folder, file)
        with open(file_path, "r") as f:
            objects = [list(map(float, line.split()[1:])) for line in f.readlines()]
        
        current_objects = []
        for obj in objects:
            assigned = False
            for prev_obj in prev_objects:
                iou = calculate_iou(obj, prev_obj["box"])
                cent_dist = centroid_distances(obj, prev_obj["box"])
                if iou > iou_threshold or cent_dist < cent_threshold: #same object if close enough
                    # check if the id is already in current objects
                    
                    current_objects.append({"id": prev_obj["id"], "box": obj})
                    assigned = True
                    break
            
            if not assigned:
                object_id += 1
                current_objects.append({"id": object_id, "box": obj})
        
        # Save results for the current frame
        for obj in current_objects:
            x, y, w, h = obj["box"]
            results.append([frame_num, obj["id"], x, y, w, h, 1.0, 0, 1.0])  # confidence=1, class=0, visibility=1
        
        prev_objects = current_objects

    # Save results to a file
    columns = ["frame", "id", "x", "y", "w", "h", "conf", "class", "vis"]
    df = pd.DataFrame(results, columns=columns)
    df.to_csv(output_file, index=False, header=False)

def draw_labels(frame, mota_labels, frame_num, colour):
    # Get objects for the current frame
    objects = mota_labels[mota_labels["frame"] == frame_num]

    # Draw bounding boxes and object IDs
    for _, obj in objects.iterrows():
        x, y, w, h = obj["x"], obj["y"], obj["w"], obj["h"]
        obj_id = int(obj["id"])
        
        # Convert normalized YOLO format to pixel coordinates
        img_h, img_w, _ = frame.shape
        x1 = int((x - w / 2) * img_w)
        y1 = int((y - h / 2) * img_h)
        x2 = int((x + w / 2) * img_w)
        y2 = int((y + h / 2) * img_h)
        
        # Draw bounding box and ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(frame, f"ID: {obj_id}", (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
        idpos = -10 + int(h * img_h)
        # print(int(h))
        cv2.putText(frame, f"ID: {obj_id}", (x1, y1 + idpos),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
        
    return frame
def display_mota_labels(image_folder, mota_labels_file):
    """
    Display frames with bounding boxes and object IDs from MOTA labels.

    Args:
        image_folder: Path to the folder containing images.
        mota_labels_file: Path to the MOTA labels file.
    """
    # Load MOTA labels
    columns = ["frame", "id", "x", "y", "w", "h", "conf", "class", "vis"]
    mota_labels = pd.read_csv(mota_labels_file, header=None, names=columns)

    # Get unique frames
    frames = sorted(mota_labels["frame"].unique())
    frame_index = 0
    direction = 0 # 0 for forward, 1 for backward
    while True:
        # Load the current frame
        frame_num = frames[frame_index]
        frame_path = os.path.join(image_folder, f"000_{frame_num:06d}.png")
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Frame {frame_num:06d} not found.")
            if direction == 0:
                frame_index = (frame_index + 1) % len(frames)
            else:
                frame_index = (frame_index - 1) % len(frames)
            continue

        # Get objects for the current frame
        frame = draw_labels(frame, mota_labels, frame_num, (0, 255, 0))

        # Display the frame
        cv2.imshow("MOTA Visualization", frame)

        # Handle keyboard input
        key = cv2.waitKey(0)
        # print(key)
        if key == 27:  # ESC key to exit
            break
        elif key == 100:  # D arrow key
            frame_index = (frame_index + 1) % len(frames)
            direction = 0
        elif key == 97:  # A arrow key
            frame_index = (frame_index - 1) % len(frames)
            direction = 1

    cv2.destroyAllWindows()

def disp_compare_gt_pred(image_folder, gt_labels_file, pred_labels_file):
    # display side by side gt and pred, also going through images with A and D keys
    # Load MOTA labels
    columns = ["frame", "id", "x", "y", "w", "h", "conf", "class", "vis"]
    gt_labels = pd.read_csv(gt_labels_file, header=None, names=columns)
    pred_labels = pd.read_csv(pred_labels_file, header=None, names=columns)


    # Get unique frames
    frames = sorted(gt_labels["frame"].unique())
    frame_index = 0
    direction = 0 # 0 for forward, 1 for backward
    while True:
        gt_labels = pd.read_csv(gt_labels_file, header=None, names=columns)
        pred_labels = pd.read_csv(pred_labels_file, header=None, names=columns)
        # Load the current frame
        frame_num = frames[frame_index]
        frame_path = os.path.join(image_folder, f"000_{frame_num:06d}.png")
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Frame {frame_num:06d} not found.")
            if direction == 0:
                frame_index = (frame_index + 1) % len(frames)
            else:
                frame_index = (frame_index - 1) % len(frames)
            continue

        # Get objects for the current frame
        frame = draw_labels(frame, pred_labels, frame_num, (0, 0, 255))
        frame = draw_labels(frame, gt_labels, frame_num, (0, 255, 0))
        # draw frame number:

        #rotate frame
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        cv2.putText(frame, f"Frame: {frame_num:06d}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        #resize to fit screen:
        frame = cv2.resize(frame, (0, 0), fx=0.7, fy=0.7)

        # Display the frame
        cv2.imshow("Labels Comparison", frame)

        # Handle keyboard input
        key = cv2.waitKey(0)
        if key == 27:  # ESC key to exit
            break
        elif key == 100:  # D arrow key
            frame_index = (frame_index + 1) % len(frames)
            direction = 0
        elif key == 97:  # A arrow key
            frame_index = (frame_index - 1) % len(frames)
            direction = 1
    
    cv2.destroyAllWindows()
    

    


def evaluate_tracker(gt_path, pred_path, ignore_id=None, do_print=True, maxiou=0.7, frames=None):
    """
    Evaluate a tracker using ground truth and predictions.

    Args:
        gt_path: Path to ground truth file in MOTChallenge format.
        pred_path: Path to tracker output file in MOTChallenge format.

    Returns:
        Evaluation metrics, including MOTA and MOTP.
    """
    # Load ground truth and predictions
    gt = pd.read_csv(gt_path, header=None, names=["frame", "id", "x", "y", "w", "h", "conf", "class", "vis"])
    pred = pd.read_csv(pred_path, header=None, names=["frame", "id", "x", "y", "w", "h", "conf", "class", "vis"])

    # ignore frames under frames[0] and beyond frames[1]
    if frames is not None:
        gt = gt[(gt["frame"] >= frames[0]) & (gt["frame"] <= frames[1])]
        pred = pred[(pred["frame"] >= frames[0]) & (pred["frame"] <= frames[1])]
    

    # Ignore specified CLASS IDs
    if ignore_id is not None:
        gt = gt[~gt["class"].isin(ignore_id)]
        pred = pred[~pred["class"].isin(ignore_id)]
    
    # check which one has more frames and crop to the smaller one
    if len(gt["frame"].unique()) > len(pred["frame"].unique()):
        gt = gt[gt["frame"].isin(pred["frame"].unique())]
    else:
        pred = pred[pred["frame"].isin(gt["frame"].unique())]
    
    # print(gt["frame"].unique())

    # Create an accumulator for evaluation
    acc = mm.MOTAccumulator(auto_id=True)

    # Iterate over unique frames
    for frame in sorted(gt["frame"].unique()):
        try:
            gt_frame = gt[gt["frame"] == frame]
            pred_frame = pred[pred["frame"] == frame]

            # Compute distances (IoU) between ground truth and predicted boxes
            distances = mm.distances.iou_matrix(
                gt_frame[["x", "y", "w", "h"]].values,
                pred_frame[["x", "y", "w", "h"]].values,
                max_iou=maxiou  # IoU threshold for matches
            )

            # Update accumulator
            acc.update(
                gt_frame["id"].values,  # Ground truth IDs
                pred_frame["id"].values,  # Predicted IDs
                distances
            )
        except Exception as e:
            print(f"Error processing frame {frame}: {e}")

    # Compute metrics
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=None, name='overall')
    # summary = mh.compute(acc)
    if do_print: print(summary)
    return summary

if __name__ == "__main__":    
    # Example usage
    evaluate_tracker("mota_labels.txt", "pred_labels.txt")
