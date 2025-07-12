#!/usr/bin/env python3
"""
Live 6D Pose Estimation with Intel RealSense + YOLOv8
=====================================================

Captures frames from an Intel RealSense camera, detects cell phones (class_id 67)
using YOLOv8 segmentation, and estimates 6D pose for each detection.

- Only cell phone detections are processed.
- Centroid is calculated as the center of the mask.
- Results are visualized live.
- Press 's' to save a screenshot and pose data.
- Press 'q' to quit.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import cv2
import numpy as np
import json
import time
import pyrealsense2 as rs
from ultralytics import YOLO
import open3d as o3d

# --- CONFIG ---
YOLO_MODEL_PATH = 'data/my_models/yolov8s-seg.pt'
PLY_MODEL_PATH = 'data/my_models/MZVB5IQUB.ply'
CAMERA_INTRINSICS_PATH = 'data/my_test_images/scene_camera.json'
OBJECT_DIMENSIONS = (0.20, 0.085)  # (height, width) in meters
OUTPUT_DIR = 'results_live'
YOLO_CONFIDENCE = 0.5
YOLO_IOU = 0.45
CELL_PHONE_CLASS_ID = 67

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load Camera Intrinsics ---
def load_camera_intrinsics(json_path):
    with open(json_path, 'r') as f:
        camera_data = json.load(f)
    camera_id = list(camera_data.keys())[0]
    cam_K = camera_data[camera_id]['cam_K']
    return np.array(cam_K).reshape(3, 3)

K = load_camera_intrinsics(CAMERA_INTRINSICS_PATH)
fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]

# --- Load YOLOv8 Model ---
yolo_model = YOLO(YOLO_MODEL_PATH)

# --- RealSense Setup ---
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
pipeline.start(config)

print("Press 's' to save screenshot and pose, 'q' to quit.")

def get_3d_bbox_corners(ply_path):
    mesh = o3d.io.read_triangle_mesh(ply_path)
    bbox = mesh.get_axis_aligned_bounding_box()
    corners = np.asarray(bbox.get_box_points())
    return corners  # shape (8, 3)

def get_2d_bbox_corners(mask):
    # Find the largest contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    # Approximate to polygon (try to get 4 corners for a rectangle)
    epsilon = 0.05 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    if len(approx) < 4:
        # fallback: use bounding box
        x, y, w, h = cv2.boundingRect(cnt)
        return np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.float32)
    else:
        return approx[:, 0, :].astype(np.float32)  # shape (N, 2)

def estimate_pose_solvepnp(mask, ply_path, K):
    # 1. Get 3D corners
    obj_points = get_3d_bbox_corners(ply_path)  # (8, 3)
    # 2. Get 2D corners
    mask_uint8 = mask.astype(np.uint8)
    img_points = get_2d_bbox_corners(mask_uint8)  # (4, 2) or (N, 2)
    if img_points is None or len(img_points) < 4:
        return None  # Not enough points
    # For a box, use only 4 corners (choose the best matching order)
    if len(img_points) > 4:
        img_points = img_points[:4]
    # Select 4 corresponding 3D points (e.g., bottom face of the box)
    obj_points = obj_points[:4]
    # 3. Solve PnP
    dist_coeffs = np.zeros((4, 1))  # Assuming no distortion
    success, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, dist_coeffs)
    if not success:
        return None
    # 4. Convert rotation vector to RPY
    R, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        roll = np.arctan2(R[2,1], R[2,2])
        pitch = np.arctan2(-R[2,0], sy)
        yaw = np.arctan2(R[1,0], R[0,0])
    else:
        roll = np.arctan2(-R[1,2], R[1,1])
        pitch = np.arctan2(-R[2,0], sy)
        yaw = 0
    return tvec.flatten(), (roll, pitch, yaw)

while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        continue
    image = np.asanyarray(color_frame.get_data())
    orig_image = image.copy()

    # --- YOLOv8 Segmentation ---
    results = yolo_model(image, conf=YOLO_CONFIDENCE, iou=YOLO_IOU, device='cpu')
    detections = []
    for result in results:
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            boxes = result.boxes.data.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()
            for i in range(len(masks)):
                class_id = int(class_ids[i])
                if class_id != CELL_PHONE_CLASS_ID:
                    continue
                mask = masks[i]
                box = boxes[i]
                conf = confidences[i]
                mask_resized = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                M = cv2.moments(mask_resized)
                if M["m00"] != 0:
                    center_x = int(M["m10"] / M["m00"])
                    center_y = int(M["m01"] / M["m00"])
                else:
                    x, y, w, h = [int(v) for v in box[:4]]
                    center_x = x + w // 2
                    center_y = y + h // 2
                x, y, x2, y2 = [int(v) for v in box[:4]]
                w, h = x2 - x, y2 - y
                if h > w:
                    depth = (OBJECT_DIMENSIONS[0] * fy) / h
                else:
                    depth = (OBJECT_DIMENSIONS[1] * fx) / w
                X = (center_x - cx) * depth / fx
                Y = (center_y - cy) * depth / fy
                Z = depth
                contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) > 0:
                    contour = contours[0]
                    if len(contour) >= 5:
                        ellipse = cv2.fitEllipse(contour)
                        _, _, angle = ellipse
                        yaw = np.radians(angle)
                    else:
                        yaw = 0.0
                else:
                    yaw = 0.0
                roll = pitch = 0.0
                detection = {
                    'mask': mask_resized.tolist(),
                    'bbox': [x, y, w, h],
                    'centroid_2d': [center_x, center_y],
                    'translation': [X, Y, Z],
                    'rotation_rpy_deg': [np.degrees(roll), np.degrees(pitch), np.degrees(yaw)],
                    'confidence': float(conf)
                }
                detections.append(detection)
                # Visualization
                mask_overlay = np.zeros_like(image)
                mask_overlay[mask_resized.astype(bool)] = [0, 255, 255]
                image = cv2.addWeighted(image, 0.7, mask_overlay, 0.3, 0)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.circle(image, (center_x, center_y), 8, (0, 255, 0), -1)
                cv2.circle(image, (center_x, center_y), 10, (255, 255, 255), 2)
                axis_length = 30
                end_x = int(center_x + axis_length)
                end_y = center_y
                cv2.line(image, (center_x, center_y), (end_x, end_y), (0, 0, 255), 2)
                end_x = center_x
                end_y = int(center_y + axis_length)
                cv2.line(image, (center_x, center_y), (end_x, end_y), (0, 255, 0), 2)
                cv2.circle(image, (center_x, center_y), 4, (255, 0, 0), -1)
                # --- Print 6D pose in terminal ---
                print(f"Cell Phone 6D Pose: X={X:.3f}m, Y={Y:.3f}m, Z={Z:.3f}m, Roll={np.degrees(roll):.1f}°, Pitch={np.degrees(pitch):.1f}°, Yaw={np.degrees(yaw):.1f}°")
                # --- Draw 6D pose on image ---
                pose_text1 = f"X={X:.3f}m Y={Y:.3f}m Z={Z:.3f}m"
                pose_text2 = f"Roll={np.degrees(roll):.1f} Pitch={np.degrees(pitch):.1f} Yaw={np.degrees(yaw):.1f}"
                cv2.putText(image, pose_text1, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(image, pose_text2, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("RealSense YOLOv8 Cell Phone Pose", image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('s'):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        img_path = os.path.join(OUTPUT_DIR, f"realsense_yolo_{timestamp}.png")
        json_path = os.path.join(OUTPUT_DIR, f"realsense_yolo_{timestamp}.json")
        cv2.imwrite(img_path, orig_image)
        with open(json_path, 'w') as f:
            json.dump({'detections': detections}, f, indent=2)
        print(f"Saved screenshot to {img_path} and pose data to {json_path}")
pipeline.stop()
cv2.destroyAllWindows() 