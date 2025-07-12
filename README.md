# YOLO-based 6D Pose Estimation with solvePnP

This project implements real-time 6D pose estimation using YOLOv8 segmentation and OpenCV's solvePnP algorithm. The system detects cell phones in live video streams from Intel RealSense cameras and estimates their 3D position and orientation relative to the camera.

## Overview

The system combines state-of-the-art object detection (YOLOv8) with classical computer vision techniques (solvePnP) to achieve accurate 6D pose estimation:

1. **YOLOv8 Segmentation**: Detects cell phones (class_id 67) and generates precise masks
2. **3D-2D Correspondence**: Maps 3D model corners to 2D image points
3. **solvePnP Algorithm**: Solves the Perspective-n-Point problem to estimate pose
4. **Real-time Processing**: Live visualization with pose information

## Technical Approach

### solvePnP Algorithm

The core of this system uses OpenCV's `cv2.solvePnP()` function, which implements the Perspective-n-Point algorithm. This is a classical computer vision technique that:

- Takes 3D object points and their corresponding 2D image projections
- Estimates the camera pose (rotation and translation) that best explains the observations
- Uses iterative optimization to minimize reprojection error

### Implementation Details

```python
# 1. Extract 3D bounding box corners from PLY model
obj_points = get_3d_bbox_corners(ply_path)  # (8, 3)

# 2. Extract 2D corners from segmentation mask
img_points = get_2d_bbox_corners(mask)  # (4, 2)

# 3. Solve PnP to get rotation and translation vectors
success, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, dist_coeffs)

# 4. Convert rotation vector to rotation matrix and RPY angles
R, _ = cv2.Rodrigues(rvec)
```

### 3D Model Usage

The system uses the 3D PLY model in two complementary ways:

1. **3D Bounding Box Corners**: The `get_3d_bbox_corners()` function extracts the 8 corners of the 3D model's axis-aligned bounding box using Open3D:
   ```python
   mesh = o3d.io.read_triangle_mesh(ply_path)
   bbox = mesh.get_axis_aligned_bounding_box()
   corners = np.asarray(bbox.get_box_points())  # (8, 3)
   ```

2. **solvePnP Object Points**: These 3D corners serve as the object points for the solvePnP algorithm, providing geometric constraints for pose estimation.

3. **Object Dimensions**: Additionally, the system uses predefined object dimensions (height, width) for depth estimation when the 3D model alone is insufficient.

### Key Features

- **YOLOv8 Segmentation**: Precise object detection and mask generation
- **solvePnP Pose Estimation**: Robust 6D pose calculation using 3D-2D correspondences
- **RealSense Integration**: Live camera feed processing
- **Cell Phone Detection**: Specialized for cell phone class (ID 67)
- **Real-time Visualization**: Live pose display with coordinate axes
- **Data Export**: Save screenshots and pose data with timestamps

## Project Structure

```
yolo_pose_estimation_project/
├── live_infer_3Dply_pose_yolo.py    # Main pose estimation script
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
├── data/
│   ├── my_test_images/              # Test images and camera data
│   │   ├── *.jpg                    # Test images
│   │   └── scene_camera.json        # Camera intrinsics
│   └── my_models/                   # Models and 3D data
│       ├── yolov8s-seg.pt           # YOLOv8 segmentation model
│       └── MZVB5IQUB.ply            # 3D CAD model
└── results_live/                    # Output directory with example results
```

## Requirements

- Python 3.7 or higher
- Intel RealSense camera (D415, D435, etc.)
- CUDA-compatible GPU (optional, runs on CPU by default)
- YOLOv8 segmentation model
- 3D CAD model in PLY format

## Installation

1. Clone or download this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure your Intel RealSense camera is connected
4. Download YOLOv8 segmentation model if not included

## Usage

1. Connect your Intel RealSense camera
2. Update configuration in `live_infer_3Dply_pose_yolo.py`:
   - Set correct paths for YOLO model and PLY file
   - Adjust object dimensions if needed
   - Modify confidence and IoU thresholds

3. Run the pose estimation:
   ```bash
   python live_infer_3Dply_pose_yolo.py
   ```

4. Controls:
   - Press 's' to save screenshot and pose data
   - Press 'q' to quit

## Configuration

Key parameters in the script:

```python
YOLO_MODEL_PATH = 'data/my_models/yolov8s-seg.pt'
PLY_MODEL_PATH = 'data/my_models/MZVB5IQUB.ply'
CAMERA_INTRINSICS_PATH = 'data/my_test_images/scene_camera.json'
OBJECT_DIMENSIONS = (0.20, 0.085)  # (height, width) in meters
YOLO_CONFIDENCE = 0.5
YOLO_IOU = 0.45
CELL_PHONE_CLASS_ID = 67
```

## Input Data Format

### Camera Intrinsics (scene_camera.json)
```json
{
  "0": {
    "cam_K": [fx, 0, cx, 0, fy, cy, 0, 0, 1],
    "cam_R_w2c": [1, 0, 0, 0, 1, 0, 0, 0, 1],
    "cam_t_w2c": [0, 0, 0]
  }
}
```

### 3D Model Requirements
- PLY format with vertices and faces
- Units in meters
- Origin at object center
- Proper bounding box for corner extraction
- The model is used to extract 3D bounding box corners for solvePnP algorithm
- Object dimensions are also used as fallback for depth estimation

## Output

The system provides:

1. **Real-time Visualization**:
   - Object detection with bounding boxes
   - Segmentation masks overlaid
   - Coordinate axes showing orientation
   - Pose information displayed on screen

2. **Terminal Output**:
   - 6D pose coordinates (X, Y, Z, Roll, Pitch, Yaw)
   - Detection confidence scores

3. **Saved Data** (when 's' is pressed):
   - Screenshot: `realsense_yolo_YYYYMMDD_HHMMSS.png`
   - Pose data: `realsense_yolo_YYYYMMDD_HHMMSS.json`

## solvePnP Algorithm Benefits

- **Robust**: Handles partial occlusions and noise
- **Accurate**: Minimizes reprojection error
- **Fast**: Efficient iterative optimization
- **Well-tested**: Classical computer vision approach
- **No Training**: Works with any 3D model

## Limitations

- Requires accurate 3D model of the object
- Depends on good 2D-3D point correspondences
- Sensitive to camera calibration accuracy
- Assumes rigid objects (no deformation)

## Troubleshooting

- **No detections**: Check YOLO model path and confidence threshold
- **Poor pose accuracy**: Verify camera intrinsics and object dimensions
- **Camera not found**: Ensure RealSense camera is properly connected
- **Performance issues**: Consider using GPU acceleration

## Dependencies

- **ultralytics**: YOLOv8 model loading and inference
- **opencv-python**: Computer vision operations and solvePnP
- **pyrealsense2**: Intel RealSense camera interface
- **open3d**: 3D model processing
- **numpy**: Numerical computations
- **torch/torchvision**: Deep learning framework (for YOLO)

## References

- [OpenCV solvePnP Documentation](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga549c2075fac14829ff4a58bc931c033d)
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Intel RealSense SDK](https://github.com/IntelRealSense/librealsense) 