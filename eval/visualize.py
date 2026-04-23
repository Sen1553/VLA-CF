"""
visualize.py

Visualizes 3D robot trajectories overlaid on rollout videos.

This script takes trajectory JSON files and corresponding rollout videos,
projects the 3D trajectory points onto the 2D video frames, and creates
a new video with the trajectory overlaid.

Usage:
    python experiments/robot/libero/visualize.py \
        --trajectory_json <PATH_TO_TRAJECTORY_JSON> \
        --video_path <PATH_TO_ROLLOUT_VIDEO> \
        --output_path <OUTPUT_VIDEO_PATH> \
        --camera_params <PATH_TO_CAMERA_PARAMS_JSON> \
        [--show_progress] \
        [--trajectory_color <COLOR>] \
        [--point_size <SIZE>] \
        [--line_width <WIDTH>]

Features:
    - Projects 3D trajectory points onto 2D video frames
    - Color-codes trajectory progression (red to green gradient)
    - Adds trajectory path lines connecting consecutive points
    - Supports custom camera parameters
    - Creates smooth trajectory visualization with configurable styling
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import cv2
import imageio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import tqdm


def load_trajectory_data(json_path: str) -> Tuple[np.ndarray, Dict[str, Any], Optional[np.ndarray]]:
    """
    Load trajectory data from JSON file.
    
    Args:
        json_path: Path to trajectory JSON file
        
    Returns:
        Tuple of (trajectory_points: Nx3 array, metadata: dict, gripper_states: Optional[N])
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # {
    #   "task_description": str,
    #   "episode_index": int,
    #   "result": "success" | "failure",
    #   "conditions": [str],
    #   "per_condition": [{ "condition": str, "touched": bool, "success": bool }, ...],
    #   "steps": [
    #       {
    #         "timestep": int,
    #         "action": [float, ...],
    #         "eef_pos": [x, y, z],
    #         "eef_quat": [w, x, y, z],
    #         "gripper_qpos": [float, ...],
    #         "touched_any": bool
    #       },
    #       ...
    #   ]
    # }
    steps = data.get('steps', [])
    points_list: List[List[float]] = []
    gripper_list: List[float] = []

    for step in steps:
        eef_pos = step.get('eef_pos', [])
        points_list.append([float(eef_pos[0]), float(eef_pos[1]), float(eef_pos[2])])
        gq = step.get('gripper_qpos', [])
        gripper_list.append(np.array(gq, dtype=float))
    
    assert len(points_list) == len(gripper_list), f"points_list: {len(points_list)}, gripper_list: {len(gripper_list)}"
    
    trajectory_points = np.asarray(points_list, dtype=float)
    gripper_states = np.asarray(gripper_list, dtype=float)
    
    # Assemble metadata from top-level keys
    metadata: Dict[str, Any] = {
        "task_description": data.get("task_description"),
        "episode_index": data.get("episode_index"),
        "result": data.get("result"),
        "conditions": data.get("conditions", []),
        "per_condition": data.get("per_condition", []),
    }
    return trajectory_points, metadata, gripper_states


def load_camera_params(params_path: str) -> Dict[str, Any]:
    """
    Load camera parameters from JSON file.
    
    Args:
        params_path: Path to camera parameters JSON file
        
    Returns:
        Dictionary containing camera parameters
    """
    with open(params_path, 'r') as f:
        return json.load(f)


def get_default_camera_params() -> Dict[str, Any]:
    """
    Get default camera parameters for LIBERO environment.
    Uses the exact camera parameters from LIBERO's agentview camera setup.
    """
    # LIBERO agentview camera parameters (from _setup_camera method)
    # camera_pos = np.array([0.5386131746834771, 0.0, 1.4903500240372423])
    camera_pos = np.array([0.659, 0.0, 1.610])
    # camera_pos = np.array([0.65, 0.0, 1.55])
    camera_quat = np.array([0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349])
    # camera_quat = np.array([0.56, 0.43, 0.43, 0.56])
    
    # Convert quaternion to rotation matrix
    # Quaternion format: [w, x, y, z]
    w, x, y, z = camera_quat
    
    # Convert quaternion to rotation matrix (camera-to-world)
    R_c2w = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])
    
    # World-to-camera rotation is transpose of camera-to-world
    R = R_c2w.T
    
    # Create translation vector (world-to-camera)
    t = -R @ camera_pos
    
    # Create extrinsic matrix (world to camera transformation)
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = t
    
    return {
        "intrinsic_matrix": [
            [309.019, 0.0, 127.5],      # fx, 0, cx (match 256x256 frames)
            [0.0, 309.019, 127.5],      # 0, fy, cy  
            [0.0, 0.0, 1.0]           # 0, 0, 1
        ],
        "extrinsic_matrix": extrinsic.tolist(),
        "image_size": [256, 256],     # [width, height]
        "camera_position": camera_pos.tolist(),
        "camera_quat": camera_quat.tolist(),
        "camera_target": [0.0, 0.0, 0.0],  # Looking at workspace center
        "camera_up": [0.0, 0.0, 1.0]       # Z-axis up
    }


def project_3d_to_2d(points_3d: np.ndarray, camera_params: Dict[str, Any]) -> np.ndarray:
    """
    Project 3D points to 2D image coordinates.
    
    Args:
        points_3d: Array of shape (N, 3) containing 3D points
        camera_params: Dictionary containing camera parameters
        
    Returns:
        Array of shape (N, 2) containing 2D image coordinates
    """
    # print(f"project_3d_to_2d called with {len(points_3d)} points")
    
    # Convert to homogeneous coordinates
    points_3d_homo = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
    
    # Get camera matrices
    intrinsic = np.array(camera_params['intrinsic_matrix'])
    extrinsic = np.array(camera_params['extrinsic_matrix'])
    
    # print(f"Intrinsic matrix shape: {intrinsic.shape}")
    # print(f"Extrinsic matrix shape: {extrinsic.shape}")
    
    # Project to camera coordinates
    camera_coords = extrinsic @ points_3d_homo.T
    
    # Convert to 3D camera coordinates (remove homogeneous coordinate)
    camera_coords_3d = camera_coords[:3, :]
    
    # Debug: Check camera coordinates
    print(f"Camera coords bounds: X=[{camera_coords_3d[0, :].min():.3f}, {camera_coords_3d[0, :].max():.3f}], "
          f"Y=[{camera_coords_3d[1, :].min():.3f}, {camera_coords_3d[1, :].max():.3f}], "
          f"Z=[{camera_coords_3d[2, :].min():.3f}, {camera_coords_3d[2, :].max():.3f}]")
    
    # Project to image coordinates
    image_coords_homo = intrinsic @ camera_coords_3d
    
    # Convert from homogeneous coordinates
    image_coords = image_coords_homo[:2, :] / image_coords_homo[2, :]
    
    print(f"Final 2D coords shape: {image_coords.T.shape}")
    return image_coords.T


def save_trajectory_2d_json(
    trajectory_2d: np.ndarray,
    output_json_path: str,
    camera_params: Dict[str, Any],
    metadata: Dict[str, Any],
) -> None:
    """
    Save 2D trajectory points and context to a JSON file.
    """
    data = {
        "trajectory_points_2d": trajectory_2d.tolist(),
        "num_points": int(trajectory_2d.shape[0]),
        "image_size": camera_params.get("image_size", None),
        "intrinsic_matrix": camera_params.get("intrinsic_matrix", None),
        "extrinsic_matrix": camera_params.get("extrinsic_matrix", None),
        "metadata": metadata or {},
    }
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Saved 2D trajectory JSON to: {output_json_path}")


def create_trajectory_colors(num_points: int, color_scheme: str = 'red_green') -> List[Tuple[int, int, int]]:
    """
    Create color gradient for trajectory points.
    
    Args:
        num_points: Number of trajectory points
        color_scheme: Color scheme ('red_green', 'blue_red', 'rainbow')
        
    Returns:
        List of RGB color tuples
    """
    if color_scheme == 'red_green':
        # Red to green gradient
        colors = []
        for i in range(num_points):
            ratio = i / max(1, num_points - 1)
            red = int(255 * (1 - ratio))
            green = int(255 * ratio)
            blue = 0
            colors.append((blue, green, red))  # BGR for OpenCV
    elif color_scheme == 'blue_red':
        # Blue to red gradient
        colors = []
        for i in range(num_points):
            ratio = i / max(1, num_points - 1)
            blue = int(255 * (1 - ratio))
            red = int(255 * ratio)
            green = 0
            colors.append((blue, green, red))  # BGR for OpenCV
    elif color_scheme == 'rainbow':
        # Rainbow gradient
        colors = []
        for i in range(num_points):
            ratio = i / max(1, num_points - 1)
            # Use matplotlib colormap
            cmap = plt.cm.rainbow
            color = cmap(ratio)
            # Convert to BGR for OpenCV
            blue = int(color[2] * 255)
            green = int(color[1] * 255)
            red = int(color[0] * 255)
            colors.append((blue, green, red))
    else:
        # Default to red
        colors = [(0, 0, 255)] * num_points
    
    return colors


def overlay_trajectory_on_frame(
    frame: np.ndarray,
    trajectory_2d: np.ndarray,
    colors: List[Tuple[int, int, int]],
    point_size: int = 3,
    line_width: int = 2,
    show_points: bool = True,
    show_lines: bool = True
) -> np.ndarray:
    """
    Overlay trajectory on a single frame.
    
    Args:
        frame: Input video frame
        trajectory_2d: 2D trajectory points
        colors: List of colors for each point
        point_size: Size of trajectory points
        line_width: Width of trajectory lines
        show_points: Whether to show trajectory points
        show_lines: Whether to show trajectory lines
        
    Returns:
        Frame with trajectory overlaid
    """
    frame_with_trajectory = frame.copy()
    
    # Filter out points that are outside the image bounds
    height, width = frame.shape[:2]
    valid_points = []
    valid_colors = []
    
    for i, (point, color) in enumerate(zip(trajectory_2d, colors)):
        x, y = point
        if 0 <= x < width and 0 <= y < height:
            valid_points.append(point)
            valid_colors.append(color)
    
    if len(valid_points) == 0:
        return frame_with_trajectory
    
    valid_points = np.array(valid_points)
    
    # Draw trajectory lines
    if show_lines and len(valid_points) > 1:
        for i in range(len(valid_points) - 1):
            pt1 = tuple(map(int, valid_points[i]))
            pt2 = tuple(map(int, valid_points[i + 1]))
            color = valid_colors[i]
            cv2.line(frame_with_trajectory, pt1, pt2, color, line_width)
    
    # Draw trajectory points
    if show_points:
        for point, color in zip(valid_points, valid_colors):
            x, y = map(int, point)
            cv2.circle(frame_with_trajectory, (x, y), point_size, color, -1)
    
    return frame_with_trajectory


def create_trajectory_video(
    video_path: str,
    trajectory_json_path: str,
    output_path: str,
    camera_params_path: Optional[str] = None,
    show_progress: bool = True,
    trajectory_color: str = 'red_green',
    point_size: int = 3,
    line_width: int = 2,
    show_points: bool = True,
    show_lines: bool = True
) -> None:
    """
    Create video with trajectory overlaid.
    
    Args:
        video_path: Path to input video
        trajectory_json_path: Path to trajectory JSON file
        output_path: Path to output video
        camera_params_path: Path to camera parameters JSON (optional)
        show_progress: Whether to show progress bar
        trajectory_color: Color scheme for trajectory
        point_size: Size of trajectory points
        line_width: Width of trajectory lines
        show_points: Whether to show trajectory points
        show_lines: Whether to show trajectory lines
    """
    # Load trajectory data
    trajectory_points, metadata, gripper_states = load_trajectory_data(trajectory_json_path)
    print(f"Loaded trajectory with {len(trajectory_points)} points")
    print(f"Task: {metadata.get('task_description', 'Unknown')}")
    
    # Load camera parameters
    if camera_params_path and os.path.exists(camera_params_path):
        camera_params = load_camera_params(camera_params_path)
        print(f"Loaded camera parameters from {camera_params_path}")
    else:
        camera_params = get_default_camera_params()
        print("Using default camera parameters")
    
    # Project 3D trajectory to 2D
    print(f"Trajectory points shape: {trajectory_points.shape}")
    # print(f"camera_params keys: {list(camera_params.keys())}")
    trajectory_2d = project_3d_to_2d(trajectory_points, camera_params)
     #TODO: current projected 2D position is not well aligned, hard-coded for now, need to fix this later
    trajectory_2d -= np.array([12, 10])
    # print("project_3d_to_2d completed")
    
    # Debug: Print trajectory bounds
    print(f"3D trajectory bounds: X=[{trajectory_points[:, 0].min():.3f}, {trajectory_points[:, 0].max():.3f}], "
          f"Y=[{trajectory_points[:, 1].min():.3f}, {trajectory_points[:, 1].max():.3f}], "
          f"Z=[{trajectory_points[:, 2].min():.3f}, {trajectory_points[:, 2].max():.3f}]")
    print(f"2D trajectory bounds: X=[{trajectory_2d[:, 0].min():.1f}, {trajectory_2d[:, 0].max():.1f}], "
          f"Y=[{trajectory_2d[:, 1].min():.1f}, {trajectory_2d[:, 1].max():.1f}]")
    # print(f"Image size: {camera_params['image_size']}")
    
    # Create trajectory colors
    colors = create_trajectory_colors(len(trajectory_points), trajectory_color)
    
    # Load video
    video_reader = imageio.get_reader(video_path)
    fps = video_reader.get_meta_data()['fps']
    
    # Create output video writer
    video_writer = imageio.get_writer(output_path, fps=fps)
    
    # Save 2D trajectory JSON next to the output video
    try:
        base, _ = os.path.splitext(output_path)
        trajectory_2d_json = base + "_2d.json"
        save_trajectory_2d_json(trajectory_2d, trajectory_2d_json, camera_params, metadata)
    except Exception as e:
        print(f"Failed to save 2D trajectory JSON: {e}")

    print(f"gripper_states shape: {gripper_states.shape}")

    try:
        gripper_scalar = gripper_states[:, 0]
        close_indices = np.where(gripper_scalar < 0.038)[0]
        if close_indices.size > 0:
            first_close_idx = int(close_indices[0] + 1)  # index after the diff
            first_close_point_2d = trajectory_2d[first_close_idx].tolist()
            close_json = base + "_gripper_close_2d.json"
            payload = {
                "index": first_close_idx,
                "point_2d": first_close_point_2d,
                "gripper_value_before": float(gripper_scalar[first_close_idx - 1]) if first_close_idx > 0 else None,
                "gripper_value_after": float(gripper_scalar[first_close_idx]),
                "image_size": camera_params.get("image_size", None),
                "intrinsic_matrix": camera_params.get("intrinsic_matrix", None),
                "extrinsic_matrix": camera_params.get("extrinsic_matrix", None),
                "metadata": metadata or {},
            }
            with open(close_json, "w") as f:
                json.dump(payload, f, indent=4)
            print(f"Saved first-gripper-close 2D point JSON to: {close_json}")

            # Also overlay this point on the first frame and save as image
            try:
                first_frame = video_reader.get_data(0)
                h, w = first_frame.shape[:2]
                x_i, y_i = map(int, first_close_point_2d)
                if 0 <= x_i < w and 0 <= y_i < h:
                    overlay_bgr = cv2.cvtColor(first_frame.copy(), cv2.COLOR_RGB2BGR)
                    # cv2.circle(overlay_bgr, (x_i, y_i), 10, (0, 0, 255), thickness=2)
                    cv2.circle(overlay_bgr, (x_i, y_i), 4, (0, 0, 255), -1)
                    out_img_path = base + "_gripper_close_overlay.png"
                    cv2.imwrite(out_img_path, overlay_bgr)
                    print(f"Saved gripper-close overlay image to: {out_img_path}")
            except Exception as e_img:
                print(f"Failed to save gripper-close overlay image: {e_img}")
    except Exception as e:
        print(f"Failed to save gripper-close 2D JSON: {e}")

    # Process each frame
    frame_count = 0
    total_frames = len(video_reader)
    
    if show_progress:
        pbar = tqdm.tqdm(total=total_frames, desc="Processing frames")
    
    try:
        for frame in video_reader:
            # Determine which trajectory points to show based on current frame
            # For now, show all trajectory points (can be modified for progressive display)
            current_trajectory = trajectory_2d
            current_colors = colors
            
            # Overlay trajectory on frame
            frame_with_trajectory = overlay_trajectory_on_frame(
                frame, current_trajectory, current_colors,
                point_size, line_width, show_points, show_lines
            )
            
            # Write frame to output video
            video_writer.append_data(frame_with_trajectory)
            
            frame_count += 1
            if show_progress:
                pbar.update(1)
    
    finally:
        video_reader.close()
        video_writer.close()
        if show_progress:
            pbar.close()
    
    print(f"Created trajectory video: {output_path}")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Visualize 3D trajectories on rollout videos")
    
    parser.add_argument("--trajectory_json", type=str, required=True,
                       help="Path to trajectory JSON file")
    parser.add_argument("--video_path", type=str, required=True,
                       help="Path to rollout video")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Path to output video")
    parser.add_argument("--camera_params", type=str, default=None,
                       help="Path to camera parameters JSON file")
    parser.add_argument("--show_progress", action="store_true",
                       help="Show progress bar")
    parser.add_argument("--trajectory_color", type=str, default="red_green",
                       choices=["red_green", "blue_red", "rainbow"],
                       help="Color scheme for trajectory")
    parser.add_argument("--point_size", type=int, default=3,
                       help="Size of trajectory points")
    parser.add_argument("--line_width", type=int, default=2,
                       help="Width of trajectory lines")
    parser.add_argument("--hide_points", action="store_true",
                       help="Hide trajectory points")
    parser.add_argument("--hide_lines", action="store_true",
                       help="Hide trajectory lines")
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.trajectory_json):
        print(f"Error: Trajectory JSON file not found: {args.trajectory_json}")
        return
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        return
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Create trajectory video
    create_trajectory_video(
        video_path=args.video_path,
        trajectory_json_path=args.trajectory_json,
        output_path=args.output_path,
        camera_params_path=args.camera_params,
        show_progress=args.show_progress,
        trajectory_color=args.trajectory_color,
        point_size=args.point_size,
        line_width=args.line_width,
        show_points=not args.hide_points,
        show_lines=not args.hide_lines
    )


if __name__ == "__main__":
    main()
