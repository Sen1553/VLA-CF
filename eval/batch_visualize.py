"""
batch_visualize.py

Batch processes multiple trajectory videos for visualization.

This script automatically finds matching trajectory JSON files and rollout videos,
then creates trajectory-overlaid videos for all of them.

Usage:
    python experiments/robot/libero/batch_visualize.py \
        --rollouts_dir <PATH_TO_ROLLOUTS_DIR> \
        [--pattern <FILE_PATTERN>]

Example:
    python experiments/robot/libero/batch_visualize.py \
        --rollouts_dir ./rollouts/2025_01_23-10_30_45 
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import cv2
import imageio
import tqdm

# Import the visualization function
from visualize import create_trajectory_video


def find_matching_files(video_dir: str, trajectory_dir: str, pattern: str = "*") -> List[Tuple[str, str, str]]:
    """
    Find matching trajectory JSON and video files.
    
    Args:
        video_dir: Directory containing rollout videos
        trajectory_dir: Directory containing trajectory JSON files
        pattern: Pattern to match files (e.g., "*success=True*")
        
    Returns:
        List of tuples (video_path, trajectory_json_path, output_path)
    """
    matches = []
    
    # Find all video files
    video_pattern = os.path.join(video_dir, f"{pattern}.mp4")
    video_files = sorted(glob.glob(video_pattern))
    print(f"Found {len(video_files)} video files")
    
    for video_path in video_files:
        # Extract base filename without extension
        video_basename = os.path.basename(video_path)
        base_name = video_basename.replace('.mp4', '')
        
        # Look for corresponding JSON file
        # json_pattern = os.path.join(trajectories_dir, f"{base_name}.json")
        # json_pattern = glob.glob(os.path.join(trajectories_dir, f"{base_name[:50]}*.json"))[0]
        json_pattern = os.path.join(trajectory_dir, f"{base_name}.json")
        if os.path.exists(json_pattern):
            # Create output path
            output_path = os.path.join(
                os.path.dirname(video_path).replace('videos', 'visualizations'),
                f"{base_name}.mp4"
            )
            matches.append((video_path, json_pattern, output_path))
    
    return matches


def batch_process_trajectories(
    rollouts_dir: str,
    pattern: str = "*",
    trajectory_color: str = 'red_green',
    point_size: int = 3,
    line_width: int = 2,
    show_points: bool = True,
    show_lines: bool = True
) -> None:
    """
    Batch process multiple trajectory videos.
    
    Args:
        rollouts_dir: Directory containing rollout videos
        pattern: Pattern to match files
        trajectory_color: Color scheme for trajectory
        point_size: Size of trajectory points
        line_width: Width of trajectory lines
        show_points: Whether to show trajectory points
        show_lines: Whether to show trajectory lines
    """
    # Find matching files
    video_dir = os.path.join(rollouts_dir, "videos")
    trajectory_dir = os.path.join(rollouts_dir, "trajectories")
    output_dir = os.path.join(rollouts_dir, "visualizations")
    
    matches = find_matching_files(video_dir, trajectory_dir, pattern)
    
    if not matches:
        print("No matching files found!")
        return
    
    print(f"Found {len(matches)} matching files to process")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all gripper-close 2D points across videos
    all_close_points: List[Tuple[int, int]] = []
    point_sources: List[str] = []
    first_frame = None
    first_frame_size = None


    pbar = tqdm.tqdm(matches, desc="Processing videos")
    
    success_count = 0
    error_count = 0
    cnt = 0
    
    for video_path, json_path, output_path in pbar:
        cnt += 1
        try:
            print(f"=" * 50)
            print(f"Processing {video_path}")
            # Update output path to use the specified output directory
            output_filename = os.path.basename(output_path)
            final_output_path = os.path.join(output_dir, output_filename)
            
            # Create trajectory video
            create_trajectory_video(
                video_path=video_path,
                trajectory_json_path=json_path,
                output_path=final_output_path,
                camera_params_path=None,  # Use default camera parameters
                show_progress=False,  # Don't show individual progress
                trajectory_color=trajectory_color,
                point_size=point_size,
                line_width=line_width,
                show_points=show_points,
                show_lines=show_lines
            )
            
            # Capture first frame from the first video for global overlays
            if first_frame is None:
                try:
                    vr = imageio.get_reader(video_path)
                    first_frame = vr.get_data(0)
                    vr.close()
                    h, w = first_frame.shape[:2]
                    first_frame_size = (w, h)
                except Exception:
                    first_frame = None
                    first_frame_size = None

            # Read the per-video gripper-close 2D JSON produced by visualize.py
            try:
                base, _ = os.path.splitext(final_output_path)
                close_json_path = base + "_gripper_close_2d.json"
                if os.path.exists(close_json_path):
                    with open(close_json_path, "r") as f:
                        payload = json.load(f)
                    pt = payload.get("point_2d", None)
                    if pt is not None and isinstance(pt, list) and len(pt) == 2:
                        x_i, y_i = int(round(pt[0])), int(round(pt[1]))
                        all_close_points.append((x_i, y_i))
                        point_sources.append(os.path.basename(video_path))
            except Exception:
                pass

            success_count += 1
            pbar.set_postfix({"Success": success_count, "Errors": error_count})
                
        except Exception as e:
            error_count += 1
            print(f"Error processing {video_path}: {e}")
            pbar.set_postfix({"Success": success_count, "Errors": error_count})
    
        if cnt % 50 == 0:
            try:
                agg_json_path = os.path.join(rollouts_dir, f"task{cnt // 50:02d}_all_gripper_close_points.json")
                agg_payload = {
                    "image_size": list(first_frame_size) if first_frame_size else None,
                    "num_points": len(all_close_points),
                    "points_2d": [[int(x), int(y)] for (x, y) in all_close_points],
                    "sources": point_sources,
                }
                with open(agg_json_path, "w") as f:
                    json.dump(agg_payload, f, indent=4)
                print(f"Saved aggregated close points JSON to: {agg_json_path}")
            except Exception as e:
                print(f"Failed to save aggregated close points JSON: {e}")

            # Create overlays (points and heatmap) on the first frame
            try:
                if first_frame is not None and len(all_close_points) > 0:
                    h, w = first_frame.shape[:2]
                    # Points overlay
                    points_bgr = cv2.cvtColor(first_frame.copy(), cv2.COLOR_RGB2BGR)
                    for (x, y) in all_close_points:
                        if 0 <= x < w and 0 <= y < h:
                            cv2.circle(points_bgr, (x, y), 3, (0, 0, 255), -1)
                    out_points_path = os.path.join(rollouts_dir, f"task{cnt // 50:02d}_all_gripper_close_points_overlay.png")
                    cv2.imwrite(out_points_path, points_bgr)
                    print(f"Saved points overlay image to: {out_points_path}")

                    # Heatmap overlay with percentile + gamma normalization for better color separation
                    heat = np.zeros((h, w), dtype=np.float32)
                    # Radius proportional to image size; ensures visible footprint
                    rad = max(6, int(0.04 * min(h, w)))
                    for (x, y) in all_close_points:
                        if 0 <= x < w and 0 <= y < h:
                            # Add a filled circle kernel around each point
                            cv2.circle(heat, (x, y), rad, 1.0, thickness=-1)
                    # Smooth to create a continuous heat distribution
                    sigma = max(3.0, 0.5 * rad)
                    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=sigma, sigmaY=sigma)

                    # TODO: improve heatmap visualization to distinguish densities
                    # Percentile-based normalization to avoid one outlier dominating
                    nonzero_vals = heat[heat > 0]
                    if nonzero_vals.size > 0:
                        p90 = float(np.percentile(nonzero_vals, 90.0))
                        denom = max(p90, 1e-6)
                        norm = np.clip(heat / denom, 0.0, 1.0)
                    else:
                        norm = heat
                    # Gamma correction (<1.0 makes more values tend toward red)
                    gamma = 0.6
                    norm_gamma = np.power(norm, gamma)
                    heat_norm = np.uint8(255.0 * norm_gamma)
                    heat_color = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)
                    # Blend with the frame
                    frame_bgr = cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR)
                    overlay = cv2.addWeighted(frame_bgr, 0.6, heat_color, 0.4, 0)
                    out_heatmap_path = os.path.join(rollouts_dir, f"task{cnt // 50:02d}_all_gripper_close_points_heatmap.png")
                    cv2.imwrite(out_heatmap_path, overlay)
                    print(f"Saved heatmap overlay image to: {out_heatmap_path}")
            except Exception as e:
                print(f"Failed to create aggregated overlays: {e}")

            print("=" * 50)
            print(f"Batch processing complete!")
            print(f"Successfully processed: {success_count} videos")
            print(f"Errors: {error_count} videos")
            print(f"Output directory: {output_dir}")

            all_close_points = []
            point_sources = []
            first_frame = None
            first_frame_size = None


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Batch process trajectory visualizations")
    
    parser.add_argument("--rollouts_dir", type=str, required=True,
                       help="Directory containing rollout videos")
    parser.add_argument("--pattern", type=str, default="*",
                       help="Pattern to match files (e.g., '*success=True*')")
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
    
    # Validate input directories
    if not os.path.exists(args.rollouts_dir):
        print(f"Error: Rollouts directory not found: {args.rollouts_dir}")
        return
    
    batch_process_trajectories(
        rollouts_dir=args.rollouts_dir,
        pattern=args.pattern,
        trajectory_color=args.trajectory_color,
        point_size=args.point_size,
        line_width=args.line_width,
        show_points=not args.hide_points,
        show_lines=not args.hide_lines
    )


if __name__ == "__main__":
    main()
