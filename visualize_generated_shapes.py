#!/usr/bin/env python3
"""
Visualization tool for generated 3D point clouds.

This script provides interactive visualization of generated 3D shapes
using Open3D and matplotlib for analysis and inspection.
"""

import argparse
import numpy as np
import os
import sys
from pathlib import Path

def load_generated_shapes(file_path: str) -> np.ndarray:
    """
    Load generated shapes from npz file.
    
    Args:
        file_path: Path to the generated shapes npz file
        
    Returns:
        np.ndarray: Array of point clouds with shape (N, num_points, 3)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Generated shapes file not found: {file_path}")
    
    data = np.load(file_path)
    if 'points' not in data:
        raise ValueError("No 'points' key found in the npz file")
    
    points = data['points']
    print(f"Loaded {points.shape[0]} generated shapes with {points.shape[1]} points each")
    return points

def visualize_with_open3d(points: np.ndarray, num_shapes: int = 5):
    """
    Visualize point clouds using Open3D.
    
    Args:
        points: Array of point clouds
        num_shapes: Number of shapes to visualize
    """
    try:
        import open3d as o3d
    except ImportError:
        print("Open3D not installed. Install with: pip install open3d")
        return
    
    num_shapes = min(num_shapes, len(points))
    
    for i in range(num_shapes):
        print(f"Visualizing shape {i+1}/{num_shapes}")
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[i])
        
        # Color the point cloud
        colors = np.random.rand(len(points[i]), 3)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Visualize
        o3d.visualization.draw_geometries([pcd], 
                                        window_name=f"Generated Shape {i+1}",
                                        width=800, height=600)

def visualize_with_matplotlib(points: np.ndarray, num_shapes: int = 5):
    """
    Visualize point clouds using matplotlib.
    
    Args:
        points: Array of point clouds
        num_shapes: Number of shapes to visualize
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("Matplotlib not installed. Install with: pip install matplotlib")
        return
    
    num_shapes = min(num_shapes, len(points))
    
    # Create subplots
    fig = plt.figure(figsize=(15, 10))
    
    cols = min(3, num_shapes)
    rows = (num_shapes + cols - 1) // cols
    
    for i in range(num_shapes):
        ax = fig.add_subplot(rows, cols, i+1, projection='3d')
        
        # Plot points
        x, y, z = points[i][:, 0], points[i][:, 1], points[i][:, 2]
        ax.scatter(x, y, z, c=z, cmap='viridis', s=1)
        
        ax.set_title(f'Generated Shape {i+1}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Equal aspect ratio
        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
        mid_x = (x.max()+x.min()) * 0.5
        mid_y = (y.max()+y.min()) * 0.5
        mid_z = (z.max()+z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.show()

def print_statistics(points: np.ndarray):
    """
    Print statistics about the generated shapes.
    
    Args:
        points: Array of point clouds
    """
    print(f"\n=== Generated Shapes Statistics ===")
    print(f"Number of shapes: {points.shape[0]}")
    print(f"Points per shape: {points.shape[1]}")
    print(f"Dimensions: {points.shape[2]}")
    
    # Overall statistics
    all_points = points.reshape(-1, points.shape[2])
    print(f"\nOverall point statistics:")
    print(f"  Min coordinates: {all_points.min(axis=0)}")
    print(f"  Max coordinates: {all_points.max(axis=0)}")
    print(f"  Mean coordinates: {all_points.mean(axis=0)}")
    print(f"  Std coordinates: {all_points.std(axis=0)}")
    
    # Per-shape statistics
    print(f"\nPer-shape statistics:")
    shape_mins = points.min(axis=1)
    shape_maxs = points.max(axis=1)
    shape_means = points.mean(axis=1)
    
    print(f"  Min extents: {shape_mins.min(axis=0)} to {shape_mins.max(axis=0)}")
    print(f"  Max extents: {shape_maxs.min(axis=0)} to {shape_maxs.max(axis=0)}")
    print(f"  Mean positions: {shape_means.min(axis=0)} to {shape_means.max(axis=0)}")

def main():
    """Main function to run the visualization tool."""
    parser = argparse.ArgumentParser(description='Visualize generated 3D point clouds')
    parser.add_argument('file_path', type=str, 
                       help='Path to the generated shapes npz file')
    parser.add_argument('--num-shapes', type=int, default=5,
                       help='Number of shapes to visualize (default: 5)')
    parser.add_argument('--backend', type=str, choices=['open3d', 'matplotlib', 'both'], 
                       default='both',
                       help='Visualization backend to use (default: both)')
    parser.add_argument('--stats', action='store_true',
                       help='Print statistics about the generated shapes')
    
    args = parser.parse_args()
    
    try:
        # Load generated shapes
        points = load_generated_shapes(args.file_path)
        
        # Print statistics if requested
        if args.stats:
            print_statistics(points)
        
        # Visualize based on backend choice
        if args.backend in ['open3d', 'both']:
            print(f"\nVisualizing with Open3D...")
            visualize_with_open3d(points, args.num_shapes)
        
        if args.backend in ['matplotlib', 'both']:
            print(f"\nVisualizing with matplotlib...")
            visualize_with_matplotlib(points, args.num_shapes)
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()