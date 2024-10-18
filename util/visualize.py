import open3d as o3d
import numpy as np

def load_kitti_from_file(bin_file):
    point_cloud = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)
    points = point_cloud[:, :3]
    return points

def to_open3d_point_cloud(np):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np)
    return pcd

def open3d_point_cloud(pcd, results=None, window_name="Open3D Point Cloud"):
    """
    Visualizes a NumPy point cloud with optional detection results (e.g., bounding boxes).
    
    Parameters:
    - np_pointcloud: NumPy array containing the point cloud data (shape Nx3 or Nx4)
    - results: Optional detection results such as bounding boxes (list of geometries)
    - window_name: The name of the Open3D visualization window
    
    Returns:
    - pcd: The Open3D PointCloud object created from the NumPy point cloud
    """

    geometries = [pcd]
    
    if results:
        geometries.extend(results)

    o3d.visualization.draw_geometries(geometries, window_name=window_name)