import numpy as np
import open3d as o3d


def load_pcd(file_path):
    """
    Load .pcd file using Open3D
    """
    pcd = o3d.io.read_point_cloud(file_path)

    if len(pcd.points) == 0:
        raise ValueError("Empty or invalid PCD file")

    xyz = np.asarray(pcd.points)

    # Try to extract intensity if available
    intensity = None

    # Some PCDs store intensity in colors (rare but happens)
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        intensity = colors[:, 0]  # assume grayscale stored in R

    return xyz, intensity, pcd


def apply_intensity_coloring(pcd, intensity):
    """
    Apply grayscale intensity coloring if intensity exists
    """
    if intensity is None:
        print("No intensity field found. Using default color.")
        return pcd

    intensity_norm = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-6)
    colors = np.stack([intensity_norm]*3, axis=1)

    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def visualize(pcd):
    """
    Visualize with dark background + axes
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="LiDAR Viewer", width=1280, height=800)

    vis.add_geometry(pcd)

    # Coordinate frame
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])
    vis.add_geometry(axis)

    opt = vis.get_render_option()
    opt.background_color = np.array([0, 0, 0])
    opt.point_size = 1.5

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    file_path = "/workspace/lidar-modules/src/lidar-mapping-sematic/lidar-odom/map/map.pcd"

    xyz, intensity, pcd = load_pcd(file_path)

    print(f"Loaded {len(xyz)} points")

    pcd = apply_intensity_coloring(pcd, intensity)

    visualize(pcd)