import numpy as np
import open3d as o3d
import struct
import sys
import os


# ── Paths ────────────────────────────────────────────────────────────────────
MAP_DIR    = "/workspace/lidar-modules/src/lidar-mapping-sematic/lidar-odom/map"
MAP_PCD    = os.path.join(MAP_DIR, "map.pcd")
POSES_BIN  = os.path.join(MAP_DIR, "std_db/keyframe_poses.bin")
PLANES_BIN = os.path.join(MAP_DIR, "std_db/std_planes.bin")


# ── Loaders ──────────────────────────────────────────────────────────────────

def load_pcd(path):
    pcd = o3d.io.read_point_cloud(path)
    if len(pcd.points) == 0:
        raise ValueError(f"Empty or invalid PCD: {path}")
    print(f"[map]      {len(pcd.points):,} points loaded from {path}")
    return pcd


def load_keyframe_poses(path):
    """Read keyframe_poses.bin written by saveSTDDatabase()."""
    poses = []
    with open(path, 'rb') as f:
        n = struct.unpack('I', f.read(4))[0]
        for _ in range(n):
            x, y, z = struct.unpack('3d', f.read(24))
            poses.append([x, y, z])
    poses = np.array(poses)
    print(f"[poses]    {len(poses)} keyframe positions loaded")
    return poses


def load_std_planes(path):
    """Read std_planes.bin — one plane cloud per keyframe."""
    keyframes = []
    with open(path, 'rb') as f:
        n_frames = struct.unpack('I', f.read(4))[0]
        for _ in range(n_frames):
            n_pts = struct.unpack('I', f.read(4))[0]
            pts = []
            for _ in range(n_pts):
                buf = struct.unpack('6f', f.read(24))
                pts.append(buf)         # x, y, z, nx, ny, nz
            keyframes.append(np.array(pts) if pts else np.zeros((0, 6)))
    print(f"[planes]   {n_frames} keyframes, plane clouds loaded")
    return keyframes


# ── Geometry builders ────────────────────────────────────────────────────────

def make_trajectory_line(poses, color=(0.0, 1.0, 0.2)):
    """Connect keyframe positions with a coloured line strip."""
    if len(poses) < 2:
        return None
    lines  = [[i, i + 1] for i in range(len(poses) - 1)]
    colors = [list(color)] * len(lines)
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(poses)
    ls.lines  = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


def make_keyframe_spheres(poses, radius=0.25):
    """One small sphere per keyframe, coloured along a blue→red gradient."""
    mesh = o3d.geometry.TriangleMesh()
    n = len(poses)
    for i, pos in enumerate(poses):
        t = i / max(n - 1, 1)
        color = [t, 0.2, 1.0 - t]          # blue → red along the path
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=8)
        sphere.translate(pos)
        sphere.paint_uniform_color(color)
        mesh += sphere
    mesh.compute_vertex_normals()
    return mesh


def make_plane_cloud(keyframes, every_nth=1):
    """
    Merge plane centroids from every N-th keyframe into a single point cloud.
    Each keyframe gets a distinct colour so you can see per-frame coverage.
    """
    all_pts  = []
    all_cols = []
    n = len(keyframes)
    cmap = plt_cmap(n)                      # one colour per keyframe

    for i, kf in enumerate(keyframes):
        if i % every_nth != 0:
            continue
        if len(kf) == 0:
            continue
        centers = kf[:, :3]
        all_pts.append(centers)
        all_cols.append(np.tile(cmap[i], (len(centers), 1)))

    if not all_pts:
        return None

    pts  = np.vstack(all_pts)
    cols = np.vstack(all_cols)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    pc.colors = o3d.utility.Vector3dVector(cols)
    return pc


def plt_cmap(n):
    """Simple rainbow colour map for n entries."""
    t  = np.linspace(0, 1, max(n, 1))
    r  = np.clip(np.abs(t * 6 - 3) - 1, 0, 1)
    g  = np.clip(2 - np.abs(t * 6 - 2), 0, 1)
    b  = np.clip(2 - np.abs(t * 6 - 4), 0, 1)
    return np.stack([r, g, b], axis=1)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # ── Load data ────────────────────────────────────────────────────────────
    pcd = load_pcd(MAP_PCD)

    poses     = None
    keyframes = None

    if os.path.exists(POSES_BIN):
        poses = load_keyframe_poses(POSES_BIN)
    else:
        print(f"[warn] {POSES_BIN} not found — no pose overlay")

    if os.path.exists(PLANES_BIN):
        keyframes = load_std_planes(PLANES_BIN)
    else:
        print(f"[warn] {PLANES_BIN} not found — no plane overlay")

    # ── Style the map cloud ──────────────────────────────────────────────────
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        intensity = colors[:, 0]
        intensity_norm = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-6)
        grey = np.stack([intensity_norm * 0.8] * 3, axis=1)
        pcd.colors = o3d.utility.Vector3dVector(grey)
    else:
        pcd.paint_uniform_color([0.45, 0.45, 0.45])

    # ── Build scene ──────────────────────────────────────────────────────────
    geometries = [pcd]

    # World-frame axes
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])
    geometries.append(axis)

    if poses is not None and len(poses) > 0:
        traj = make_trajectory_line(poses, color=(0.0, 0.9, 0.1))
        if traj:
            geometries.append(traj)

        spheres = make_keyframe_spheres(poses, radius=0.25)
        geometries.append(spheres)

        print(f"[info]     trajectory: {len(poses)} keyframes, "
              f"length ≈ {np.sum(np.linalg.norm(np.diff(poses, axis=0), axis=1)):.1f} m")

    if keyframes is not None:
        plane_pc = make_plane_cloud(keyframes, every_nth=1)
        if plane_pc:
            geometries.append(plane_pc)

    # ── Render ───────────────────────────────────────────────────────────────
    print("\nControls:")
    print("  Left-drag  — rotate")
    print("  Scroll     — zoom")
    print("  Shift+drag — pan")
    print("  Q / Esc    — quit\n")

    o3d.visualization.draw_geometries(
        geometries,
        window_name="Map + Keyframes",
        width=1280,
        height=800,
        point_show_normal=False,
    )


if __name__ == "__main__":
    main()