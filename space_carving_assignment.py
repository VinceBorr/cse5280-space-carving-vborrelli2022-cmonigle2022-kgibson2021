import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import trimesh
import open3d as o3d


def make_dirs():
    os.makedirs("outputs/silhouettes", exist_ok=True)
    os.makedirs("outputs/cameras", exist_ok=True)
    os.makedirs("outputs/results", exist_ok=True)


def load_mesh(path=None):
    if path:
        mesh = trimesh.load(path, force="mesh")
    else:
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)

    mesh.apply_translation(-mesh.centroid)
    scale = 1.0 / np.max(mesh.extents)
    mesh.apply_scale(scale * 2.0)

    return mesh


def look_at(camera_pos, target=np.array([0, 0, 0]), up=np.array([0, 1, 0])):
    forward = target - camera_pos
    forward = forward / np.linalg.norm(forward)

    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)

    true_up = np.cross(right, forward)

    R = np.vstack([right, true_up, forward])
    T = -R @ camera_pos

    return R, T


def project_points(points, K, R, T):
    cam_points = (R @ points.T + T.reshape(3, 1)).T

    z = cam_points[:, 2]
    valid = z > 0

    img_points = (K @ cam_points.T).T

    u = img_points[:, 0] / img_points[:, 2]
    v = img_points[:, 1] / img_points[:, 2]

    return np.stack([u, v], axis=1), valid


def make_silhouette(mesh, K, R, T, image_size):
    silhouette = np.zeros((image_size, image_size), dtype=np.uint8)

    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)

    projected, valid = project_points(vertices, K, R, T)

    for face in faces:
        if not np.all(valid[face]):
            continue

        pts = projected[face]

        min_u = max(int(np.floor(np.min(pts[:, 0]))), 0)
        max_u = min(int(np.ceil(np.max(pts[:, 0]))), image_size - 1)
        min_v = max(int(np.floor(np.min(pts[:, 1]))), 0)
        max_v = min(int(np.ceil(np.max(pts[:, 1]))), image_size - 1)

        if min_u >= image_size or max_u < 0 or min_v >= image_size or max_v < 0:
            continue

        for y in range(min_v, max_v + 1):
            for x in range(min_u, max_u + 1):
                p = np.array([x, y])

                a, b, c = pts
                v0 = c - a
                v1 = b - a
                v2 = p - a

                dot00 = np.dot(v0, v0)
                dot01 = np.dot(v0, v1)
                dot02 = np.dot(v0, v2)
                dot11 = np.dot(v1, v1)
                dot12 = np.dot(v1, v2)

                denom = dot00 * dot11 - dot01 * dot01

                if denom == 0:
                    continue

                inv_denom = 1 / denom
                u = (dot11 * dot02 - dot01 * dot12) * inv_denom
                v = (dot00 * dot12 - dot01 * dot02) * inv_denom

                if u >= 0 and v >= 0 and u + v <= 1:
                    silhouette[y, x] = 255

    return silhouette


def create_voxel_grid(resolution):
    xs = np.linspace(-1.2, 1.2, resolution)
    ys = np.linspace(-1.2, 1.2, resolution)
    zs = np.linspace(-1.2, 1.2, resolution)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.stack([X, Y, Z], axis=-1).reshape(-1, 3)


def space_carve(voxels, silhouettes, cameras, image_size):
    occupied = np.ones(len(voxels), dtype=bool)

    for i, silhouette in enumerate(silhouettes):
        K, R, T = cameras[i]

        pixels, valid_depth = project_points(voxels, K, R, T)

        u = np.round(pixels[:, 0]).astype(int)
        v = np.round(pixels[:, 1]).astype(int)

        inside_image = (
            (u >= 0) & (u < image_size) &
            (v >= 0) & (v < image_size) &
            valid_depth
        )

        inside_silhouette = np.zeros(len(voxels), dtype=bool)
        idx = np.where(inside_image)[0]

        inside_silhouette[idx] = silhouette[v[idx], u[idx]] > 0

        occupied &= inside_silhouette

        print(f"View {i + 1}: remaining voxels = {occupied.sum()}")

    return voxels[occupied]


def save_point_cloud(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    o3d.io.write_point_cloud("outputs/results/visual_hull.ply", pcd)
    return pcd


def reconstruct_mesh(points):
    if len(points) < 10:
        print("Not enough points for mesh reconstruction.")
        return

    pcd = save_point_cloud(points)
    pcd.estimate_normals()

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=7
    )

    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh("outputs/results/reconstructed_mesh.ply", mesh)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", default=None)
    parser.add_argument("--views", type=int, default=20)
    parser.add_argument("--resolution", type=int, default=48)
    parser.add_argument("--image-size", type=int, default=256)

    args = parser.parse_args()

    make_dirs()

    mesh = load_mesh(args.mesh)

    image_size = args.image_size

    fx = fy = image_size
    cx = cy = image_size / 2

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])

    silhouettes = []
    cameras = []

    angles = np.linspace(0, 2 * np.pi, args.views, endpoint=False)

    for i, angle in enumerate(angles):
        camera_pos = np.array([
            3 * np.cos(angle),
            0.8,
            3 * np.sin(angle)
        ])

        R, T = look_at(camera_pos)

        silhouette = make_silhouette(mesh, K, R, T, image_size)

        silhouettes.append(silhouette)
        cameras.append((K, R, T))

        plt.imsave(
            f"outputs/silhouettes/silhouette_{i:03d}.png",
            silhouette,
            cmap="gray"
        )

        np.savez(
            f"outputs/cameras/camera_{i:03d}.npz",
            K=K,
            R=R,
            T=T
        )

    voxels = create_voxel_grid(args.resolution)

    print(f"Starting voxels: {len(voxels)}")

    carved_points = space_carve(
        voxels,
        silhouettes,
        cameras,
        image_size
    )

    print(f"Final carved voxels: {len(carved_points)}")

    save_point_cloud(carved_points)
    reconstruct_mesh(carved_points)

    print("Done.")
    print("Check outputs/silhouettes/")
    print("Check outputs/results/visual_hull.ply")
    print("Check outputs/results/reconstructed_mesh.ply")


if __name__ == "__main__":
    main()
