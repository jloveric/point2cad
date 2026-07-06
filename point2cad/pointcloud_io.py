import glob
import os

import numpy as np
import open3d as o3d

from point2cad.utils import continuous_labels


LABEL_PROPERTY_NAMES = ("label", "surface_id", "cluster", "scalar_label", "class")


def resolve_input_paths(path_in):
    if os.path.isdir(path_in):
        paths = sorted(glob.glob(os.path.join(path_in, "*.ply")))
        if not paths:
            raise FileNotFoundError(f"No .ply files found in {path_in}")
        return paths
    return [path_in]


def load_points_and_labels(path_in, segment_unlabeled=True, max_points=10000):
    ext = os.path.splitext(path_in)[1].lower()
    if ext == ".xyzc":
        points_labels = np.loadtxt(path_in).astype(np.float32)
        if points_labels.ndim == 1:
            points_labels = points_labels[None, :]
        if points_labels.shape[1] != 4:
            raise ValueError(
                f"{path_in} must have 4 columns (x y z surface_id), got {points_labels.shape[1]}"
            )
        points = points_labels[:, :3]
        labels = continuous_labels(points_labels[:, 3].astype(np.int32))
        return farthest_point_sample(points, labels, max_points)

    if ext == ".xyz":
        points = np.loadtxt(path_in).astype(np.float32)
        if points.ndim == 1:
            points = points[None, :]
        if points.shape[1] != 3:
            raise ValueError(f"{path_in} must have 3 columns (x y z), got {points.shape[1]}")
        if not segment_unlabeled:
            raise ValueError(f"{path_in} has no surface labels; enable segmentation for .xyz input")
        points, labels = segment_planar_surfaces(points)
        return farthest_point_sample(points, labels, max_points)

    if ext == ".ply":
        return load_ply_points_and_labels(
            path_in, segment_unlabeled=segment_unlabeled, max_points=max_points
        )

    raise ValueError(f"Unsupported input format: {ext}. Use .xyzc, .xyz, or .ply")


def load_ply_points_and_labels(path_in, segment_unlabeled=True, max_points=10000):
    labels = read_ply_label_property(path_in)
    pcd = o3d.io.read_point_cloud(path_in)
    if len(pcd.points) == 0:
        raise ValueError(f"No points found in {path_in}")

    points = np.asarray(pcd.points, dtype=np.float32)
    if labels is not None:
        if len(labels) != len(points):
            raise ValueError(
                f"Label count ({len(labels)}) does not match point count ({len(points)}) in {path_in}"
            )
        labels = continuous_labels(labels.astype(np.int32))
        return farthest_point_sample(points, labels, max_points)

    if not segment_unlabeled:
        raise ValueError(f"{path_in} has no surface labels and segmentation is disabled")
    points, labels = segment_planar_surfaces(points)
    return farthest_point_sample(points, labels, max_points)


def farthest_point_sample(points, labels, max_points):
    if max_points is None or max_points <= 0 or len(points) <= max_points:
        return points, labels

    unique_labels, counts = np.unique(labels, return_counts=True)
    if len(unique_labels) * 20 <= max_points:
        min_per_label = 20
    elif len(unique_labels) <= max_points:
        min_per_label = 1
    else:
        min_per_label = 0
    targets = np.floor(counts / len(points) * max_points).astype(np.int32)
    min_targets = np.minimum(counts, min_per_label)
    targets = np.maximum(targets, min_targets)
    targets = np.minimum(targets, counts)

    while targets.sum() > max_points:
        reducible = np.where(targets > min_targets)[0]
        if len(reducible) == 0:
            break
        idx = reducible[np.argmax(targets[reducible])]
        targets[idx] -= 1

    remainders = counts / len(points) * max_points - np.floor(
        counts / len(points) * max_points
    )
    while targets.sum() < max_points and np.any(targets < counts):
        expandable = np.where(targets < counts)[0]
        idx = expandable[np.argmax(remainders[expandable])]
        targets[idx] += 1

    sampled_points = []
    sampled_labels = []
    for label, target in zip(unique_labels, targets):
        if target == 0:
            continue
        label_points = points[labels == label]
        label_points = farthest_point_sample_unlabeled(label_points, int(target))
        sampled_points.append(label_points)
        sampled_labels.append(np.full(len(label_points), label, dtype=np.int32))

    points = np.concatenate(sampled_points, axis=0).astype(np.float32)
    labels = np.concatenate(sampled_labels, axis=0).astype(np.int32)
    return points, continuous_labels(labels)


def farthest_point_sample_unlabeled(points, max_points):
    if len(points) <= max_points:
        return points

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    sampled = pcd.farthest_point_down_sample(max_points)
    return np.asarray(sampled.points, dtype=np.float32)


def read_ply_label_property(path_in):
    property_names = []
    vertex_count = None
    in_vertex = False

    with open(path_in, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped == "end_header":
                break
            if stripped.startswith("element vertex"):
                vertex_count = int(stripped.split()[-1])
                in_vertex = True
                continue
            if stripped.startswith("element ") and not stripped.startswith("element vertex"):
                in_vertex = False
                continue
            if in_vertex and stripped.startswith("property "):
                parts = stripped.split()
                property_names.append(parts[-1])

    label_name = next((name for name in LABEL_PROPERTY_NAMES if name in property_names), None)
    if label_name is None:
        return None

    columns = []
    with open(path_in, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.strip() == "end_header":
                break
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            columns.append(stripped.split())
            if len(columns) == vertex_count:
                break

    if len(columns) != vertex_count:
        raise ValueError(f"Failed to read {vertex_count} vertices from {path_in}")

    label_index = property_names.index(label_name)
    return np.array([row[label_index] for row in columns], dtype=np.int32)


def segment_planar_surfaces(
    points,
    voxel_size=0.08,
    normal_radius=0.3,
    normal_max_nn=30,
    knn=30,
    normal_variance_threshold_deg=60,
    coplanarity_deg=75,
    outlier_ratio=0.75,
    min_plane_edge_length=0.5,
    min_num_points=50,
):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

    if voxel_size > 0:
        extent = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
        effective_voxel_size = min(voxel_size, extent / 1000)
        pcd = pcd.voxel_down_sample(effective_voxel_size)

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius, max_nn=normal_max_nn
        )
    )

    patches = pcd.detect_planar_patches(
        normal_variance_threshold_deg=normal_variance_threshold_deg,
        coplanarity_deg=coplanarity_deg,
        outlier_ratio=outlier_ratio,
        min_plane_edge_length=min_plane_edge_length,
        min_num_points=min_num_points,
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn),
    )

    if len(patches) < 3:
        return segment_planes_ransac(points, min_num_points=min_num_points)

    sample_points = np.asarray(pcd.points)
    sample_labels = -np.ones(len(sample_points), dtype=np.int32)
    for label_id, obb in enumerate(
        sorted(patches, key=lambda patch: float(np.prod(patch.extent)), reverse=True)
    ):
        indices = np.asarray(obb.get_point_indices_within_bounding_box(pcd.points))
        indices = indices[sample_labels[indices] < 0]
        sample_labels[indices] = label_id

    labeled_mask = sample_labels >= 0
    if not np.any(labeled_mask):
        raise RuntimeError("Planar surface segmentation labeled no points")

    from scipy.spatial import cKDTree

    tree = cKDTree(sample_points[labeled_mask])
    _, nearest = tree.query(points.astype(np.float64), k=1)
    labels = continuous_labels(sample_labels[labeled_mask][nearest].astype(np.int32))
    return points, labels


def segment_planes_ransac(points, min_num_points=50, max_planes=128):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

    extent = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
    distance_threshold = max(extent / 200, np.finfo(np.float32).eps)
    remaining = pcd
    remaining_indices = np.arange(len(points))
    labels = -np.ones(len(points), dtype=np.int32)
    plane_models = []

    for plane_id in range(max_planes):
        if len(remaining.points) < min_num_points:
            break

        plane_model, inliers = remaining.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=3,
            num_iterations=2000,
        )
        inliers = np.asarray(inliers, dtype=np.int64)
        if len(inliers) < min_num_points:
            break

        original_indices = remaining_indices[inliers]
        labels[original_indices] = plane_id
        plane_models.append(np.asarray(plane_model, dtype=np.float64))

        remaining = remaining.select_by_index(inliers, invert=True)
        remaining_indices = np.delete(remaining_indices, inliers)

    if len(plane_models) == 0:
        raise RuntimeError("Planar surface segmentation found no patches")

    unlabeled = labels < 0
    if np.any(unlabeled):
        plane_models = np.stack(plane_models, axis=0)
        normals = plane_models[:, :3]
        offsets = plane_models[:, 3]
        distances = np.abs(points[unlabeled] @ normals.T + offsets)
        labels[unlabeled] = np.argmin(distances, axis=1)

    return points, continuous_labels(labels.astype(np.int32))
