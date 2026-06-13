from __future__ import annotations

from typing import List, Tuple

import numpy as np


def region_growing_clusters(
    points: List[Tuple[int, int]],
) -> List[List[Tuple[int, int]]]:
    """Cluster frontier cells via region growing on the grid.

    Two frontier cells are connected if they are neighbors on the grid.
    We define this using 8-connectivity.

    Key Idea:
    1. Maintain a set of unvisited points.
    2. While unvisited points remain:
         a. Pick a seed point from unvisited.
         b. Grow a cluster from the seed by adding all connected neighbors.
    Returns:
    - List of clusters, each a list of (x, y) points.
    """

    if not points:
        return []

    remaining = set(points)
    clusters: List[List[Tuple[int, int]]] = []

    deltas = [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (1, -1), (-1, 1), (-1, -1),
    ]

    #===== STUDENT IMPLEMENTATION BEGIN =====
    for point in points:
        if point not in remaining:
            continue

        cluster = []
        stack = [point]
        remaining.remove(point)

        while stack:
            current = stack.pop()
            cluster.append(current)

            for dx, dy in deltas:
                neighbor = (current[0] + dx, current[1] + dy)
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)

        clusters.append(cluster)
    # raise NotImplementedError("Student TODO: Implement region_growing_clusters")

    #===== STUDENT IMPLEMENTATION END =====

    return clusters


def _pca_first_eigen(points_xy: np.ndarray) -> tuple[float, np.ndarray]:
    """Return (largest_eigenvalue, corresponding_unit_eigenvector)."""

    # points_xy shape: (N,2)
    centered = points_xy - points_xy.mean(axis=0, keepdims=True)
    cov = (centered.T @ centered) / max(1, (len(points_xy) - 1))
    evals, evecs = np.linalg.eigh(cov)
    idx = int(np.argmax(evals))
    v = evecs[:, idx]
    v = v / (np.linalg.norm(v) + 1e-12)
    return float(evals[idx]), v


def split_cluster_pca_recursive(
    cluster: List[Tuple[int, int]],
    min_cluster_size: int,
    eigenvalue_threshold: float,
) -> List[List[Tuple[int, int]]]:
    """Recursively split a cluster into two along its first PCA axis.

    Splits only if:
    - cluster size is at least 2 * min_cluster_size
    - largest PCA eigenvalue exceeds eigenvalue_threshold
    - both halves after split have at least min_cluster_size points
    """

    if len(cluster) < 2 * max(1, min_cluster_size):
        return [cluster]

    pts = np.array(cluster, dtype=np.float32)
    lam1, v1 = _pca_first_eigen(pts)
    if lam1 <= float(eigenvalue_threshold):
        return [cluster]

    centered = pts - pts.mean(axis=0, keepdims=True)
    proj = centered @ v1
    median = float(np.median(proj))
    left_mask = proj <= median

    left = [cluster[i] for i in range(len(cluster)) if bool(left_mask[i])]
    right = [cluster[i] for i in range(len(cluster)) if not bool(left_mask[i])]

    if len(left) < min_cluster_size or len(right) < min_cluster_size:
        return [cluster]

    # Recurse
    out: List[List[Tuple[int, int]]] = []
    for part in (left, right):
        out.extend(split_cluster_pca_recursive(part, min_cluster_size=min_cluster_size, eigenvalue_threshold=eigenvalue_threshold))
    return out


def frontier_clusters_region_pca(
    points: List[Tuple[int, int]],
    min_cluster_size: int = 8,
    eigenvalue_threshold: float = 25.0,
) -> List[List[Tuple[int, int]]]:
    """Cluster frontiers by region growing then split large blobs using PCA.
    """

    clusters = region_growing_clusters(points)

    refined: List[List[Tuple[int, int]]] = []
    for c in clusters:
        refined.extend(
            split_cluster_pca_recursive(
                c,
                min_cluster_size=min_cluster_size,
                eigenvalue_threshold=eigenvalue_threshold,
            )
        )

    # Keep deterministic ordering: biggest clusters first.
    refined.sort(key=len, reverse=True)
    return refined


def cluster_representative_point(cluster: List[Tuple[int, int]]) -> Tuple[int, int]:
    """Pick a representative frontier point for a cluster.

    We choose the point closest to the cluster mean (snapped to an actual point).
    """

    if not cluster:
        raise ValueError("cluster must be non-empty")

    pts = np.array(cluster, dtype=np.float32)
    mean = pts.mean(axis=0)
    d2 = ((pts - mean[None, :]) ** 2).sum(axis=1)
    idx = int(d2.argmin())
    x, y = pts[idx]
    return (int(x), int(y))


def kmeans_centroids(points: List[Tuple[int, int]], k: int, seed: int = 0, iters: int = 15) -> List[Tuple[int, int]]:
    """Tiny K-Means implementation.

    If len(points) <= k, returns the points themselves (deduped order preserved).
    """

    if k <= 0 or not points:
        return []

    unique_points = list(dict.fromkeys(points))
    if len(unique_points) <= k:
        return unique_points

    rng = np.random.default_rng(seed)
    pts = np.array(unique_points, dtype=np.float32)

    # Init centroids by sampling points.
    centroids = pts[rng.choice(len(pts), size=k, replace=False)]

    for _ in range(iters):
        # Assign
        d2 = ((pts[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = d2.argmin(axis=1)
        # Update
        new_centroids = []
        for j in range(k):
            members = pts[labels == j]
            if len(members) == 0:
                new_centroids.append(centroids[j])
            else:
                new_centroids.append(members.mean(axis=0))
        centroids = np.array(new_centroids, dtype=np.float32)

    # Snap each centroid to the nearest original point so targets are always valid cells.
    result: List[Tuple[int, int]] = []
    for c in centroids:
        d2 = ((pts - c[None, :]) ** 2).sum(axis=1)
        idx = int(d2.argmin())
        x, y = pts[idx]
        result.append((int(x), int(y)))

    # Deduplicate while preserving order.
    return list(dict.fromkeys(result))
