import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree

def extract_mesh_boundary(mesh):
    edges = set()
    boundary_edges = set()

    for face in mesh.triangles:
        for i in range(3):
            edge = (face[i], face[(i + 1) % 3])
            if edge[1] < edge[0]:
                edge = (edge[1], edge[0])
            else:
                egde = (edge[0], edge[1])

            if edge in edges:
                edges.remove(edge)
            else:
                edges.add(edge)

    boundary_edges = list(edges)
    return boundary_edges

def extract_mesh_edges(mesh):
    edges = set()

    for face in mesh.triangles:
        for i in range(3):
            edge = (face[i], face[(i + 1) % 3])
            if edge[1] < edge[0]:
                edge = (edge[1], edge[0])
            else:
                edge = (edge[0], edge[1])
            edges.add(edge)

    return edges

def extract_mesh_wireframe(mesh, color=[1, 0, 0]):
    mesh_edges = extract_mesh_edges(mesh)

    wireframe = o3d.geometry.LineSet()
    wireframe.points = mesh.vertices
    wireframe.lines = o3d.utility.Vector2iVector(mesh_edges)
    wireframe.colors = o3d.utility.Vector3dVector(np.array([color] * len(mesh_edges)))

    return wireframe

def extract_colored_vertex_selected_wireframe(mesh, vSelected, mesh_color=[0, 0, 0], submesh_color=[0, 0, 1]):
    all_edges = extract_mesh_edges(mesh)

    vSelected_set = set(vSelected)

    lines = []
    colors = []
    for edge in all_edges:
        if edge[0] in vSelected_set and edge[1] in vSelected_set:
            colors.append(submesh_color)
        else:
            colors.append(mesh_color)
        lines.append(edge)

    wireframe = o3d.geometry.LineSet()
    wireframe.points = mesh.vertices
    wireframe.lines = o3d.utility.Vector2iVector(lines)
    wireframe.colors = o3d.utility.Vector3dVector(np.array(colors))

    return wireframe

def save_pc_snap(pc_list, point_size, camera_info_json, output_png_path):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=800, height=800, left=50, top=50)
    for geom in pc_list:
        vis.add_geometry(geom)
    render_option = vis.get_render_option()
    render_option.point_size = point_size

    ctr = vis.get_view_control()
    camera_info = o3d.io.read_pinhole_camera_parameters(camera_info_json)
    ctr.convert_from_pinhole_camera_parameters(camera_info)
    # vis.update_geometry(pc)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(output_png_path)
    vis.destroy_window()

def find_k_nearest_neighbors_to_point(points: np.ndarray, query_point: np.ndarray, k: int):
    """
    Find k nearest neighbors in 3D point cloud to an arbitrary query point.

    Args:
        points (np.ndarray): N x 3 array of 3D coordinates.
        query_point (np.ndarray): 3D coordinate of the query point.
        k (int): Number of neighbors to find.

    Returns:
        List[int]: Indices of the k nearest neighbors in `points`.
    """
    assert points.shape[1] == 3, "Points must be 3D."
    assert query_point.shape == (3,), "Query point must be 3D."

    tree = cKDTree(points)
    dists, indices = tree.query(query_point, k=k)

    # Handle scalar case when k=1
    if k == 1:
        return [indices]
    else:
        return indices.tolist()

def find_9_center_idxs(points: np.ndarray):
    """
    Input:
        points: Nx3 numpy array representing 3D point coordinates
    Output:
        idxs: A list of 9 indices:
              - First is the index of the globally closest point to the center
              - The remaining 8 are indices of the closest points to the center
                within each of the 8 spatial octants (None if an octant has no points)
    """
    assert points.shape[1] == 3, "Input must be a Nx3 point cloud"

    # Step 1: Compute the overall center of all points
    center = np.mean(points, axis=0)

    # Step 2: Find the point closest to the center (global)
    dists_to_center = np.linalg.norm(points - center, axis=1)
    global_idx = np.argmin(dists_to_center)

    # Step 3: Compute bounding box center for octant partitioning
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    bbox_center = (min_bound + max_bound) / 2

    # Step 4: Divide points into 8 octants and find closest point in each
    octant_idxs = []
    for i in range(8):
        # Binary code (000 to 111) indicates which side of bbox center in each dimension
        mask = np.ones(len(points), dtype=bool)
        for dim in range(3):
            if (i >> dim) & 1:
                mask &= points[:, dim] >= bbox_center[dim]
            else:
                mask &= points[:, dim] < bbox_center[dim]

        if np.any(mask):
            sub_points = points[mask]
            sub_dists = np.linalg.norm(sub_points - center, axis=1)
            local_idx_in_sub = np.argmin(sub_dists)
            global_masked_indices = np.nonzero(mask)[0]
            octant_idxs.append(global_masked_indices[local_idx_in_sub])
        else:
            octant_idxs.append(None)

    return [global_idx] + octant_idxs


import numpy as np

def find_tangent_points(sdfSamples: np.ndarray, idx: int, planeThre: float, r: float):
    """
    Find points P' such that:
        - P' lies approximately on the tangent plane at point P
        - distance |PP'| is within r * bbox_diagonal

    Args:
        sdfSamples (np.ndarray): N x 14 array, each row is:
            [sampleXYZ (3), nearestXYZ (3), color (3), normal (3), sdf (1), lp_idx]
        idx (int): index of reference point P
        planeThre (float): threshold for distance from the tangent plane (e.g., 0.01)
        r (float): normalized radius relative to bounding box diagonal (range: 0 ~ 1)

    Returns:
        List of indices of qualifying points P'
    """

    sample_points = sdfSamples[:, :3]
    normals = sdfSamples[:, 9:12]

    P = sample_points[idx]            # Reference point
    normal_vec = normals[idx]         # Normal at reference point

    # Compute bounding box diagonal and scaled radius
    min_bound = np.min(sample_points, axis=0)
    max_bound = np.max(sample_points, axis=0)
    diag_length = np.linalg.norm(max_bound - min_bound)
    radius = r * diag_length

    # Vector PP' for all points
    vec_PPprime = sample_points - P  # shape: (N, 3)
    dist_PPprime = np.linalg.norm(vec_PPprime, axis=1)

    # Compute signed distance from each P' to the tangent plane at P
    # Using point-to-plane distance formula: (P' - P) · n / ||n||
    plane_dists = np.abs(np.dot(vec_PPprime, normal_vec)) / (np.linalg.norm(normal_vec) + 1e-8)

    # Conditions
    cond_plane = plane_dists < planeThre
    cond_dist = dist_PPprime < radius

    final_mask = cond_plane & cond_dist
    final_mask[idx] = False  # Exclude the point itself

    qualifying_indices = np.nonzero(final_mask)[0].tolist()
    return qualifying_indices

def find_normal_points(sdfSamples: np.ndarray, idx: int, norThre: float, r: float):
    """
    Find points P' such that:
        - vector PP' is approximately perpendicular to the normal at Pn
        - distance |PP'| is within r * bbox_diagonal

    Args:
        sdfSamples (np.ndarray): N x 14 array, each row is:
            [sampleXYZ (3), nearestXYZ (3), color (3), normal (3), sdf (1), lp_idx]
        idx (int): index of reference point P
        norThre (float): cosine threshold for "perpendicularity" (e.g., 0.2)
        r (float): normalized radius relative to bounding box diagonal (range: 0 ~ 1)

    Returns:
        List of indices of qualifying points P'
    """

    sample_points = sdfSamples[:, :3]
    normals = sdfSamples[:, 9:12]  # Surface normals

    P = sample_points[idx]             # The selected sample point
    normal_vec = normals[idx]          # Normal at Pn (from nearest surface point)

    # Compute bounding box diagonal
    min_bound = np.min(sample_points, axis=0)
    max_bound = np.max(sample_points, axis=0)
    diag_length = np.linalg.norm(max_bound - min_bound)
    radius = r * diag_length

    # Vector PP' for all points
    vec_PPprime = sample_points - P  # shape: (N, 3)
    dist_PPprime = np.linalg.norm(vec_PPprime, axis=1)  # shape: (N,)

    # Compute cosine of angle between PP' and normal (no need to normalize)
    dot_products = np.abs(np.dot(vec_PPprime, normal_vec))
    norm_PPprime = dist_PPprime
    norm_normal = np.linalg.norm(normal_vec)
    cos_angles = dot_products / (norm_PPprime * norm_normal + 1e-8)

    # Conditions
    cond_angle = cos_angles > 1 - norThre
    cond_dist = dist_PPprime < radius

    final_mask = cond_angle & cond_dist
    final_mask[idx] = False  # exclude self

    qualifying_indices = np.nonzero(final_mask)[0].tolist()
    return qualifying_indices


def find_tangent_points2(sdfSamples: np.ndarray, idx: int, planeThre: float, r: float):
    """
    Find points P' such that:
        - P' lies approximately on the tangent plane at point P
        - distance |PP'| is within r * bbox_diagonal

    Args:
        sdfSamples (np.ndarray): N x 14 array, each row is:
            [sampleXYZ (3), nearestXYZ (3), color (3), normal (3), sdf (1), lp_idx]
        idx (int): index of reference point P
        planeThre (float): maximum allowed point-to-plane distance (e.g., 0.01)
        r (float): normalized radius relative to bounding box diagonal

    Returns:
        List[int]: indices of points that lie close to the tangent plane at P
    """

    sample_points = sdfSamples[:, :3]
    normals = sdfSamples[:, 9:12]

    P = sample_points[idx]
    normal_vec = normals[idx]

    # Compute bounding box diagonal length and search radius
    min_bound = np.min(sample_points, axis=0)
    max_bound = np.max(sample_points, axis=0)
    diag_length = np.linalg.norm(max_bound - min_bound)
    radius = r * diag_length

    # Compute vectors PP' and their lengths
    vec_PPprime = sample_points - P  # shape: (N, 3)
    dist_PPprime = np.linalg.norm(vec_PPprime, axis=1)

    # Compute signed distance to tangent plane: (P' - P) · n / ||n||
    plane_dists = np.abs(np.dot(vec_PPprime, normal_vec)) / (np.linalg.norm(normal_vec) + 1e-8)

    # Apply distance-to-plane and radius filters
    cond_plane = plane_dists < planeThre * diag_length
    cond_radius = dist_PPprime < radius

    final_mask = cond_plane & cond_radius
    final_mask[idx] = False  # Exclude self

    return np.nonzero(final_mask)[0].tolist()

def find_normal_points2(sdfSamples: np.ndarray, idx: int, disThre: float, r: float):
    """
    Find points P' such that:
        - distance from P' to the normal line at P is less than disThre
        - and |PP'| < r * bbox_diagonal

    Args:
        sdfSamples (np.ndarray): N x 14 array, each row is:
            [sampleXYZ (3), nearestXYZ (3), color (3), normal (3), sdf (1), lp_idx]
        idx (int): index of reference point P
        disThre (float): max distance to the normal line (e.g., 0.01)
        r (float): normalized radius relative to bounding box diagonal

    Returns:
        List of indices of qualifying points P'
    """
    sample_points = sdfSamples[:, :3]
    normals = sdfSamples[:, 9:12]

    P = sample_points[idx]
    normal_vec = normals[idx]

    # Compute bounding box diagonal
    min_bound = np.min(sample_points, axis=0)
    max_bound = np.max(sample_points, axis=0)
    diag_length = np.linalg.norm(max_bound - min_bound)
    radius = r * diag_length

    vec_PPprime = sample_points - P  # shape: (N, 3)
    dist_PPprime = np.linalg.norm(vec_PPprime, axis=1)

    # Compute distance from point to the line along normal direction
    cross_products = np.cross(vec_PPprime, normal_vec)           # shape: (N, 3)
    dist_to_line = np.linalg.norm(cross_products, axis=1) / (np.linalg.norm(normal_vec) + 1e-8)

    cond_line = dist_to_line < (disThre * diag_length)
    cond_dist = dist_PPprime < radius

    final_mask = cond_line & cond_dist
    final_mask[idx] = False

    return np.nonzero(final_mask)[0].tolist()