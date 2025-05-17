import numpy as np
import open3d as o3d

def farthest_point_sample(pointCloud, numPointsToSample):
    numPoints = pointCloud.shape[0]

    sampledIndices = np.zeros(numPointsToSample, dtype=int)
    minDistances = np.full(numPoints, np.inf)

    center = np.mean(pointCloud, axis=0)
    firstIndex = np.argmax(np.sqrt(np.sum((pointCloud - center) ** 2, axis=1)))
    sampledIndices[0] = firstIndex

    for i in range(1, numPointsToSample):
        lastIndex = sampledIndices[i - 1]
        lastPoint = pointCloud[lastIndex, :]

        distances = np.sqrt(np.sum((pointCloud - lastPoint) ** 2, axis=1))
        minDistances = np.minimum(minDistances, distances)

        farthestIndex = np.argmax(minDistances)
        sampledIndices[i] = farthestIndex

    return sampledIndices


def compute_face_areas(vertices, faces):
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    edge1 = v1 - v0
    edge2 = v2 - v0
    normals = np.cross(edge1, edge2)

    areas = 0.5 * np.linalg.norm(normals, axis=1)

    return areas


def compute_signed_distance_and_closest_goemetry(query_points, scene):
    closest_points = scene.compute_closest_points(query_points)
    distance = np.linalg.norm(query_points - closest_points['points'].numpy(),
                              axis=-1)
    direction = (query_points - closest_points['points'].numpy()) / distance[:, np.newaxis]

    rays = np.concatenate([query_points, direction], axis=-1)

    intersection_counts = scene.count_intersections(rays).numpy()
    is_inside = intersection_counts % 2 == 1
    distance[is_inside] *= -1
    return distance, closest_points


def compute_rgb_at_uv(uv_coords, texture_image, bilinear = True):
    map_size = [texture_image.shape[0], texture_image.shape[1]]
    if not bilinear:
        # change from uv space to image space
        map_coords = np.column_stack((uv_coords[:, 0] * map_size[1], uv_coords[:, 1] * map_size[0])).astype(int)

        # flip the y image vertically and clamp
        clamped_coord_x = np.clip(map_coords[:, 0], 0, map_size[1] - 1)
        clamped_coord_y = map_size[0] - 1 - np.clip(map_coords[:, 1], 0, map_size[0] - 1)
        clamped_coord = np.column_stack((clamped_coord_x, clamped_coord_y))

        r = texture_image[clamped_coord[:, 1], clamped_coord[:, 0], 0]
        g = texture_image[clamped_coord[:, 1], clamped_coord[:, 0], 1]
        b = texture_image[clamped_coord[:, 1], clamped_coord[:, 0], 2]

        rgb = np.column_stack((r, g, b))
    else:
        # four nearest pixel
        x = uv_coords[:, 0] * (map_size[1] - 1)
        y = (1 - uv_coords[:, 1]) * (map_size[0] - 1)
        x0, y0, x1, y1 = np.floor(x).astype(int), np.floor(y).astype(int), np.ceil(x).astype(int), np.ceil(y).astype(int)

        # clip to resolution range
        x0 = np.clip(x0, 0, map_size[1] - 1)
        x1 = np.clip(x1, 0, map_size[1] - 1)
        y0 = np.clip(y0, 0, map_size[0] - 1)
        y1 = np.clip(y1, 0, map_size[0] - 1)

        weight_x1 = x - x0
        weight_x0 = 1 - weight_x1
        weight_y1 = y - y0
        weight_y0 = 1 - weight_y1
        weight_x0 = weight_x0.reshape(-1, 1)
        weight_x1 = weight_x1.reshape(-1, 1)
        weight_y0 = weight_y0.reshape(-1, 1)
        weight_y1 = weight_y1.reshape(-1, 1)


        pixel00 = texture_image[y0, x0]
        pixel01 = texture_image[y1, x0]
        pixel10 = texture_image[y0, x1]
        pixel11 = texture_image[y1, x1]

        rgb =  (pixel00 * weight_x0 * weight_y0 +
                pixel01 * weight_x0 * weight_y1 +
                pixel10 * weight_x1 * weight_y0 +
                pixel11 * weight_x1 * weight_y1)
    return rgb



def SampleFromSurfaceOld(vertices, faces, face_normal, face_area, face_selected, num_sampled_points, var1, var2, noise_dir, gaussian=True):
    # Generate random values u and v
    if noise_dir == "uniform":
        num_sampled_points = num_sampled_points * 2
    u = np.random.rand(num_sampled_points, 1)
    v = np.random.rand(num_sampled_points, 1)

    # Calculate barycentric coordinates w0, w1, and w2
    w0 = 1 - np.sqrt(u)
    w1 = np.sqrt(u) * (1 - v)
    w2 = np.sqrt(u) * v

    # Calculate probabilities for face sampling
    probabilities = face_area[face_selected] / np.sum(face_area[face_selected])

    # Sample face indices using the probability distribution
    sample_face_idxs = np.random.choice(face_selected, num_sampled_points, p=probabilities)

    # Calculate sample points
    v0 = vertices[faces[sample_face_idxs, 0], :]
    v1 = vertices[faces[sample_face_idxs, 1], :]
    v2 = vertices[faces[sample_face_idxs, 2], :]
    surface_points = v0 * w0 + v1 * w1 + v2 * w2

    # Add noise to the normal direction or uniform directions
    sample_normal = face_normal[sample_face_idxs, :]
    if noise_dir == "normal":
        if gaussian:
            noise1 = np.random.normal(0, var1 ** 0.5, (num_sampled_points, 1))
            noise11 = -noise1
            noise2 = np.random.normal(0, var2 ** 0.5, (num_sampled_points, 1))
            noise21 = -noise2
        else:
            noise1 = np.random.uniform(-var1** 0.5, var1** 0.5, (num_sampled_points, 1))
            noise11 = -noise1
            noise2 = np.random.uniform(-var2** 0.5, var2** 0.5, (num_sampled_points, 1))
            noise21 = -noise2
        sampled_points = np.vstack((surface_points + sample_normal * noise1, surface_points + sample_normal * noise2, \
                                    surface_points + sample_normal * noise11, surface_points + sample_normal * noise21))
    elif noise_dir == "uniform":
        if gaussian:
            noise1 = np.random.normal(0, var1**0.5, (num_sampled_points, 3))
            noise2 = np.random.normal(0, var2**0.5, (num_sampled_points, 3))
        else:
            noise1 = np.random.uniform(-var1** 0.5, var1** 0.5, (num_sampled_points, 3))
            noise2 = np.random.uniform(-var2** 0.5, var2** 0.5, (num_sampled_points, 3))
        
        sampled_points = np.vstack((surface_points + noise1, surface_points + noise2))

    return sampled_points

def SampleFromSurface(vertices, faces, face_area, face_selected, num_sampled_points, var, symmetric=True):
    if symmetric:
        num_sampled_points = int(num_sampled_points / 2)
    # Generate random values u and v
    u = np.random.rand(num_sampled_points, 1)
    v = np.random.rand(num_sampled_points, 1)

    # Calculate barycentric coordinates w0, w1, and w2
    w0 = 1 - np.sqrt(u)
    w1 = np.sqrt(u) * (1 - v)
    w2 = np.sqrt(u) * v

    # Calculate probabilities for face sampling
    probabilities = face_area[face_selected] / np.sum(face_area[face_selected])

    # Sample face indices using the probability distribution
    sample_face_idxs = np.random.choice(face_selected, num_sampled_points, p=probabilities)

    # Calculate sample points
    v0 = vertices[faces[sample_face_idxs, 0], :]
    v1 = vertices[faces[sample_face_idxs, 1], :]
    v2 = vertices[faces[sample_face_idxs, 2], :]
    surface_points = v0 * w0 + v1 * w1 + v2 * w2

    if symmetric:
        noise1 = np.random.normal(0, var**0.5, (num_sampled_points, 3))
        noise2 = np.random.normal(0, var**0.5, (num_sampled_points, 3))
        sampled_points = np.vstack((surface_points + noise1, surface_points + noise2))
    else:
        noise = np.random.normal(0, var**0.5, (num_sampled_points, 3))    
        sampled_points = surface_points + noise
        
    return sampled_points

def SampleToPC(vertices, faces, face_normal, faces_uv, mesh_texture, face_area, face_selected, num_sampled_points):
    # Generate random values u and v
    u = np.random.rand(num_sampled_points, 1)
    v = np.random.rand(num_sampled_points, 1)

    # Calculate barycentric coordinates w0, w1, and w2
    w0 = 1 - np.sqrt(u)
    w1 = np.sqrt(u) * (1 - v)
    w2 = np.sqrt(u) * v

    # Calculate probabilities for face sampling
    probabilities = face_area[face_selected] / np.sum(face_area[face_selected])

    # Sample face indices using the probability distribution
    sample_face_idxs = np.random.choice(face_selected, num_sampled_points, p=probabilities)

    # Calculate sample points
    v0 = vertices[faces[sample_face_idxs, 0], :]
    v1 = vertices[faces[sample_face_idxs, 1], :]
    v2 = vertices[faces[sample_face_idxs, 2], :]
    surface_points = v0 * w0 + v1 * w1 + v2 * w2
    surface_normal = face_normal[sample_face_idxs, :]

    textureTriUVs = faces_uv[sample_face_idxs, :, :]
    pointUVs = textureTriUVs[:, 0, :].reshape(num_sampled_points, 2) * w0 + \
               textureTriUVs[:, 1, :].reshape(num_sampled_points, 2) * w1 + \
               textureTriUVs[:, 2, :].reshape(num_sampled_points, 2) * w2

    surface_points_color = compute_rgb_at_uv(pointUVs, mesh_texture, True)

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(surface_points)
    point_cloud.colors = o3d.utility.Vector3dVector(surface_points_color / 255)
    # point_cloud.normals = o3d.utility.Vector3dVector(surface_normal)
    return point_cloud


def SampleFromCube(num_rand_points, cube_length = 1):
    sampled_points = np.random.uniform(-cube_length / 2, cube_length / 2, (num_rand_points, 3))
    return sampled_points


def compute_vertex_ring(vertices, triangles):
    # one_ring_neighbors = [[] for i in range(vertices.shape[0])]
    # for triangle in triangles:
    #     for vertex_id in triangle:
    #         for neighbor_id in triangle:
    #             if neighbor_id != vertex_id and neighbor_id not in one_ring_neighbors[vertex_id]:
    #                 one_ring_neighbors[vertex_id].append(neighbor_id)

    # more fast using set
    one_ring_neighbors = [set() for i in range(vertices.shape[0])]
    for triangle in triangles:
        for vertex_id in triangle:
            neighbors = set(triangle) - {vertex_id}
            one_ring_neighbors[vertex_id].update(neighbors)
    return one_ring_neighbors


def compute_vertex_to_face(vertices, triangles):
    # vertex_to_faces = {i: [] for i in range(vertices.shape[0])}
    #
    # for i, triangle in enumerate(triangles):
    #     for vertex_id in triangle:
    #         if i not in vertex_to_faces[vertex_id]:
    #             vertex_to_faces[vertex_id].append(i)

    vertex_to_faces = {i: set() for i in range(vertices.shape[0])}

    for i, triangle in enumerate(triangles):
        for vertex_id in triangle:
            vertex_to_faces[vertex_id].add(i)

    return vertex_to_faces

def SampleFromBoundingSphere(center, radius, numSamples):
    """
    Sample points uniformly from a bounding sphere.

    Args:
        center (np.ndarray): Center of the sphere.
        radius (float): Radius of the sphere.
        numSamples (int): Number of samples to generate.

    Returns:
        np.ndarray: Sampled points in shape (numSamples, 3).
    """
    assert len(center) == 3, "Center must be a 3D point."
    assert radius > 0, "Radius must be positive."

    points = np.random.normal(size=(numSamples, 3))
    points /= np.linalg.norm(points, axis=1)[:, np.newaxis]  # Normalize to unit sphere
    points *= radius  # Scale to the desired radius
    points += center  # Translate to the center

    return points

def SampleFromAABBSurface(min_bound, max_bound, numSamples):
    """
    Uniformly sample points from the surface of an axis-aligned bounding box (AABB).
    
    Args:
        min_bound (np.ndarray): 3D coordinate of the minimum corner (shape: (3,))
        max_bound (np.ndarray): 3D coordinate of the maximum corner (shape: (3,))
        numSamples (int): Number of points to sample.
        
    Returns:
        np.ndarray: Sampled points on the surface, shape (numSamples, 3)
    """
    min_bound = np.asarray(min_bound).reshape(3)
    max_bound = np.asarray(max_bound).reshape(3)
    
    assert np.all(max_bound > min_bound), "max_bound must be greater than min_bound in all dimensions"
    
    # Box dimensions
    x0, y0, z0 = min_bound
    x1, y1, z1 = max_bound
    dx, dy, dz = x1 - x0, y1 - y0, z1 - z0

    # Surface areas of the 6 faces
    areas = np.array([
        dy * dz,  # x-min or x-max face
        dy * dz,
        dx * dz,  # y-min or y-max face
        dx * dz,
        dx * dy,  # z-min or z-max face
        dx * dy
    ])
    face_probs = areas / np.sum(areas)

    # Choose face index for each sample
    face_indices = np.random.choice(6, size=numSamples, p=face_probs)

    samples = np.zeros((numSamples, 3))
    for i, face in enumerate(face_indices):
        u = np.random.rand()
        v = np.random.rand()

        if face == 0:  # x = x0
            samples[i] = [x0, y0 + u * dy, z0 + v * dz]
        elif face == 1:  # x = x1
            samples[i] = [x1, y0 + u * dy, z0 + v * dz]
        elif face == 2:  # y = y0
            samples[i] = [x0 + u * dx, y0, z0 + v * dz]
        elif face == 3:  # y = y1
            samples[i] = [x0 + u * dx, y1, z0 + v * dz]
        elif face == 4:  # z = z0
            samples[i] = [x0 + u * dx, y0 + v * dy, z0]
        elif face == 5:  # z = z1
            samples[i] = [x0 + u * dx, y0 + v * dy, z1]

    return samples