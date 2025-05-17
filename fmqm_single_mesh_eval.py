import pandas as pd
import open3d as o3d
import pymeshlab
import os
from PIL import Image
from utils.sample_utils import *
from utils.feature_utils import *
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description="FMQM calculate single mesh score")
parser.add_argument("--ref_obj_path", type=str, required=True, help="Reference mesh object path")
parser.add_argument("--ref_tex_path", type=str, required=True, help="Reference mesh texture path")
parser.add_argument("--dis_obj_path", type=str, required=True, help="Distorted mesh object path")
parser.add_argument("--dis_tex_path", type=str, required=True, help="Distorted mesh texture path")
args = parser.parse_args()

# Hyperparameters
# Sampling configuration
numberFPS = 250
noiseVar = (0.05)**2
targetSampleNumber = 200000
sample_mode = "fps"
# Feature extraction thresholds
tangentThreLevel = 5
normalThreLevel = 5
tangentPlaneLevel = 10
numberFeatures = 5
radius_level = 1.25

localMinRatio = 1 / numberFPS
localMaxRatio = localMinRatio * 1.1
maxRings = 200
num_samp_near_surf_ratio = 1

ref_obj_path = args.ref_obj_path
ref_tex_path = args.ref_tex_path
dis_obj_path = args.dis_obj_path
dis_tex_path = args.dis_tex_path

ms = pymeshlab.MeshSet()
ms.load_new_mesh(ref_obj_path)
ms.meshing_remove_unreferenced_vertices()
ms.meshing_remove_duplicate_vertices()
ms.meshing_remove_duplicate_faces()
ms.meshing_remove_null_faces()

refV = ms.current_mesh().vertex_matrix()
refF = ms.current_mesh().face_matrix()
refVN = ms.current_mesh().vertex_normal_matrix()
refUV = ms.current_mesh().wedge_tex_coord_matrix().reshape(refF.shape[0], 3, 2)
refFN = ms.current_mesh().face_normal_matrix()

faceArea = compute_face_areas(refV, refF)
totalArea = sum(faceArea)

refTexture = np.array(Image.open(ref_tex_path))
refMesh = o3d.geometry.TriangleMesh()
refMesh.vertices = o3d.utility.Vector3dVector(refV)
refMesh.triangles = o3d.utility.Vector3iVector(refF)
refMesh.triangle_normals = o3d.utility.Vector3dVector(refFN)
refMesh.triangle_uvs = o3d.utility.Vector2dVector(refUV.reshape(refF.shape[0] * 3, 2))

vRing = compute_vertex_ring(refV, refF)
vertexToFace = compute_vertex_to_face(refV, refF)

total_fps = numberFPS
if sample_mode == "fps":
    fps_idx = farthest_point_sample(refV, total_fps)
else:
    fps_idx = np.random.choice(refV.shape[0], size=total_fps, replace=False)

local_numbers = fps_idx.shape[0]

sampledBbox = np.zeros((local_numbers, 4))
sampledValues = np.array([])

pbar = tqdm(zip(range(local_numbers), fps_idx), desc=f"Sampling reference {os.path.basename(ref_obj_path).split('.')[0]}")
for idx, vId in pbar:
    # 1-ring
    vSelected = []
    fSelected = list(vertexToFace[vId])
    for f in fSelected:
        for v in refF[f]:
            if v not in vSelected:
                vSelected.append(v)
    localRatio = 0
    rings = 1
    processedVertices = set()
    processedFaces = set()
    while localRatio < localMinRatio:
        newVSelected = []
        newFSelected = []
        for v in vSelected:
            if v in processedVertices:
                continue
            for vNeighborFace in vertexToFace[v]:
                if vNeighborFace not in fSelected and vNeighborFace not in newFSelected and vNeighborFace not in processedFaces:
                    newFSelected.append(vNeighborFace)
                    processedFaces.add(vNeighborFace)
                    for vNeighbor in refF[vNeighborFace]:
                        if vNeighbor not in vSelected and vNeighbor not in newVSelected:
                            newVSelected.append(vNeighbor)
            processedVertices.add(v)
        fSelected.extend(newFSelected)
        vSelected.extend(newVSelected)
        rings += 1
        localRatio = np.sum(faceArea[fSelected]) / np.sum(faceArea)
        if localRatio > localMaxRatio or rings > maxRings:
            break
    vSelected.sort()
    fSelected.sort()
    d = {
        'idx': idx,
        'v': vSelected,
        'f': fSelected,
        'rings': rings
    }
    if rings == maxRings + 1:
        continue

    localSampleNumber = int(targetSampleNumber * np.sum(faceArea[fSelected] / np.sum(faceArea)))
    if localSampleNumber > targetSampleNumber * localMaxRatio:
        localSampleNumber = int(targetSampleNumber * localMaxRatio)
    pbar.set_postfix({"local idx": f"{idx:04d}", "samples": f"{localSampleNumber:05d}", "ring": f"{rings:02d}"})
    # sub-mesh
    subRefV = refV[vSelected]
    subMesh = o3d.geometry.TriangleMesh()
    subMesh.vertices = o3d.utility.Vector3dVector(subRefV)
    bbox = subMesh.get_axis_aligned_bounding_box()
    box_center = 0.5 * (bbox.max_bound + bbox.min_bound)
    box_diagonal = np.linalg.norm(bbox.max_bound - bbox.min_bound)

    scaledRefV = (refV - box_center) / box_diagonal
    refMesh.vertices = o3d.utility.Vector3dVector(scaledRefV)

    num_samp_near_surf = int(num_samp_near_surf_ratio * localSampleNumber) // 2 * 2
    surfSamples = SampleFromSurface(scaledRefV, refF, faceArea, fSelected, num_samp_near_surf, noiseVar)
    randSamples = SampleFromCube(localSampleNumber - num_samp_near_surf, 1)

    totalSamples = np.vstack((surfSamples, randSamples))
    totalSamples = totalSamples.astype(np.float32)

    # get SDF values
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(refMesh))
    refD, refIntersectionInfo = compute_signed_distance_and_closest_goemetry(totalSamples, scene)

    # you can use refIntersectionInfo to calculate nearest points, colors ...
    refIntersectionTris = refF[refIntersectionInfo['primitive_ids'].numpy()]
    refIntersectionNormals = refFN[refIntersectionInfo['primitive_ids'].numpy()]
    # (SampleNumber, 3, 2)
    refIntersectionUVs = refIntersectionInfo['primitive_uvs'].numpy()
    refIntersectionTriUVs = refUV[refIntersectionInfo['primitive_ids'].numpy(), :, :]
    refIntersectionTextureUVs = refIntersectionTriUVs[:, 0, :].reshape(localSampleNumber, 2) * (
            1 - refIntersectionUVs[:, 0].reshape(-1, 1) - refIntersectionUVs[:, 1].reshape(-1, 1)) + \
                                refIntersectionTriUVs[:, 1, :].reshape(localSampleNumber,
                                                                       2) * refIntersectionUVs[:,
                                                                            0].reshape(-1, 1) + \
                                refIntersectionTriUVs[:, 2, :].reshape(localSampleNumber,
                                                                       2) * refIntersectionUVs[:,
                                                                            1].reshape(-1, 1)
    # v0 + (v1 - v0) * u + (v2 - v0) * v
    refIntersectionPoints = refIntersectionInfo['points'].numpy()
    refIntersectionColors = compute_rgb_at_uv(refIntersectionTextureUVs, refTexture, True)

    # samplePointXYZ 3 NerghborhoodPointXYZ 3 RGB 3 normal 3 SDF 1 localPatchIdx 1
    sdfSamplesWithColor = np.hstack(
        (totalSamples, refIntersectionPoints, refIntersectionColors, refIntersectionNormals, refD.reshape(-1, 1),
         idx * np.ones((localSampleNumber, 1))))
    sampledValues = np.append(sampledValues, sdfSamplesWithColor)
    # local patch bbox center and diagonal
    sampledBbox[idx, :] = np.hstack((box_center, box_diagonal))

refSDFInfo = sampledValues.reshape(-1, 14)

ms = pymeshlab.MeshSet()
ms.load_new_mesh(dis_obj_path)
ms.meshing_remove_unreferenced_vertices()
ms.meshing_remove_duplicate_vertices()
ms.meshing_remove_duplicate_faces()
ms.meshing_remove_null_faces()

disV = ms.current_mesh().vertex_matrix()
disF = ms.current_mesh().face_matrix()
disVN = ms.current_mesh().vertex_normal_matrix()
disUV = ms.current_mesh().wedge_tex_coord_matrix().reshape(disF.shape[0], 3, 2)
disFN = ms.current_mesh().face_normal_matrix()

disTexture = np.array(Image.open(dis_tex_path))
disMesh = o3d.geometry.TriangleMesh()
disMesh.vertices = o3d.utility.Vector3dVector(disV)
disMesh.triangles = o3d.utility.Vector3iVector(disF)
disMesh.triangle_normals = o3d.utility.Vector3dVector(disFN)
disMesh.triangle_uvs = o3d.utility.Vector2dVector(disUV.reshape(disF.shape[0] * 3, 2))

sampledPositions = refSDFInfo[:, :3]
indicies, local_numbers = np.unique(refSDFInfo[:, 13], return_counts=True)
sampledValues = np.array([])

pbar = tqdm(zip(indicies, local_numbers), desc=f"Sampling distortion {os.path.basename(dis_obj_path).split('.')[0]}")
for idx, localSampleNumber in pbar:
    pbar.set_postfix({
        "file": dis_obj_path.split('/')[-1],
        "idx": f"{int(idx):03d}",
        "samples": f"{localSampleNumber:04d}"
    })
    sampledPosition = sampledPositions[refSDFInfo[:, 13] == idx].astype(np.float32)
    bbox = sampledBbox[int(idx)]
    scaledDisV = (disV - bbox[:3]) / bbox[3]
    disMesh.vertices = o3d.utility.Vector3dVector(scaledDisV)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(disMesh))
    disD, disIntersectionInfo = compute_signed_distance_and_closest_goemetry(sampledPosition, scene)

    # you can use refIntersectionInfo to calculate nearest points, colors ...
    disIntersectionTris = disF[disIntersectionInfo['primitive_ids'].numpy()]
    disIntersectionNormals = disFN[disIntersectionInfo['primitive_ids'].numpy()]
    # (SampleNumber, 3, 2)
    disIntersectionUVs = disIntersectionInfo['primitive_uvs'].numpy()
    disIntersectionTriUVs = disUV[disIntersectionInfo['primitive_ids'].numpy(), :, :]
    disIntersectionTextureUVs = disIntersectionTriUVs[:, 0, :].reshape(localSampleNumber, 2) * (
            1 - disIntersectionUVs[:, 0].reshape(-1, 1) - disIntersectionUVs[:, 1].reshape(-1, 1)) + \
                                disIntersectionTriUVs[:, 1, :].reshape(localSampleNumber,
                                                                       2) * disIntersectionUVs[:,
                                                                            0].reshape(-1, 1) + \
                                disIntersectionTriUVs[:, 2, :].reshape(localSampleNumber,
                                                                       2) * disIntersectionUVs[:,
                                                                            1].reshape(-1, 1)
    # v0 + (v1 - v0) * u + (v2 - v0) * v
    disIntersectionPoints = disIntersectionInfo['points'].numpy()
    disIntersectionColors = compute_rgb_at_uv(disIntersectionTextureUVs, disTexture, True)

    # samplePointXYZ NerghborhoodPointXYZ RGB normal SDF
    sdfSamplesWithColor = np.hstack(
        (sampledPosition, disIntersectionPoints, disIntersectionColors, disIntersectionNormals, disD.reshape(-1, 1),
         idx * np.ones((localSampleNumber, 1))))
    sampledValues = np.append(sampledValues, sdfSamplesWithColor)
disSDFInfo = sampledValues.reshape(-1, 14)

indicies, local_numbers = np.unique(refSDFInfo[:, 13], return_counts=True)
maxIndicesNumber = indicies.shape[0]
# use all the local patches
selectedIndiciesNumber = maxIndicesNumber
localFeatures = np.zeros((selectedIndiciesNumber, numberFeatures))
selectedIndicies = np.random.choice(maxIndicesNumber, size=selectedIndiciesNumber, replace=False)
selectedIndicies = np.sort(selectedIndicies)
for ii, idx, localSampleNumber in zip(range(selectedIndiciesNumber), indicies[selectedIndicies],
                                      local_numbers[selectedIndicies]):
    localRefSDFInfo = refSDFInfo[refSDFInfo[:, 13] == idx]
    localDisSDFInfo = disSDFInfo[disSDFInfo[:, 13] == idx]
    localFeatures[ii, :] = calculateSDFFeatures(localRefSDFInfo, localDisSDFInfo, tangentThreLevel, normalThreLevel,
                                                tangentPlaneLevel, radius_level * 0.1)
final_scores = np.mean(localFeatures, axis=0)
print(f"FMQM Score: [{final_scores[0]}, {final_scores[1]}, {final_scores[2]}, {final_scores[3]}, {final_scores[4]}]")