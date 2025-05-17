import pandas as pd
import os
import numpy as np
import open3d as o3d
import pymeshlab
from PIL import Image
import argparse
from utils.sample_utils import *
from utils.visualization_utils import *
from scipy.io import loadmat
from matplotlib.colors import LinearSegmentedColormap

parser = argparse.ArgumentParser(description="sample infomation")
parser.add_argument("--dataset", type=str, default="sjtumqa", help="target dataset name")
parser.add_argument("--noise_level", type=float, default=5, help="Noise level: noise variance = (noise_level * 0.01)^2")
parser.add_argument("--point_number", type=int, default=200000, help="Number of total points to visualize for a single patch")
parser.add_argument("--lp_number", type=int, default=50, help="Number of local patches")
parser.add_argument("--model_name", type=str, default="vaso", help="Model name to visualize")
args = parser.parse_args()

dataset = args.dataset
script_root = os.path.dirname(os.path.abspath(__file__))
datasetInfofile = os.path.join(script_root, "datasetInfo", f"paths{dataset.upper()}.csv")
if not os.path.exists(datasetInfofile):
    print("Do not have mesh path info, please run generate_path.py")
output_root = os.path.join(script_root, "sampleResults", dataset, f"pc_visual")
os.makedirs(output_root, exist_ok=True)

numberFPS = args.lp_number
noiseVar = (args.noise_level * 0.01) ** 2

sampleRoot = "visualUse"
localMinRatio = 1 / numberFPS
localMaxRatio = localMinRatio * 1.1
maxRings = 50

visualizeSampleNumber = args.point_number

df = pd.read_csv(datasetInfofile)
df = df.drop(columns=["modelDisName", "dis_obj_path", "dis_tex_path"])
df = df.drop_duplicates()

for index, row in df.iterrows():
    modelRefName = row["modelRefName"]
    ref_obj_path = row["ref_obj_path"]
    ref_tex_path = row["ref_tex_path"]

    if modelRefName == args.model_name:
        camera_info_json = os.path.join(script_root, "sampleResults", "visual_info", f"{modelRefName}.json")
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(ref_obj_path)
        ms.meshing_remove_unreferenced_vertices()
        ms.meshing_remove_duplicate_vertices()
        ms.meshing_remove_duplicate_faces()
        ms.meshing_remove_null_faces()

        tmp = o3d.io.read_triangle_mesh(ref_obj_path, True)

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
        refMesh.textures = tmp.textures
        refMesh.triangle_material_ids = o3d.utility.IntVector([0] * refF.shape[0])
        ref_wireframe = extract_mesh_wireframe(refMesh, [0,0,0])

        vRing = compute_vertex_ring(refV, refF)
        vertexToFace = compute_vertex_to_face(refV, refF)

        total_fps = numberFPS
        fps_idx = farthest_point_sample(refV, total_fps)
        fps_pc = o3d.geometry.PointCloud()
        fps_pc.points = o3d.utility.Vector3dVector(refV[fps_idx])
        fps_color = np.array([[1, 0, 0]]*total_fps)
        fps_pc.colors = o3d.utility.Vector3dVector(fps_color)
        local_numbers = fps_idx.shape[0]

        sampledBbox = np.zeros((local_numbers, 4))
        sampledValues = np.array([])

        for idx, vId in zip(range(local_numbers), fps_idx):
            # if not (idx == 5):
            #     continue
            # 1-ring
            vSelected = []
            fSelected = list(vertexToFace[vId])
            for f in fSelected:
                for v in refF[f]:
                    if v not in vSelected:
                        vSelected.append(v)
            localRatio = 0
            rings = 1
            while localRatio < localMinRatio:
                for v in vSelected:
                    for vNeighborFace in vertexToFace[v]:
                        if vNeighborFace not in fSelected:
                            fSelected.append(vNeighborFace)
                for f in fSelected:
                    for v in refF[f]:
                        if v not in vSelected:
                            vSelected.append(v)
                rings = rings + 1
                localRatio = np.sum(faceArea[fSelected]) / np.sum(faceArea)
                if (localRatio > localMaxRatio) or (rings > maxRings):
                    break
            if rings == maxRings + 1:
                continue
            vSelected.sort()
            fSelected.sort()

            localSampleNumber = visualizeSampleNumber
            # sub-mesh
            subRefV = refV[vSelected]
            subMesh = o3d.geometry.TriangleMesh()
            subMesh.vertices = o3d.utility.Vector3dVector(subRefV)

            subRefF = refF[fSelected]
            replace_dict = {value:index for index, value in enumerate(vSelected)}
            subRefF = np.vectorize(replace_dict.get)(subRefF)
            subRefFN = refFN[fSelected]
            subRefUV = refUV[fSelected, :, :]
            subMesh.triangles = o3d.utility.Vector3iVector(subRefF)
            subMesh.triangle_normals = o3d.utility.Vector3dVector(subRefFN)
            subMesh.triangle_uvs = o3d.utility.Vector2dVector(subRefUV.reshape(subRefF.shape[0] * 3, 2))
            tmp = o3d.io.read_triangle_mesh(ref_obj_path, True)
            subMesh.textures = tmp.textures
            subMesh.triangle_material_ids = o3d.utility.IntVector([0] * subRefF.shape[0])

            # visualize boundary egdes
            ref_local_wireframe = extract_mesh_wireframe(subMesh, [0, 0, 1])
            o3d.visualization.draw_geometries([subMesh, ref_local_wireframe, fps_pc, ref_wireframe], width=800, height=800, left=50, top=50)
            mesh_png = os.path.join(script_root, "sampleResults", "visual_info",
                                         f"{modelRefName}_mesh.png")
            # save_pc_snap([refMesh], 2.5, camera_info_json, mesh_png)
            wireframe_png = os.path.join(script_root, "sampleResults", "visual_info",
                                           f"{modelRefName}_wireframe.png")
            # save_pc_snap([fps_pc, ref_wireframe], 7.5, camera_info_json, wireframe_png)
            wireframe_lp_png = os.path.join(script_root, "sampleResults", "visual_info", f"{modelRefName}_wireframe_lp.png")
            # save_pc_snap([subMesh, ref_local_wireframe, fps_pc, ref_wireframe], 7.5, camera_info_json, wireframe_lp_png)
            # sample
            bbox = subMesh.get_axis_aligned_bounding_box()
            box_diagonal = np.linalg.norm(bbox.max_bound - bbox.min_bound)
            # Visualize the mesh and sampled points

            surfSamples = SampleFromSurface(refV, refF, faceArea, fSelected, localSampleNumber, noiseVar * box_diagonal**2)

            totalSamples = surfSamples.astype(np.float32)

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

            # samplePointXYZ NerghborhoodPointXYZ RGB normal SDF
            refsdfSamplesWithColor = np.hstack(
                (totalSamples, refIntersectionPoints, refIntersectionColors, refIntersectionNormals, refD.reshape(-1, 1), idx * np.ones((localSampleNumber, 1))))


            ref_pc_surface = o3d.geometry.PointCloud()
            ref_pc_surface.points = o3d.utility.Vector3dVector(refIntersectionPoints)
            ref_pc_surface.colors = o3d.utility.Vector3dVector(refIntersectionColors / 255)

            ref_pc_3d = o3d.geometry.PointCloud()
            ref_pc_3d.points = o3d.utility.Vector3dVector(totalSamples)
            ref_pc_3d.colors = o3d.utility.Vector3dVector(refIntersectionColors / 255)

            colors = [(1, 0, 0), (0, 1, 0)]
            cmap = LinearSegmentedColormap.from_list('CustomCmap', colors, N=256)
            tmpD = abs(refD) / max(refD)
            tmpColor = cmap(tmpD)[:, :3]
            ref_pc_sdf = o3d.geometry.PointCloud()
            ref_pc_sdf.points = o3d.utility.Vector3dVector(totalSamples)
            ref_pc_sdf.colors = o3d.utility.Vector3dVector(tmpColor)

            # sample_point_png = os.path.join(script_root, "sampleResults", "visual_info", f"{modelRefName}_wireframe_lp_points.png")
            # save_pc_snap([ref_pc_3d, ref_wireframe, ref_local_wireframe], 2.5, camera_info_json, sample_point_png)

            o3d.visualization.draw_geometries([ref_pc_surface, ref_wireframe, ref_local_wireframe], width=800, height=800, left=50, top=50)
            o3d.visualization.draw_geometries([ref_pc_3d, ref_wireframe, ref_local_wireframe], width=800, height=800, left=50, top=50)
            o3d.visualization.draw_geometries([ref_pc_sdf, ref_wireframe, ref_local_wireframe])