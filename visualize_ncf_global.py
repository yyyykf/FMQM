import pandas as pd
import open3d as o3d
import pymeshlab
import os
from PIL import Image
from utils.sample_utils import *
from utils.visualization_utils import *

script_root = os.path.dirname(os.path.abspath(__file__))

selected_models = ["mitch_fr00001", "the_great_drawing_room", "wareBowl", "plant", "bread_BakedUV", "shark_bakedUV"]
model_configs = {
    "mitch_fr00001": {
        "noise_level": [0, 0.3, 0.6, 0.9, 1.2],
        "point_size": 1.5,
        "targetSampleNumber": 5000000
    },
    "the_great_drawing_room": {
        "noise_level": [0, 0.12, 0.24, 0.36, 0.48],
        "point_size": 1.5,
        "targetSampleNumber": 20000000
    },
    "wareBowl": {
        "noise_level": [0, 0.45, 0.9, 1.35, 1.8],
        "point_size": 1.5,
        "targetSampleNumber": 5000000
    },
    "plant": {
        "noise_level": [0, 0.1, 0.2, 0.3, 0.4],
        "point_size": 1.5,
        "targetSampleNumber": 5000000
    },
    "bread_BakedUV": {
        "noise_level": [0, 0.3, 0.6, 0.9, 1.2],
        "point_size": 1.5,
        "targetSampleNumber": 5000000
    },
    "shark_bakedUV": {
        "noise_level": [0, 0.2, 0.4, 0.6, 0.8],
        "point_size": 1.5,
        "targetSampleNumber": 5000000
    }
}

for dataset in ["sjtumqa", "tsmd", "yana"]:
    datasetInfofile = os.path.join(script_root, "datasetInfo", f"paths{dataset.upper()}.csv")
    if not os.path.exists(datasetInfofile):
        print("Do not have mesh path info, please run generate_path.py")
    output_root = os.path.join(script_root, "sampleResults", dataset, f"pc_visual")
    os.makedirs(output_root, exist_ok=True)

    df = pd.read_csv(datasetInfofile)
    df = df.drop(columns=["modelDisName", "dis_obj_path", "dis_tex_path"])
    df = df.drop_duplicates()

    for index, row in df.iterrows():
        modelRefName = row["modelRefName"]
        ref_obj_path = row["ref_obj_path"]
        ref_tex_path = row["ref_tex_path"]
        if not (modelRefName in selected_models):
            continue
        tmp = o3d.io.read_triangle_mesh(ref_obj_path, True)
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
        refMesh.textures = tmp.textures
        refMesh.triangle_material_ids = o3d.utility.IntVector([0] * refF.shape[0])

        subRefV = refV
        subMesh = o3d.geometry.TriangleMesh()
        subMesh.vertices = o3d.utility.Vector3dVector(subRefV)

        bbox = subMesh.get_axis_aligned_bounding_box()
        box_center = 0.5 * (bbox.max_bound + bbox.min_bound)
        box_diagonal = np.linalg.norm(bbox.max_bound - bbox.min_bound)

        scaledRefV = (refV - box_center) / box_diagonal
        refMesh.vertices = o3d.utility.Vector3dVector(scaledRefV)

        localSampleNumber = model_configs[modelRefName]["targetSampleNumber"]
        fSelected = list(range(refF.shape[0]))
        for level, noise_level in enumerate(model_configs[modelRefName]["noise_level"]):
            noiseVar = (0.01 * noise_level) ** 2
            pc_save_root = os.path.join(output_root, modelRefName)
            os.makedirs(pc_save_root, exist_ok=True)
            pc_save_path = os.path.join(pc_save_root, f"{modelRefName}_var_{noise_level:.2f}.ply")
            # if os.path.exists(pc_save_path):
            #     continue
            print(f"sampling {modelRefName} noise_var={noise_level:.2f}")
            camera_info_json = os.path.join(script_root, "sampleResults", "visual_info", f"{modelRefName}.json")
            output_png_path = os.path.join(script_root, "sampleResults", "visual_info", f"{modelRefName}_R{level}.png")
            # if os.path.exists(output_png_path):
            #     continue
            if not (noise_level == 0):
                surfSamples = SampleFromSurface(scaledRefV, refF, faceArea, fSelected, localSampleNumber, noiseVar)

                totalSamples = np.vstack((surfSamples))
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

                # samplePointXYZ surfacePointXYZ RGB normalSDF
                sdfSamplesWithColor = np.hstack(
                    (totalSamples, refIntersectionPoints, refIntersectionColors, refIntersectionNormals, refD.reshape(-1, 1)))

                pc_sample = o3d.geometry.PointCloud()
                pc_sample.points = o3d.utility.Vector3dVector(totalSamples)
                pc_sample.colors = o3d.utility.Vector3dVector(refIntersectionColors / 255.0)

            else:
                pc_sample = SampleToPC(scaledRefV, refF, refFN, refUV, refTexture, faceArea, fSelected, localSampleNumber)

            o3d.visualization.draw_geometries([refMesh], width=2048, height=2048)
            o3d.visualization.draw_geometries([pc_sample], width=2048, height=2048)
            # save_pc_snap([pc_sample], model_configs[modelRefName]["point_size"], camera_info_json, output_png_path)
