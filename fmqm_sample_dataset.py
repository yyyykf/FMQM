import pandas as pd
import open3d as o3d
import pymeshlab
import os
from PIL import Image
from utils.sample_utils import *
import argparse
from scipy.io import savemat, loadmat
from tqdm import tqdm

# conda activate sdfqa
# python fmqm_sample_dataset.py --dataset=tsmd
# python fmqm_sample_dataset.py --dataset=sjtumqa
# python fmqm_sample_dataset.py --dataset=yana

parser = argparse.ArgumentParser(description="FMQM sample datasets")
parser.add_argument("--dataset", type=str, choices=["tsmd", "sjtumqa", "yana"], default="tsmd", 
                    help="Target dataset name: choose from tsmd, sjtumqa, yana")
parser.add_argument("--noise_level", type=float, default=5, help="Noise level: noise variance = (noise_level * 0.01)^2")
parser.add_argument("--point_number", type=int, default=200000, help="Number of total points")
parser.add_argument("--lp_number", type=int, default=250, help="Number of local patches")
parser.add_argument("--sample_mode", type=str, choices=["fps", "random"], default="fps", 
                    help="Sampling strategy: 'fps' (farthest point sampling) or 'random'")
args = parser.parse_args()

script_root = os.path.dirname(os.path.abspath(__file__))
dataset = args.dataset
datasetInfofile = os.path.join(script_root, "datasetInfo", f"paths{dataset.upper()}.csv")
if not os.path.exists(datasetInfofile):
    print("Do not have mesh path info, please run generate_path.py")

result_root = os.path.join(script_root, "sampleResults", dataset)
os.makedirs(result_root, exist_ok=True)

targetSampleNumber = args.point_number
numberFPS = args.lp_number
localMinRatio = 1 / numberFPS
localMaxRatio = localMinRatio * 1.1
maxRings = 200

topology_root = os.path.join(result_root, "topology", "topology_{:03d}".format(numberFPS))
os.makedirs(topology_root, exist_ok=True)

sampleDirection = "uniform"
noiseVar = (0.01 * args.noise_level)**2
sample_root = os.path.join(result_root, "SDFU{:.2f}_{:03d}_{:06d}".format(args.noise_level, args.lp_number, args.point_number))
os.makedirs(sample_root, exist_ok=True) 
num_samp_near_surf_ratio = 1

df = pd.read_csv(datasetInfofile)

for index, row in df.iterrows():
    modelRefName = row["modelRefName"]
    modelDisName = row["modelDisName"]
    ref_obj_path = row["ref_obj_path"]
    ref_tex_path = row["ref_tex_path"]
    dis_obj_path = row["dis_obj_path"]
    dis_tex_path = row["dis_tex_path"]
    sdfRefDir = os.path.join(sample_root, "reference")
    os.makedirs(sdfRefDir, exist_ok=True)
    sdfDisDir = os.path.join(sample_root, "distorted")
    os.makedirs(sdfDisDir, exist_ok=True)
    sdfRefMatName = os.path.join(sdfRefDir, f"{modelRefName}.mat")
    sdfRefBBoxMatName = os.path.join(sdfRefDir, f"{modelRefName}_bbox.mat")
    sdfDisMatName = os.path.join(sdfDisDir, f"{modelDisName}.mat")
    
    if not os.path.exists(sdfRefMatName):
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

        print('load mesh vertices: {}'.format(refV.shape))
        print('load mesh faces: {}'.format(refF.shape))
        print('load mesh vertex normal: {}'.format(refVN.shape))
        print('load mesh face normal: {}'.format(refFN.shape))
        print('load mesh uv coordinates: {}'.format(refUV.shape))
        
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
        fps_idx = farthest_point_sample(refV, total_fps)
        local_numbers = fps_idx.shape[0]

        sampledBbox = np.zeros((local_numbers, 4))
        sampledValues = np.array([])
        
        topoloy_file = os.path.join(topology_root, "{}.npz".format(modelRefName))
        if os.path.exists(topoloy_file):
            topology_data = np.load(topoloy_file, allow_pickle=True)
            topoloy_file_exist = True
        else:
            topology_data = []
            topoloy_file_exist = False
        # for idx, vId in zip(range(local_numbers), fps_idx):
        pbar = tqdm(zip(range(local_numbers), fps_idx), desc=f"Sampling reference {index:04d}")
        for idx, vId in pbar:
            if not topoloy_file_exist:
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
                topology_data.append(d)
            else:
                d = topology_data[f"local_{idx}"].item()
                
                vSelected = d['v']
                fSelected = d['f']
                rings = d['rings']

            if rings == maxRings + 1:
                continue
            
            localSampleNumber = int(targetSampleNumber * np.sum(faceArea[fSelected] / np.sum(faceArea)))
            if localSampleNumber > targetSampleNumber * localMaxRatio:
                localSampleNumber = int(targetSampleNumber * localMaxRatio)
            # print("{:03d}: local idx: {:04d}, sample number: {:05d}, stop at ring: {:02d}".format(index, idx, localSampleNumber, rings))
            # tqdm.write(f"{index:03d}: local idx: {idx:04d}, sample number: {localSampleNumber:05d}, stop at ring: {rings:02d}")
            pbar.set_postfix({ "local idx": f"{idx:04d}", "samples": f"{localSampleNumber:05d}", "ring": f"{rings:02d}" })
            # sub-mesh
            subRefV = refV[vSelected]
            subMesh = o3d.geometry.TriangleMesh()
            subMesh.vertices = o3d.utility.Vector3dVector(subRefV)
            bbox = subMesh.get_axis_aligned_bounding_box()
            box_center = 0.5 * (bbox.max_bound + bbox.min_bound)
            box_diagonal = np.linalg.norm(bbox.max_bound - bbox.min_bound)

            scaledRefV = (refV - box_center) / box_diagonal
            refMesh.vertices = o3d.utility.Vector3dVector(scaledRefV)

            # Visualize the mesh and sampled points
            num_samp_near_surf = int(num_samp_near_surf_ratio * localSampleNumber)
            surfSamples = SampleFromSurface(scaledRefV, refF, faceArea, fSelected, num_samp_near_surf, noiseVar)
            randSamples = SampleFromBoundingCube(localSampleNumber - num_samp_near_surf, 1)

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

            # samplePointXYZ NerghborhoodPointXYZ RGB normal SDF 
            sdfSamplesWithColor = np.hstack(
                (totalSamples, refIntersectionPoints, refIntersectionColors, refIntersectionNormals, refD.reshape(-1, 1), idx * np.ones((localSampleNumber, 1))))
            sampledValues = np.append(sampledValues, sdfSamplesWithColor)
            sampledBbox[idx, :] = np.hstack((box_center, box_diagonal))

        if not topoloy_file_exist:
            np.savez(topoloy_file, **{f"local_{i}": d for i, d in enumerate(topology_data)})
        
        sampledValues = sampledValues.reshape(-1, 14)
        if not os.path.exists(sdfRefMatName):
            savemat(sdfRefMatName, {"refSDFInfo": sampledValues})
            savemat(sdfRefBBoxMatName, {"bbox": sampledBbox})

    if not os.path.exists(sdfDisMatName):
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

        sampledInfo = loadmat(sdfRefMatName)["refSDFInfo"]
        sampledBboxs = loadmat(sdfRefBBoxMatName)["bbox"]
        sampledPositions = sampledInfo[:, :3]
        indicies, local_numbers = np.unique(sampledInfo[:, 13], return_counts=True)
        sampledValues = np.array([])

        # for idx, localSampleNumber in tqdm(zip(indicies, local_numbers), total=len(indicies), desc="Sampling"):
            # print("Sampling Dis {} of localidx: {:03d} and sample number: {:04d}".format(dis_obj_path, int(idx), localSampleNumber))
            # tqdm.write(f"Sampling Dis {dis_obj_path} - idx: {int(idx):03d}, samples: {localSampleNumber:04d}")
        pbar = tqdm(zip(indicies, local_numbers), desc=f"Sampling distortion {index:04d}")
        for idx, localSampleNumber in pbar:
            pbar.set_postfix({
                "file": dis_obj_path.split('/')[-1],
                "idx": f"{int(idx):03d}",
                "samples": f"{localSampleNumber:04d}"
            })
            sampledPosition = sampledPositions[sampledInfo[:, 13] == idx].astype(np.float32)
            bbox = sampledBboxs[int(idx)]
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
                (sampledPosition, disIntersectionPoints, disIntersectionColors, disIntersectionNormals, disD.reshape(-1, 1), idx * np.ones((localSampleNumber, 1))))
            sampledValues = np.append(sampledValues, sdfSamplesWithColor)
        sampledValues = sampledValues.reshape(-1, 14)
        savemat(sdfDisMatName, {"disSDFInfo": sampledValues})