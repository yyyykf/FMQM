import pandas as pd
import pymeshlab
import os
from PIL import Image
from utils.sample_utils import *
import argparse

parser = argparse.ArgumentParser(description="FMQM sample datasets")
parser.add_argument("--dataset", type=str, choices=["tsmd", "sjtumqa", "yana"], default="tsmd",
                    help="Target dataset name: choose from tsmd, sjtumqa, yana")
parser.add_argument("--model_name", type=str, default="vaso", help="Model name to visualize")
args = parser.parse_args()

dataset = args.dataset
script_root = os.path.dirname(os.path.abspath(__file__))
datasetInfofile = os.path.join(script_root, "datasetInfo", f"paths{dataset.upper()}.csv")
if not os.path.exists(datasetInfofile):
    print("Do not have mesh path info, please run generate_path.py")

df = pd.read_csv(datasetInfofile)
df = df.drop(columns=["modelDisName", "dis_obj_path", "dis_tex_path"])
df = df.drop_duplicates()

for index, row in df.iterrows():
    modelRefName = row["modelRefName"]
    ref_obj_path = row["ref_obj_path"]
    ref_tex_path = row["ref_tex_path"]
    if not (modelRefName in [args.model_name]):
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

    refTexture = np.array(Image.open(ref_tex_path))
    refMesh = o3d.geometry.TriangleMesh()
    refMesh.vertices = o3d.utility.Vector3dVector(refV)
    refMesh.triangles = o3d.utility.Vector3iVector(refF)
    refMesh.triangle_normals = o3d.utility.Vector3dVector(refFN)
    refMesh.triangle_uvs = o3d.utility.Vector2dVector(refUV.reshape(refF.shape[0] * 3, 2))
    refMesh.textures = tmp.textures
    refMesh.triangle_material_ids = o3d.utility.IntVector([0] * refF.shape[0])

    o3d.visualization.draw_geometries([refMesh], width=800, height=800, left=50, top=50)
    # use the below lines to create a window and select views
    # vis = o3d.visualization.Visualizer()
    # vis.create_window(width=800, height=800, left=50, top=50)
    # render_option = vis.get_render_option()
    # render_option.point_size = 2.5
    # vis.add_geometry(refMesh)
    # vis.poll_events()
    # vis.update_renderer()