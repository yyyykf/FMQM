import os
import argparse
import pandas as pd

def check_files_exist(paths):
    return all(os.path.exists(p) for p in paths)

def process_tsmd(data_root, info_root):
    info_file = os.path.join(info_root, "basicInfoTSMD.csv")
    df = pd.read_csv(info_file)
    records = []
    for _, row in df.iterrows():
        modelNameRaw = row["modelNameRaw"]
        modelRefName = row["modelRefName"]
        modelDisName = row["modelDisName"]

        ref_obj_path = os.path.join(data_root, "reference", modelRefName, f"{modelNameRaw}.obj")
        ref_tex_path = os.path.join(data_root, "reference", modelRefName, f"{modelNameRaw}_0.png")
        dis_obj_path = os.path.join(data_root, "distorted", modelRefName, f"{modelDisName}.obj")
        dis_tex_path = os.path.join(data_root, "distorted", modelRefName, f"{modelDisName}_0.png")
        paths = [ref_obj_path, ref_tex_path, dis_obj_path, dis_tex_path]
        if not check_files_exist(paths):
            print(f"Missing file: {dis_obj_path}")
        records.append([modelRefName, modelDisName, *paths])
    return pd.DataFrame(records, columns=["modelRefName", "modelDisName", "ref_obj_path", "ref_tex_path", "dis_obj_path", "dis_tex_path"])

def process_sjtumqa(data_root, info_root):
    info_file = os.path.join(info_root, "basicInfoSJTUMQA.csv")
    df = pd.read_csv(info_file)
    records = []
    for _, row in df.iterrows():
        modelRefName = row["modelRefName"]
        modelDisName = row["modelDisName"]
        disType = modelDisName.split("_")[-2]
        disLevel = modelDisName.split("_")[-1]

        ref_obj_path = os.path.join(data_root, "reference_dataset", modelRefName, f"{modelRefName}.obj")
        ref_tex_path = os.path.join(data_root, "reference_dataset", modelRefName, f"{modelRefName}.jpg")

        if disType in ["gn", "qp", "simp", "simpNoTex", "qpqt"]:
            dis_obj_path = os.path.join(data_root, "distortion_dataset", disType, modelRefName, f"{modelDisName}.obj")
            dis_tex_path = os.path.join(data_root, "distortion_dataset", disType, modelRefName, f"{modelRefName}.jpg")
        else:  # ds, JPEG, qpqtJPEG
            dis_obj_path = os.path.join(data_root, "distortion_dataset", disType, modelRefName, disLevel, f"{modelRefName}.obj")
            dis_tex_path = os.path.join(data_root, "distortion_dataset", disType, modelRefName, disLevel, f"{modelRefName}.jpg")

        paths = [ref_obj_path, ref_tex_path, dis_obj_path, dis_tex_path]
        if not check_files_exist(paths):
            print(f"Missing file: {dis_obj_path}")
        records.append([modelRefName, modelDisName, *paths])
    return pd.DataFrame(records, columns=["modelRefName", "modelDisName", "ref_obj_path", "ref_tex_path", "dis_obj_path", "dis_tex_path"])

def process_yana(data_root, info_root):
    info_file = os.path.join(info_root, "basicInfoYANA.csv")
    df = pd.read_csv(info_file)
    records = []
    for _, row in df.iterrows():
        modelRefName = row["modelName"]
        modelDisName = row["disName"]
        folderName = row["folderName"]
        model_texture_name = row["model_texture_name"]
        resolution = row["resolution"]
        qa = row["qa"]
        LQpQt = row["LQpQt"]

        ref_dir = os.path.join(data_root, folderName, "source")
        dis_dir = os.path.join(data_root, folderName, "distortions", f"JPEG_resize{resolution}_quality{qa}")

        ref_obj_path = os.path.join(ref_dir, f"{modelRefName}.obj")
        ref_tex_path = os.path.join(ref_dir, f"{model_texture_name}.jpg")
        dis_obj_path = os.path.join(dis_dir, f"{modelRefName}_simp{LQpQt}.obj")
        dis_tex_path = os.path.join(dis_dir, f"{model_texture_name}.jpg")

        paths = [ref_obj_path, ref_tex_path, dis_obj_path, dis_tex_path]
        if not check_files_exist(paths):
            print(f"Missing file: {dis_obj_path}")
        records.append([modelRefName, modelDisName, *paths])
    return pd.DataFrame(records, columns=["modelRefName", "modelDisName", "ref_obj_path", "ref_tex_path", "dis_obj_path", "dis_tex_path"])


def main(tsmd_root, sjtumqa_root, yana_root):
    script_root = os.path.dirname(__file__)
    info_root = os.path.join(script_root, "datasetInfo")
    os.makedirs(info_root, exist_ok=True)

    tsmd_df = process_tsmd(tsmd_root, info_root)
    tsmd_df.to_csv(os.path.join(info_root, "pathsTSMD.csv"), index=False)

    sjtumqa_df = process_sjtumqa(sjtumqa_root, info_root)
    sjtumqa_df.to_csv(os.path.join(info_root, "pathsSJTUMQA.csv"), index=False)

    yana_df = process_yana(yana_root, info_root)
    yana_df.to_csv(os.path.join(info_root, "pathsYANA.csv"), index=False)

    print("All path info generated and saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsmd", required=True, help="Path to TSMD dataset root")
    parser.add_argument("--sjtumqa", required=True, help="Path to SJTU-MQA dataset root")
    parser.add_argument("--yana", required=True, help="Path to YANA dataset root")
    args = parser.parse_args()
    
    main(args.tsmd, args.sjtumqa, args.yana)