
import pandas as pd
import os
from utils.feature_utils import *
import argparse
from scipy.io import savemat, loadmat
from tqdm import tqdm

# conda activate sdfqa
# python fmqm_score_dataset.py --dataset=tsmd
# python fmqm_score_dataset.py --dataset=sjtumqa
# python fmqm_score_dataset.py --dataset=yana

parser = argparse.ArgumentParser(description="FMQM calculate scores")
parser.add_argument("--dataset", type=str, choices=["tsmd", "sjtumqa", "yana"], default="sjtumqa", 
                    help="Target dataset name: choose from tsmd, sjtumqa, yana")
parser.add_argument("--sample_dir_name", type=str, default="SDFU5.00_250_200000", help="sample dir name")
args = parser.parse_args()

# hyperparameters
tangentThreLevel = 5
normalThreLevel = 5
tangentPlaneLevel = 10
numberFeatures = 5
radius_level = 1.25

script_root = os.path.dirname(os.path.abspath(__file__))
dataset = args.dataset
datasetInfofile = os.path.join(script_root, "datasetInfo", f"paths{dataset.upper()}.csv")
if not os.path.exists(datasetInfofile):
    print("Do not have mesh path info, please run generate_path.py")

result_root = os.path.join(script_root, "sampleResults", dataset)
os.makedirs(result_root, exist_ok=True)

sample_root = os.path.join(result_root, args.sample_dir_name)

df = pd.read_csv(datasetInfofile)
numberObjs = df.shape[0]
globalFeatures = np.zeros((numberObjs, numberFeatures))

for index, row in df.iterrows():
    modelRefName = row["modelRefName"]
    modelDisName = row["modelDisName"]
    sdfRefDir = os.path.join(sample_root, "reference")
    sdfDisDir = os.path.join(sample_root, "distorted")
    sdfRefMatName = os.path.join(sdfRefDir, f"{modelRefName}.mat")
    sdfRefBBoxMatName = os.path.join(sdfRefDir, f"{modelRefName}_bbox.mat")
    sdfDisMatName = os.path.join(sdfDisDir, f"{modelDisName}.mat")
    
    for path in [sdfRefMatName, sdfRefBBoxMatName, sdfDisMatName]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

    refSDFInfo = loadmat(sdfRefMatName)["refSDFInfo"]
    bbox = loadmat(sdfRefBBoxMatName)["bbox"]
    disSDFInfo = loadmat(sdfDisMatName)["disSDFInfo"]

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
        localFeatures[ii, :] = calculateSDFFeatures(localRefSDFInfo, localDisSDFInfo, tangentThreLevel, normalThreLevel, tangentPlaneLevel, radius_level * 0.1)
    globalFeatures[index, :] = np.mean(localFeatures, axis=0)
    print(f"finish {index} / {numberObjs}")
    
matFile = os.path.join(result_root, f"{dataset}_{args.sample_dir_name}.csv")
savemat(matFile, {f'{dataset}': globalFeatures})
print("save to {}".format(matFile))