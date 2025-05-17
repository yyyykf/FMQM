
---
# Implementation of the Paper

**Textured Mesh Quality Assessment using Geometry and Color Field Similarity**

---

## 1. Single Mesh Quality Evaluation

### 1.1 Command Line Usage

```bash
python fmqm_single_mesh_eval.py \
    --ref_obj_path /path/to/reference.obj \
    --ref_tex_path /path/to/reference_texture.png \
    --dis_obj_path /path/to/distorted.obj \
    --dis_tex_path /path/to/distorted_texture.png
```

### 1.2 Required Parameters

| Parameter        | Required | Description                          |
| ---------------- | -------- | ------------------------------------ |
| `--ref_obj_path` | Yes      | Path to the reference mesh (.obj)    |
| `--ref_tex_path` | Yes      | Path to the reference texture (.png) |
| `--dis_obj_path` | Yes      | Path to the distorted mesh (.obj)    |
| `--dis_tex_path` | Yes      | Path to the distorted texture (.png) |

### 1.3 Hardcoded Hyperparameters

```python
# Sampling configuration
noise_level = 5
point_number = 200000
lp_number = 250
sample_mode = "fps"

# Feature extraction thresholds
tangentThreLevel = 5
normalThreLevel = 5
tangentPlaneLevel = 10
radius_level = 1.25
```

### 1.4 Output

Prints a 5-dimensional FMQM score to the console:

```
FMQM Score: [f1, f2, f3, f4, f5]
```

---

## 2. Dataset-Level Evaluation on TSMD, SJTU-MQA, YANA

### 2.1 Step 1: Generate Path Configuration Files

Run the following script to generate CSV files listing reference and distorted mesh paths:

```bash
python generate_path.py \
  --tsmd /path/to/TSMD/dataset \
  --sjtumqa /path/to/SJTU-MQA/dataset \
  --yana /path/to/YANA/dataset
```

This will generate the following files in the `datasetInfo/` folder:

```
datasetInfo/
├── pathsTSMD.csv
├── pathsSJTUMQA.csv
└── pathsYANA.csv
```

Each CSV includes: `modelRefName, modelDisName, ref_obj_path, ref_tex_path, dis_obj_path, dis_tex_path`.

---

### 2.2 Step 2: Sample Geometry and Color Fields (SDF + NCF)

#### 2.2.1 Basic Usage

Run the default configuration aligned with the paper:

```bash
python fmqm_sample_dataset.py --dataset [tsmd|sjtumqa|yana]
```

#### 2.2.2 Full Parameter Options

```bash
python fmqm_sample_dataset.py \
    --dataset [tsmd|sjtumqa|yana] \
    --noise_level 5 \
    --point_number 200000 \
    --lp_number 250 \
    --sample_mode [fps|random]
```

#### 2.2.3 Parameter Descriptions

| Parameter        | Type   | Default | Description                              |
| ---------------- | ------ | ------- | ---------------------------------------- |
| `--dataset`      | string | —       | Dataset name (`tsmd`, `sjtumqa`, `yana`) |
| `--noise_level`  | float  | 5       | Noise variance = (value × 0.01)²         |
| `--point_number` | int    | 200000  | Number of sample points per mesh         |
| `--lp_number`    | int    | 250     | Number of local patches                  |
| `--sample_mode`  | string | fps     | Sampling mode: `fps` or `random`         |

#### 2.2.4 Output Directory Structure

```
sampleResults/
└── {dataset}/
    ├── SDFU{noise}_{lp_number}_{points}/
    │   ├── reference/
    │   │   ├── {model_name}.mat
    │   │   └── {model_name}_bbox.mat
    │   └── distorted/
    │       └── {model_name}.mat
    └── topology/
        └── topology_{lp_number}_{sample_mode}/
            └── {model_name}.npz
```

#### 2.2.5 Sample File Format

**Reference Samples (`reference/*.mat`)**

```python
{
    'SamplePointXYZ':       [N, 3],
    'NearestPointXYZ':      [N, 3],
    'NCF':                  [N, 3],
    'NearestPointNormal':   [N, 3],
    'SDF':                  [N, 1],
    'LocalPatchIdx':        [N, 1]
}
```

**Bounding Box (`reference/*_bbox.mat`)**

```python
{
    'box_center':   [lp_number, 3],
    'box_diagonal': [lp_number, 1]
}
```

**Distorted Samples (`distorted/*.mat`)**

* Same structure as reference samples.

**Topology Data (`topology_{lp_number}_{sample_mode}/*.npz`)**

```python
{
    "local_0": { 'idx': int, 'v': [N], 'f': [M], 'rings': int },
    ...
}
```

*Note: Can be reused across sampling runs if `lp_number` and `sample_mode` are constant.*

---

### 2.3 Step 3: Compute FMQM Scores

#### 2.3.1 Command Line

```bash
python fmqm_calculate_scores.py \
    --dataset=[tsmd|sjtumqa|yana] \
    --sample_dir_name=SDFU5.00_250_200000
```

#### 2.3.2 Parameters

| Parameter           | Type   | Description                                        |
| ------------------- | ------ | -------------------------------------------------- |
| `--dataset`         | string | Dataset name (`tsmd`, `sjtumqa`, `yana`)           |
| `--sample_dir_name` | string | Sampling folder name (e.g., `SDFU5.00_250_200000`) |

#### 2.3.3 Hardcoded Hyperparameters

```python
tangentThreLevel = 5
normalThreLevel = 5
radius_level = 1.25
tangentPlaneLevel = 10
numberFeatures = 5  # 0: geoSSIM, 1: geoGradientSSIM, 2: colorSSIM, 3: colorGradientSSIM, 4: FMQM
```

#### 2.3.4 Output

Results saved in:

```
sampleResults/
└── {dataset}/
    ├── {sample_dir_name}/
    └── {dataset}_{sample_dir_name}.mat
```

* Each row in the mat contains the 5-dimensional quality scores for a distorted mesh sample.

---

## 3. Visualization

### 3.1 Visualize Mesh

```bash
python visualize_mesh.py \
    --dataset [tsmd|sjtumqa|yana] \
    --model_name vaso
```

| Argument       | Description                                |
| -------------- | ------------------------------------------ |
| `--dataset`    | Dataset name: `tsmd`, `sjtumqa`, or `yana` |
| `--model_name` | Model name to visualize                    |

---

### 3.2 Visualize Local Patch SDF & NCF

```bash
python visualize_SDF_NCF.py \
    --dataset [tsmd|sjtumqa|yana] \
    --model_name vaso
```

| Argument         | Description                             |
| ---------------- | --------------------------------------- |
| `--dataset`      | Dataset name                            |
| `--noise_level`  | Noise level: variance = (value × 0.01)² |
| `--point_number` | Number of points to visualize           |
| `--lp_number`    | Number of local patches                 |
| `--model_name`   | Target model for visualization          |

---

## 4. Author & Contact

**Kaifa Yang**  
Cooperative Medianet Innovation Center  
Shanghai Jiao Tong University  
✉️ Email: <sekiroyyy@sjtu.edu.cn>

---