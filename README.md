# 🦐 Prawn Dataset Overview & Excel Description

This repository contains prawn keypoint datasets collected in **two different seasons (2024 and 2025)**, as well as an Excel file summarizing all annotated keypoints.

---

## 📂 Image Directories

### 1. `train_on_all/` — 2024 Season Dataset

This directory contains **all annotated prawn images collected during the 2024 season**.

### 2. `prawn_2025_circ_small_v1/` — Small 2025 Evaluation Set

This folder contains **a small number of annotated prawn images from the 2025 circular pond**.

This dataset is used as a separate evaluation set to compare prawn appearance and annotation patterns across seasons.


- `images/` — all prawn images from **2024**
- `labels/` — YOLO keypoint label files (`.txt`), one per image

Each label file contains the four anatomical keypoints used for the prawns:

| Index | Keypoint Name | Description        |
|-------|----------------|--------------------|
| 0     | Carapace       | Carapace start     |
| 1     | Eyes           | Eyes midpoint      |
| 2     | Rostrum        | Rostrum tip        |
| 3     | Tail           | Tail end           |

Coordinates are stored in normalized YOLO format (values in the range \[0, 1\]).


---

## 📊 Excel File — `keypoints_from_label_txt.xlsx`

This Excel file contains a **long-format table** where each row corresponds to **one keypoint of one prawn**.

Columns included in the Excel file:

| Column           | Description |
|------------------|-------------|
| `image_stem`     | Image filename without extension |
| `object_id`      | Index of the object within the label file |
| `class_id`       | YOLO class index (typically 0 for prawn) |
| `keypoint_index` | Which keypoint (0=carapace, 1=eyes, 2=rostrum, 3=tail) |
| `x_norm`         | Normalized x-coordinate (range \[0, 1\]) |
| `y_norm`         | Normalized y-coordinate (range \[0, 1\]) |
| `visibility`     | YOLO visibility flag (0/1/2) |

### 🔢 Normalized Coordinates
All keypoints in the label files are stored in **normalized YOLO format**.  
This means that the coordinates are given as values between **0 and 1**, relative to the image size:

- `x_norm = x_pixel / image_width`
- `y_norm = y_pixel / image_height`

For example, `x_norm = 0.5` means the keypoint is exactly at the horizontal center of the image.  
Normalization allows labels to remain consistent across images of different resolutions.


The Excel file provides a complete and clean summary of all available annotations from both seasons.


