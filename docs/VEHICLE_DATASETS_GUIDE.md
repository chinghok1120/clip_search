# Vehicle Datasets for CLIP Fine-Tuning

Practical guide to downloading and preparing vehicle datasets for fine-tuning CLIP.

---

## Dataset Options

### 1. CompCars Dataset (Recommended)

**Source:** CUHK Multimedia Lab  
**URL:** http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/  
**Size:** ~136 GB  
**Images:** 214,345 images  
**Classes:** 1,716 car models from 163 manufacturers  

**What's included:**
- **Web-nature images**: 136,726 images (frontal view, high quality)
- **Surveillance-nature images**: 50,000 images (surveillance view)
- **Part annotations**: 27,619 images with car part labels
- **Attributes**: Make, model, year (1950s-2016)
- **Bounding boxes**: For parts and whole vehicles

**Why good for you:**
- ✅ Has surveillance-view images (matches your use case!)
- ✅ Large and diverse
- ✅ Make/model labels included
- ✅ Free for research

**Download:**

```bash
# 1. Visit the dataset page
# http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/

# 2. Fill out the request form (takes 1-2 days for approval)
# You need to provide:
# - Name, email, organization
# - Intended use (research/academic)
# - Agreement to terms

# 3. After approval, you'll receive download links

# 4. Download data (multiple parts)
cd ~/datasets
mkdir compcars && cd compcars

# Download links will be provided via email (example structure):
wget http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/data/image.tar.gz
wget http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/data/label.tar.gz
wget http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/data/misc.tar.gz

# 5. Extract
tar -xzf image.tar.gz
tar -xzf label.tar.gz
tar -xzf misc.tar.gz
```

**Dataset structure:**
```
compcars/
├── data/
│   ├── image/              # All images
│   │   ├── 1/              # Make 1 (e.g., Acura)
│   │   │   ├── 1/          # Model (e.g., Acura Integra)
│   │   │   │   ├── 2007/  # Year
│   │   │   │   │   ├──*.jpg
│   │   ├── 2/              # Make 2 (e.g., Aston Martin)
│   │   ...
│   ├── label/              # Annotations
│   ├── train_test_split/  # Train/test splits
│   └── misc/               # Metadata
└── sv_data/                # Surveillance view data
    ├── image/
    └── label/
```

---

### 2. Stanford Cars Dataset (Easier to Get)

**Source:** Stanford AI Lab  
**URL:** http://ai.stanford.edu/~jkrause/cars/car_dataset.html  
**Size:** ~1.9 GB  
**Images:** 16,185 images  
**Classes:** 196 car classes (make/model/year combinations)  

**Download:**
```bash
cd ~/datasets
mkdir stanford_cars && cd stanford_cars

# Download images
wget http://ai.stanford.edu/~jkrause/car196/cars_train.tgz
wget http://ai.stanford.edu/~jkrause/car196/cars_test.tgz
wget http://ai.stanford.edu/~jkrause/car196/car_devkit.tgz

# Extract
tar -xzf cars_train.tgz
tar -xzf cars_test.tgz
tar -xzf car_devkit.tgz
```

**Structure:**
```
stanford_cars/
├── cars_train/
│   ├── 00001.jpg
│   ├── 00002.jpg
│   ...
├── cars_test/
│   ├── 00001.jpg
│   ...
└── devkit/
    ├── cars_meta.mat        # Car class names
    ├── cars_train_annos.mat # Train annotations
    └── cars_test_annos.mat  # Test annotations
```

**Pros:**
- ✅ Easy to download (no approval needed)
- ✅ Clean, high-quality images
- ✅ Good for initial experiments

**Cons:**
- ⚠️ Smaller (16K vs 214K images)
- ⚠️ No surveillance view
- ⚠️ Limited to 196 classes

---

### 3. VeRi Dataset (Vehicle Re-Identification)

**Source:** SJTU & TongJi University  
**URL:** https://vehiclereid.github.io/VeRi/  
**Size:** ~2 GB  
**Images:** 50,000 images of 776 vehicles  
**Cameras:** 20 surveillance cameras  

**What's included:**
- ✅ Surveillance camera views (perfect for your use case!)
- ✅ Vehicle color, type, brand labels
- ✅ Multiple views of same vehicle
- ✅ Tracking IDs

**Download:**
```bash
# 1. Visit: https://vehiclereid.github.io/VeRi/
# 2. Fill request form
# 3. Download after approval

cd ~/datasets
mkdir veri && cd veri

# Extract
unzip VeRi.zip
```

**Structure:**
```
veri/
├── image_train/
│   ├── 0001_c001_00000001_0.jpg  # vehicle_camera_frame_view.jpg
│   ...
├── image_test/
├── image_query/
└── train_label.xml  # Vehicle attributes
```

---

### 4. VehicleX (Synthetic + Real)

**Source:** NVLabs  
**URL:** https://github.com/yorkeyao/VehicleX  
**Size:** ~50 GB  
**Images:** 1,362 synthetic + real vehicle images  

**Unique feature:** Generated using Unity3D with realistic rendering

**Download:**
```bash
git clone https://github.com/yorkeyao/VehicleX.git
cd VehicleX
# Follow instructions to download data
```

---

### 5. Quick Start: Use ImageNet Cars Subset

**For immediate testing (no download wait):**

```bash
# Download ImageNet car classes (if you have ImageNet access)
# Or use COCO dataset (has "car" class)

# COCO 2017 (you already have this!)
# /home/chester/datasets/coco-2017/

# Extract cars from COCO
python extract_cars_from_coco.py \
  --coco ~/datasets/coco-2017 \
  --output ~/datasets/coco_cars
```

**COCO car categories:**
- Car, truck, bus, motorcycle
- ~35,000+ vehicle images
- Can start fine-tuning immediately

---

## Recommended Download Strategy

### Option A: Quick Start (Today)

1. **Use COCO dataset** (you already have it)
   - Extract vehicle images
   - Start fine-tuning experiments
   - ~35K images, good for proof-of-concept

2. **Download Stanford Cars** (no approval)
   - Small, fast download
   - Test fine-tuning pipeline
   - Validate approach

### Option B: Production Quality (1-2 weeks)

1. **Request CompCars** (submit form today)
   - Wait 1-2 days for approval
   - Best dataset for your use case
   - Has surveillance view

2. **Meanwhile: Use COCO + Stanford**
   - Start development immediately
   - Build fine-tuning pipeline
   - Ready when CompCars arrives

---

## Dataset Preparation for CLIP

Once you have a dataset, prepare it for CLIP fine-tuning:

### Script: `prepare_vehicle_dataset.py`

```python
#!/usr/bin/env python3
"""
Prepare vehicle dataset for CLIP fine-tuning
Converts dataset to: image_path, caption format
"""

import json
import csv
from pathlib import Path
from typing import List, Tuple

def prepare_compcars(data_dir: Path, output_file: Path):
    """Convert CompCars to CLIP format"""
    
    # Load make/model mappings
    with open(data_dir / 'misc/make_model_name.mat', 'rb') as f:
        # Parse mat file
        makes = load_makes(f)
        models = load_models(f)
    
    samples = []
    
    # Walk through images
    for img_path in (data_dir / 'image').rglob('*.jpg'):
        # Parse: data/image/make_id/model_id/year/image.jpg
        parts = img_path.parts
        make_id = int(parts[-4])
        model_id = int(parts[-3])
        year = parts[-2]
        
        make_name = makes[make_id]  # e.g., "Toyota"
        model_name = models[model_id]  # e.g., "Camry"
        
        # Create caption
        caption = f"{make_name} {model_name} {year}"
        
        samples.append((str(img_path), caption))
    
    # Save as CSV
    with open(output_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'caption'])
        writer.writerows(samples)
    
    print(f"✓ Prepared {len(samples)} samples")
    print(f"✓ Saved to: {output_file}")

def prepare_stanford_cars(data_dir: Path, output_file: Path):
    """Convert Stanford Cars to CLIP format"""
    
    import scipy.io
    
    # Load annotations
    annos = scipy.io.loadmat(data_dir / 'devkit/cars_train_annos.mat')
    meta = scipy.io.loadmat(data_dir / 'devkit/cars_meta.mat')
    
    class_names = [c[0] for c in meta['class_names'][0]]
    
    samples = []
    for anno in annos['annotations'][0]:
        img_name = anno[0][0]
        class_id = anno[-1][0][0] - 1  # MATLAB indexing
        
        img_path = data_dir / 'cars_train' / img_name
        caption = class_names[class_id]  # e.g., "Acura Integra Type R 2001"
        
        samples.append((str(img_path), caption))
    
    # Save
    with open(output_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'caption'])
        writer.writerows(samples)
    
    print(f"✓ Prepared {len(samples)} samples")

def prepare_coco_cars(coco_dir: Path, output_file: Path):
    """Extract cars from COCO dataset"""
    
    from pycocotools.coco import COCO
    
    # Load COCO annotations
    coco = COCO(coco_dir / 'annotations/instances_train2017.json')
    
    # Get car category IDs
    car_cats = coco.loadCats(coco.getCatIds(catNms=['car', 'truck', 'bus']))
    car_cat_ids = [cat['id'] for cat in car_cats]
    
    samples = []
    for cat_id in car_cat_ids:
        img_ids = coco.getImgIds(catIds=[cat_id])
        
        for img_id in img_ids:
            img_info = coco.loadImgs(img_id)[0]
            img_path = coco_dir / 'train2017' / img_info['file_name']
            
            # Get category name
            cat_name = coco.loadCats([cat_id])[0]['name']
            caption = f"{cat_name}"
            
            samples.append((str(img_path), caption))
    
    # Save
    with open(output_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'caption'])
        writer.writerows(samples)
    
    print(f"✓ Prepared {len(samples)} car images from COCO")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['compcars', 'stanford', 'coco'])
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_file = Path(args.output)
    
    if args.dataset == 'compcars':
        prepare_compcars(input_dir, output_file)
    elif args.dataset == 'stanford':
        prepare_stanford_cars(input_dir, output_file)
    elif args.dataset == 'coco':
        prepare_coco_cars(input_dir, output_file)
```

---

## Quick Start Guide

### Today: Use COCO (You Already Have It!)

```bash
cd ~/projects/clip_search

# 1. Extract cars from COCO
python scripts/prepare_vehicle_dataset.py \
  --dataset coco \
  --input ~/datasets/coco-2017 \
  --output ~/datasets/vehicle_training_data.csv

# 2. Start fine-tuning
python scripts/finetune_clip.py \
  --data ~/datasets/vehicle_training_data.csv \
  --model EVA02-B-16 \
  --output models/eva02_vehicles.pt

# 3. Test improved model
python scripts/search_images.py \
  --embeddings ./test_embeddings \
  --query "blue car" \
  --model-weights models/eva02_vehicles.pt
```

### This Week: Download Stanford Cars

```bash
# No approval needed - download now
cd ~/datasets
mkdir stanford_cars && cd stanford_cars

wget http://ai.stanford.edu/~jkrause/car196/cars_train.tgz
wget http://ai.stanford.edu/~jkrause/car196/car_devkit.tgz

tar -xzf cars_train.tgz
tar -xzf car_devkit.tgz

# Prepare for training
python prepare_vehicle_dataset.py \
  --dataset stanford \
  --input ~/datasets/stanford_cars \
  --output ~/datasets/stanford_cars_training.csv
```

### Next Week: Request CompCars

1. Visit: http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/
2. Fill form with your email
3. Wait 1-2 days for approval
4. Download and prepare

---

## Summary

| Dataset | Size | Images | Approval | Surveillance View | Best For |
|---------|------|--------|----------|-------------------|----------|
| **COCO** (have it!) | 25GB | 35K cars | ✅ No | ❌ No | Quick start today |
| **Stanford Cars** | 2GB | 16K | ✅ No | ❌ No | Easy testing |
| **VeRi** | 2GB | 50K | ⚠️ Yes (1-2 days) | ✅ Yes | Surveillance focus |
| **CompCars** | 136GB | 214K | ⚠️ Yes (1-2 days) | ✅ Yes | Production quality |

**Recommended path:**
1. ✅ Today: Extract cars from COCO, start experiments
2. ✅ This week: Download Stanford Cars, test pipeline
3. ⏳ Next week: Request CompCars for production fine-tuning

---

**Last Updated:** 2026-05-13  
**Contact:** chinghokuk@gmail.com
