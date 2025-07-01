# Microscopic Aquatic Organism Detection and Classification Pipeline

## Overview

This project implements an automated computer vision pipeline for detecting and classifying microscopic aquatic organisms from depth profile images. The system processes images through multiple stages including duplicate removal, depth profiling, flatfielding, object detection, and neural network-based classification.

### Target Organisms
The pipeline classifies the following organisms: 
- Cladocera
- Copepod
- Rotifer
- Junk

## Setup and Installation

Either install directly from the `environment.yml` file included with the project using conda or other similar service, or install packages manually into an environment by reading through the `environment_from_history.yml` (installing just those packaged will necessarily install the other prerequisites)

## Input Data Structure

The pipeline expects images organized in the following directory structure:

```
profiles/
├── Project_Name/
│   ├── YYYYMMDD/         
│   │   ├── time                                             # 'Day' or 'Night'
│   │   │   ├── location/                                    # Location subfolders
│   │   │   │   ├── Camera_YYYYMMDD_HHMMSSFFF_001.tiff
│   │   │   │   ├── Camera_YYYYMMDD_HHMMSSFFF_002.tiff
│   │   │   │   └── ...
│   │   ├── time2     
│   └── YYYYMMDD2/
└── Project_Name2/
```

### Image Naming Convention
Format: `Camera_YYYYMMDD_HHMMSSFFF_XXX.tiff`
- `Camera`: Camera name or identifier
- `YYYYMMDD`: Date
- `HHMMSSFFF`: Time with milliseconds
- `XXX`: Sequential number

## Usage

### Quick Start
```bash
python main.py -d ./profiles -o ./output -m ./model
```
#### Main Pipeline (`main.py`)
- `-d, --dataset_path`: Input directory path (default: `./profiles`)
- `-o, --root_output_path`: Output directory path (default: `.`)  
- `-m, --model_path`: Model directory path (default: `./model`)

### Execution Order

#### 1. Duplicate Image Removal `duplicate_image_removal.py`
- **Purpose**: Remove duplicate images using perceptual hashing
- **Method**: Difference hash (dHash) algorithm
- **Output**: Unique image set with duplicates identified/removed

#### 2. Depth Profiling `depth_profiler.py`
- **Purpose**: Convert raw images to depth-corrected profiles
- **Method**: Filename-based depth calculation and overlap correction
- **Parameters**: 
  - Depth multiplier: 100cm per unit
  - Image height: 4.3cm (2048 pixels)

#### 3. Flatfield Correction `flatfielding_profiles.py`
- **Purpose**: Correct for uneven illumination across images
- **Method**: Normalization using reference flatfield
- **Target**: 235 intensity normalization factor

#### 4. Object Detection `object_detection.py`
- **Purpose**: Identify and extract individual organisms
- **Method**: 
  - Binary thresholding (threshold: 190)
  - Connected component analysis
  - Size filtering (75-5000 pixels)
  - Shape filtering (eccentricity < 0.97)
- **Output**: Cropped object images (vignettes) and measurement data

#### 5. Classification `object_classification.py`
- **Purpose**: Classify detected objects into biological categories
- **Method**: Convolutional Neural Network (CNN)
- **Architecture**: 
  - Input: 50x50 grayscale images
  - 3 convolutional blocks with LRN
  - 3 fully connected layers with dropout
  - 4-class output (cladocera, copepod, junk, rotifer)

## Output Structure

```
output/
├── depth_profiles/                         # Depth-corrected images
├── flatfield_profiles/                     # Illumination-corrected images  
├── vignettes/                              # Extracted object crops
│   ├── measurements.csv                    # Object measurements
│   └── *.jpeg                              # Individual object images
└── classification/                         # Classification results
    ├── predictions.csv                     # Classification predictions
    └── confidence_scores.json
```

## Configuration

Key parameters can be modified in `constants.py`:

### Detection Parameters
- `THRESHOLD_VALUE`: Binary threshold (default: 190)
- `MIN_OBJECT_SIZE`: Minimum object size (default: 75 pixels)
- `MAX_OBJECT_SIZE`: Maximum object size (default: 5000 pixels)
- `MAX_ECCENTRICITY`: Maximum shape eccentricity (default: 0.97)

### Processing Parameters  
- `BATCH_SIZE`: Processing batch size (default: 10)