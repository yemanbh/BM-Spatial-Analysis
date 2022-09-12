# Spatial Mapping of Bone Marrow Microevironment using Deep Learning

This repository contains scripts developed for spatial mapping of the hematopoietic ecosytem of bone marrow and 
the mosaic bone marrow tissue.

## Implementation
- Fully implemented in Python


## Installation
**Download code to your local computer**
```
git clone https://github.com/YemanBrhane/BM-Spatial-Analysis.git
```
**For local virtual environment**
- scikit-image==0.16.2
- numpy==1.16.4
- pandas==1.0.5
- matplotlib==3.2.2
- seaborn==0.10.1
- scipy==1.4.1
- opencv-python==4.2.0.34
- scikit-learn==0.21.2
- statannot==0.2.3

or use the requirement.txt

**Using Docker or Sigularity**

Make sure you are in the Docker folder and type this in terminal:
```
docker build -t ImageName:TAG .  
```
(example: docker build -t bmnet:01 .)


## Input data directory structure
Dataset structure.
```

Data(main folder)
		+++ Batch1(folder1)
			- file name 1
			- file name 2
			- file name 3
			.
			.
			.
		+++ Batch2(folder2)
			- file name 1
			- file name 2
			- file name 3
			.
			.
			.
```

## AwareNet pipeline: Cell detection and classification

The details of AwareNet pipeline can be found in <a href="https://github.com/YemanBrhane/AwareNet"> AwareNet </a>.

![AwareNet output sample image](Images/cd_cc.png)
```
Figure 1| Sample image showing cell detection and classifcation results of AwarNet pipeline. BLIMP1+ myeloma cells (Magente); CD8+ T cells (White); CD4+ T cells (Yellow)
```

## MoSaicNet pipeline: Tissue Compartments segmentation pipeline
![MoSaicNet output sample image](Images/MoSaicNet.png)
```
Figure 2| Sample image showing the differnt tissue compartments automatically segmented by MoSaicNet pipeline. 
```

# How to run the pipeline
A code and template for running the code locally and on remote computer (using Docker or Singularity is provided)

## Locally
Once you created a python environment using the above required libraries and update the required variables in Configuration/my_configuration file:
```
python main.py
```
## Remote computer or cluster
A template is provided in submit-slurm-test.slurm for remote using singularity. The template could be easily adopted to docker engine.
```
sbatch --array=0-n submit-slurm-test.slurm
```

## Citation

Will be updated.