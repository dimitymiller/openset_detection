# Uncertainty for Identifying Open-Set Errors in Visual Object Detection
This repository contains the training and evaluation code from the paper:

**Uncertainty for Identifying Open-Set Errors in Visual Object Detection**

*Dimity Miller, Niko Suenderhauf, Michael Milford, Feras Dayoub*

<!-- ![GMM-Det](images/GMMDet.png) -->
<p align="center">
  <img width="600" height="281" src=images/GMMDet.png>
</p>

If you use this repository, please cite:

```text
@article{miller2021uncertainty,
  title={Uncertainty for Identifying Open-Set Errors in Visual Object Detection},
  author={Miller, Dimity and S{\"u}nderhauf, Niko and Milford, Michael and Dayoub, Feras},
  journal={arXiv preprint arXiv:2104.01328},
  year={2021}
}
```

**Contact**

If you have any questions or comments, please contact [Dimity Miller](mailto:d24.miller@qut.edu.au).

## Installation

This code was developed with Python 3.7 on Ubuntu 20.04. It requires a GPU. 
 
### Installing via conda environment (recommended)
We have included the os_det.yml file for the conda environment we used during this project. To create your own conda environment you can run:

```bash
conda env create -f os_det.yml
```

You should then activate this environment before running any of the code:

```bash
conda activate os_det
```

### Otherwise (without using a conda environment)
Python requirements can be installed by:

```bash
pip install -r requirements.txt
```

## Datasets
Our experiments build off PASCAL VOC, COCO, and the iCubWorld Transformations datasets. These datasets should be available in the `datasets/data/` folder inside this repository. 

### PASCAL VOC and VOC-OS
Pascal VOC data can be downloaded from [here](http://host.robots.ox.ac.uk/pascal/VOC/). The VOC2007 training/validation data, VOC2007 annotated test data, and VOC2012 training/validation data should be downloaded. The data should be in the following format:
 
 <br>
 
    └── datasets
        └── data              
            ├── VOCdevkit
            |    ├── VOC2007
            |    |    ├── Annotations
            |    |    ├── ImageSets
            |    |    ├── JPEGImages
            |    |    └── ...                 
            |    └── VOC2012
            |         ├── Annotations
            |         ├── ImageSets
            |         ├── JPEGImages
            |         └── ...     
            └── ...

### Setting up VOC-OS and COCO-OS



## Pre-trained Models

## Evaluation

## Training 
