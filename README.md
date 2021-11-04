# Uncertainty for Identifying Open-Set Errors in Visual Object Detection
*** Currently under construction, due date in 2 weeks ***

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
  journal={IEEE Robotics and Automation Letters}, 
  year={2021},
  pages={1-1},
  doi={10.1109/LRA.2021.3123374}
}
```

**Contact**

If you have any questions or comments, please contact [Dimity Miller](mailto:d24.miller@qut.edu.au).

**Progress**
- [ ] Dataset setup (ETA: 1 day)
- [ ] Faster R-CNN Evaluations (ETA: 1 day)
- [ ] Faster R-CNN GMM-Det Training (ETA: 5 days)
- [ ] RetinaNet Evaluations (ETA: 1 week)
- [ ] RetinaNet GMM-Det Training (ETA: 1-2 weeks)

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

### Folder structure
Pascal VOC data can be downloaded from [here](http://host.robots.ox.ac.uk/pascal/VOC/). The VOC2007 training/validation data, VOC2007 annotated test data, and VOC2012 training/validation data should be downloaded. 

COCO data can be downloaded from [here](https://cocodataset.org/#download). The COCO 2017 train images, 2017 val images, and 2017 train/val annotations should be downloaded.

The datasets should be in the following format:
 
 <br>
 
    └── datasets
        └── data
            ├── VOCdevkit
            |    ├── VOC2007               # containing train/val and test data from VOC2007
            |    |    ├── Annotations      # xml annotation for each image
            |    |    ├── ImageSets
            |    |    |   ├── Main         # train, val and test txt files
            |    |    |   └── ... 
            |    |    ├── JPEGImages       # 9,963 images
            |    |    └── ...                 
            |    └── VOC2012               # containing train and val data from VOC2012
            |         ├── Annotations      # xml annotation for each image
            |         ├── ImageSets
            |         |   ├── Main         # train and val txt files
            |         |   └── ... 
            |         ├── JPEGImages       # 17,125 images
            |         └── ...     
            └── coco
                ├── images
                |   ├── train2017          # 118,287 images
                |   └── val2017            # 5,019 images
                ├── annotations
                |   ├── instances_train2017.json 
                |   └── instances_val2017.json
                └── ... 
                

### Creating open-set datasets
To create the open-set variant of each dataset, VOC-OS and COCO-OS, run the following commands:

```bash
cd datasets
python create_osdata.py --dataset voc
python create_osdata.py --dataset coco
```

This script will create 'closed-set' forms of VOC and COCO (i.e. VOC-CS and COCO-CS), and the original VOC and COCO will then be open-set datasets (i.e. VOC-OS and COCO-OS). For VOC, this is done by creating a new VOC2007CS and VOC2012CS folder with only closed-set images and closed-set annotations. For COCO, a new trainCS2017 and valCS2017 folder will be created, as well as new annotation files instances_trainCS2017.json and instances_valCS2017.json.                

## Pre-trained Models

## Evaluation

## Training 
