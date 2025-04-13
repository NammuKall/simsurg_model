SimSurgSkill Dataset Processing and Analysis
This repository contains code for processing and analyzing the SimSurgSkill 2021 dataset, which provides surgical skill simulation data, and implements object detection models for this task.
Overview
The SimSurgSkill dataset contains videos of simulated surgical procedures with ground truth annotations for skill metrics and bounding boxes of instruments. This repository provides:

Data processing utilities to convert videos to images
Visualization tools for skill metrics and bounding boxes
Implementation of deep learning models (ResNet and EfficientDet) for object detection

Repository Structure
sim_surg_skill/
├── main.py                # Main script to run the pipeline
├── README.md              # This file
└── src/                   # Source code directory
    ├── __init__.py        # Makes the directory a Python package
    ├── data_loader.py     # Functions for loading and preprocessing data
    ├── visualization.py   # Functions for data visualization
    ├── models.py          # Neural network model definitions
    └── utils.py           # Utility functions
Installation
bash# Clone the repository
git clone https://github.com/yourusername/sim_surg_skill.git
cd sim_surg_skill

# Install required packages
pip install -r requirements.txt
Dataset Structure
The code expects the SimSurgSkill dataset to be organized as follows:
simsurgskill_2021_dataset/
├── train_v1/
│   ├── videos/
│   │   ├── fps1/
│   │   └── fps30/
│   └── annotations/
│       ├── skill_metric_gt.csv
│       └── bounding_box_gt/
├── train_v2/
│   ├── videos/
│   │   ├── fps1/
│   │   └── fps30/
│   └── annotations/
│       └── bounding_box_gt/
└── test/
    ├── videos/
    │   ├── fps1/
    │   └── fps30/
    └── annotations/
Usage

Update the paths in main.py to point to your dataset location
Run the main script to process videos and load data:

bashpython main.py

For custom processing, you can import the modules directly:

pythonfrom src.data_loader import process_videos_to_images, load_train_data
from src.visualization import visualize_bounding_box
# ... and so on
Models
ResNet
A basic ResNet implementation is provided with customizable layers.
EfficientDet
A simplified EfficientDet implementation with:

ResNet50 backbone (pretrained on ImageNet)
BiFPN feature fusion
Classification and box regression heads

Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgements
The SimSurgSkill 2021 dataset is used for research purposes only. Please refer to the original dataset authors for proper citation and usage terms.
