# SimSurgSkill Analysis Framework

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/license-MIT-green)

A comprehensive framework for processing, analyzing, and training deep learning models on the SimSurgSkill 2021 dataset, which contains surgical skill simulation data for surgical instrument tracking and skill assessment.

![Surgical Simulation](https://img.shields.io/badge/Surgical%20Simulation-Computer%20Vision-blue)

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Usage](#usage)
- [Models](#models)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Citations](#citations)

## 🔍 Overview

This repository provides a robust framework for the analysis of surgical skills through computer vision. It processes video data from simulated surgical procedures, extracts frames, detects instruments, and assesses surgical skill metrics. The implementation includes state-of-the-art deep learning models for object detection and skill prediction.

## ✨ Features

- **Video Processing**: Convert surgical procedure videos to frame sequences
- **Data Visualization**: Tools for visualizing skill metrics and instrument tracking
- **Object Detection**: Implement ResNet and EfficientDet models for instrument tracking
- **Skill Assessment**: Analyze metrics such as needle drops and out-of-view events
- **Modular Design**: Easy to extend with new models and analysis methods

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/NammuKall/simsurg_model.git
cd sim_surg_skill

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📁 Project Structure

```
sim_surg_skill/
├── main.py               # Entry point for running the pipeline
├── requirements.txt      # Required dependencies
├── README.md             # Project documentation
└── src/                  # Source code modules
    ├── __init__.py       # Package initialization
    ├── data_loader.py    # Data loading and preprocessing
    ├── visualization.py  # Data visualization tools
    ├── models.py         # Neural network implementations
    └── utils.py          # Utility functions
```

## 💾 Dataset

The SimSurgSkill 2021 dataset contains videos of simulated surgical procedures with annotated skill metrics and bounding boxes for instruments. The dataset is organized as follows:

```
simsurgskill_2021_dataset/
├── train_v1/             # Training set (version 1)
│   ├── videos/           # Video recordings
│   └── annotations/      # Ground truth annotations
├── train_v2/             # Training set (version 2)
│   ├── videos/
│   └── annotations/
└── test/                 # Test set
    ├── videos/
    └── annotations/
```

Key metrics in the dataset include:
- Needle drop counts
- Instrument out-of-view counts
- Bounding box coordinates for instruments

## 🚀 Usage

### Basic Usage

```python
# Run the complete pipeline
python main.py

# Specify data paths (optional)
python main.py --data_dir /path/to/dataset --output_dir /path/to/results
```

### Advanced Usage

```python
# Import modules for custom processing
from src.data_loader import process_videos_to_images, load_train_data
from src.visualization import visualize_metrics, visualize_bounding_box
from src.models import EfficientDetModel

# Process videos to images
process_videos_to_images('/path/to/videos')

# Load and visualize data
train_data, train_array = load_train_data('/path/to/images')
metrics_df = pd.read_csv('/path/to/metrics.csv')
visualize_metrics(metrics_df, 'needle_drop_counts', 'instrument_out_of_view_counts')

# Initialize and train model
model = EfficientDetModel(num_classes=3)
# Train model...
```

## 🧠 Models

### ResNet

A customizable implementation of the ResNet architecture for image classification and feature extraction.

- Configurable number of layers
- Residual connections for better gradient flow
- Batch normalization for stable training

### EfficientDet

A state-of-the-art object detection model with:

- ResNet50 backbone (pretrained on ImageNet)
- Bidirectional Feature Pyramid Network (BiFPN) for multi-scale feature fusion
- Classification and bounding box regression heads
- Optimized for real-time instrument tracking

## 📊 Results

(Note: This section would include model performance metrics and visualizations from your experiments.)

## 👐 Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue to discuss potential improvements.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 Citations

If you use this code or the SimSurgSkill dataset in your research, please cite the following:

```
@article{simsurgskill2021,
  title={SimSurgSkill: A Supervised Learning Approach for Surgical Skill Assessment},
  author={[Dataset Authors]},
  journal={[Journal]},
  year={2021}
}
```

## 📧 Contact

Your Name - [your.email@example.com](mailto:your.email@example.com)

Project Link: [https://github.com/yourusername/sim_surg_skill](https://github.com/yourusername/sim_surg_skill)
