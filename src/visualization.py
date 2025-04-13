#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualization functions for the SimSurgSkill dataset
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
from PIL import Image, ImageDraw

def visualize_metrics(metric_df, x_column, y_column):
    """
    Create a scatter plot of two metrics
    
    Args:
        metric_df (pandas.DataFrame): DataFrame containing metrics
        x_column (str): Column name for x-axis
        y_column (str): Column name for y-axis
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=metric_df, x=x_column, y=y_column)
    plt.title(f'{x_column} vs {y_column}')
    plt.grid(True)
    plt.show()

def visualize_matplotlib_bbox(x, y, w, h, xlim=(1, 9), ylim=(4, 12)):
    """
    Visualize a bounding box using Matplotlib
    
    Args:
        x, y (float): Coordinates of the top-left corner
        w, h (float): Width and height of the bounding box
        xlim, ylim (tuple): Limits for the axes
    """
    fig, ax = plt.subplots()
    rectangle = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='y')
    ax.add_patch(rectangle)
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(True)
    
    plt.show()

def visualize_bounding_box(image_path, x, y, w, h):
    """
    Visualize a bounding box on an image using PIL
    
    Args:
        image_path (str): Path to the image file
        x, y (int): Coordinates of the top-left corner
        w, h (int): Width and height of the bounding box
    """
    img = Image.open(image_path)
    
    draw = ImageDraw.Draw(img)
    rectangle = [x, y, x+w, y+h]
    draw.rectangle(rectangle, outline='red', width=2)
    
    image_np = np.array(img)
    plt.figure(figsize=(12, 8))
    plt.imshow(image_np)
    plt.axis('off')
    plt.title('Image with Bounding Box')
    plt.show()

def display_image(image, title="Image"):
    """
    Display an image using OpenCV
    
    Args:
        image (numpy.ndarray): Image to display
        title (str): Title for the plot
    """
    plt.figure(figsize=(12, 8))
    # Convert BGR to RGB for matplotlib
    if image.shape[-1] == 3:
        image = image[:, :, ::-1]
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()
  
