# Road and Lane Segmentation Model Using Carla Simulator
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white)

> A Machine Learning project for road and lane segmentation.

## About the Project

A project to train a segmentation model for road and lane.
This serves as a foundation for reducing the input complexity of autonomous driving models, and I expect it to enable the implementation of high-performance models with lower resource usage.

## Preview

> Preview of segmentation.</br>

<div align="center">
  <table>
    <tr align="center">
      <th>Raw Image</th>
      <th>Segmented Image</th>
      <th>Composited Image</th>
    </tr>
    <tr align="center">
      <td><img height=150 src="https://github.com/user-attachments/assets/d81681cf-2646-479d-94af-0ae7b517e8f6"/></td>
      <td><img height=150 src="https://github.com/user-attachments/assets/d8e71c2e-f891-4280-8d03-0d000faeb4d4"/></td>
      <td><img height=150 src="https://github.com/user-attachments/assets/4a6bc96e-fee1-4a1b-9df3-4bff090ac5d0"/></td>
    </tr>
  </table>
</div>

## Requirements

- PyTorch
- Carla(v0.9.15)
- Carla Additional Maps(optional)

## Features

- Generate a dataset for training a new segmentation model.
- Train a new segmentation model for road and lane.
- Test segmentation with pre-trained model.

## To do

- Expand the segmentation scope to include not only roads and lanes but also surrounding objects such as other vehicles and pedestrians.
- Additional performance improvements are needed to meet real-time processing requirements.
