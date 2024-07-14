# Medical Image Segmentation Preprocessing

This repo contains the preprocessing pipeline for 3D medical image segmentation tasks. The pipeline is built using the MONAI framework and PyTorch, focusing on tasks such as loading, spacing, intensity scaling, cropping, and resizing of 3D medical images.

## Features
- Loading of 3D Medical Images and Labels in NifTI format (`*.nii.gz`).
- **Tranformations**: A comprehensive suite of preprocessing transformations including:
  - Adding channels
  - Adjusting voxel spacing
  - Scaling intensity values
  - Cropping foreground regions
  - Resizing to standard dimensions
- **Data Loading**: Streamlined dataset and dataloader creation for easy integration into training workflows.

## Prerequisites
- Python
- PyTorch
- MONAI
- Matplotlib

You can install MONAI using the following command:
```bash
pip install monai
```

## Example
![image](https://github.com/user-attachments/assets/97d21f06-83fd-4f7d-b21e-20722a66c538)
