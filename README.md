# ComfyUI-Real3D

## Acknowledgement
This project integrates the Real3D model, based on TripoSR, into ComfyUI. Real3D is a state-of-the-art Large Reconstruction Model (LRM) for 3D reconstruction from single-view real-world images. The ComfyUI-Real3D module is built upon the ComfyUI-Flowty-TripoSR code, modified to use the Real3D model instead. Special thanks to:

- [Hanwen Jiang](https://github.com/hwjiang1510) for creating [Real3D](https://github.com/hwjiang1510/Real3D)
- [flowtyone](https://github.com/flowtyone) for creating [ComfyUI-Flowty-TripoSR](https://github.com/flowtyone/ComfyUI-Flowty-TripoSR)

## Overview
Real3D introduces a novel self-training framework that can benefit from both existing 3D/multi-view synthetic data and diverse single-view real images. This repository aims to simplify the use of Real3D within ComfyUI for fast feedforward 3D reconstruction from a single image.

## Installation

### Prerequisites
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)

### Installation Steps

1. **Install ComfyUI-Real3D**

   - Set ComfyUI manager to "weak" security.
   - Import the Git repository in the manager:

     ```sh
     https://github.com/izuc/ComfyUI-Real3D.git
     ```

2. **Place Real3D Model and Configuration**

   - Download and place the Real3D model in the checkpoints directory (`ComfyUI/models/checkpoints`).
   - Place the `config.yaml` file in the same directory.

3. **Install Dependencies**
   
   ComfyUI will automatically read the `requirements.txt` file and install the necessary dependencies.

## Usage

### Example Workflow
Use the example workflows provided to test the functionality of the Real3D model in ComfyUI. You can find these workflows in the repository.

### Running the Model
1. **Load an Image**
   - Load your input image in ComfyUI.
   
2. **Configure Parameters**
   - Set the parameters such as geometry resolution, threshold, and model save format.

3. **Run the Model**
   - Execute the workflow to run the Real3D model and generate a 3D mesh from the input image.

