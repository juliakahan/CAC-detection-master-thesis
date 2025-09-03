# Coronary Artery Calcification Detection on Non-Gated Chest CT

This repository contains the code and scripts developed as part of the MSc thesis:

Detection and quantitative assessment of coronary artery calcifications on chest computed tomography
Author: Julia Kahan, AGH University of Science and Technology, Kraków, 2025

📌 Overview

Coronary artery calcification (CAC) is a well-established biomarker of atherosclerosis and an independent predictor of major cardiovascular events.
This project implements a deep learning–based pipeline for the segmentation of coronary arteries and mediastinum using nnU-Net, followed by automated CAC detection and Agatston scoring on non-gated chest CT scans.

The pipeline is composed of four main stages:

* Dataset preparation – exporting labelmaps, converting to NIfTI, and assembling the nnU-Net raw dataset.

* Segmentation – training and inference using nnU-Net (v2).

* CAC detection and scoring – applying intensity thresholding and Agatston methodology within segmented vessels.

* Evaluation – numeric correlations, categorical agreement, and distribution summaries.


## Repository structure
<pre ''' >CAC-detection-master-thesis/
│
├── dataset_preparation/     # Scripts for preparing CT + labelmaps for nnU-Net
│   ├── build_nnunet_dataset.py
│   └── check_labels.py
│
├── athena_training_scripts/ # Scripts of batch jobs for training/inference on Athena 
│
├── cac_detection/           # CAC detection and Agatston scoring
│   ├── run_cac_in_slicer.py # Main function (3D Slicer integration)
│   └── run_cac_cli.py       # Thin CLI wrapper for Slicer execution
│
├── evaluation/              # Post-processing and evaluation scripts
│   ├── build_master_dataset.py
│   ├── eval_cac_numeric.py
│   ├── eval_cac_categories.py
│   └── summarize_cac_distribution.py
│
├── segmentation_eval/       # Segmentation quality evaluation (e.g., Dice, HD)
│
└── README.md                # (this file) 
  ``` </pre>
