---
title: ONEAT Event Detection
emoji: 🔬
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: bsd-3-clause
---

# ONEAT Event Detection

Detect spatio-temporal events (e.g. mitosis) in 3D+T microscopy data using trained DenseVollNet models.

## How to use

1. Upload your **raw timelapse** (.tif) and **segmentation timelapse** (.tif)
2. Select a model checkpoint from the dropdown
3. Adjust model architecture parameters to match your trained model
4. Click **Run Prediction**
5. View results in the table and download as CSV

## Adding models

Place your `.ckpt` files in the `models/` directory and click "Refresh Models".
