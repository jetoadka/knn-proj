# Style Transfer Component (CUT)

This module is responsible for transforming clean faces (Domain A) into newspaper-style historical photos (Domain B) using the Contrastive Unpaired Translation (CUT) architecture.

## Overview of Contents

* `cut_model/` - The core CUT model repository (cloned and integrated).
* `prepare_cut_data.py` - Script to flatten and format nested datasets into the strict `trainA/` and `trainB/` structure required by the CUT PyTorch DataLoader.
* `extract_images.py` - Script to efficiently extract a specific number of clean images (e.g., 15,000) directly from the `.tar.gz` shard without unpacking the entire dataset.
* `data_*/` - Auto-generated directories containing the flattened datasets for experiments (ignored in git).
* `wandb/` - Local Weights & Biases synchronization logs.

## Quick Start

### 1. Extract Clean Images
Before formatting, extract a subset of clean faces directly from the WebDataset shard. This script pulls the exact number of images needed (e.g., 15,000) for training the generator without unpacking the massive archive.

```bash
python extract_images.py \
    --shard-path ../../data/webface4m/webface4m-0000.tar.gz \
    --output-dir ../../data/webface4m/clean_15k \
    --count 15000
```

### 2. Format the Data
The standard CUT model cannot directly read nested identity directories (e.g., `PersonName/01.jpg`). Use our preparation script to flatten them safely (it automatically resolves naming collisions). You can pass multiple datasets to merge them into a single target domain!

```bash
python prepare_cut_data.py \
    --clean ../../data/webface4m \
    --noisy ../../data/people_gator/aligned_112/train ../../data/wiki_face_112 \
    --output cut_dataset_mixed
```

### 3. Start Training
Navigate into the cut_model directory and start the training process. We explicitly use 112x112 load and crop sizes to match the input requirements for the downstream Face Recognition task.

```bash
python train.py \
    --dataroot ../data \
    --name exp_combined_200 \
    --model cut \
    --load_size 112 \
    --crop_size 112 \
    --n_epochs 100 \
    --n_epochs_decay 100 \
    --batch_size 8 \
    --num_threads 4 \
    --display_freq 1000
```
(Note: Append --gpu_ids -1 to the command if you are testing the pipeline locally on a CPU.)

### 4. Generate Synthetic Data (Inference)
Once the model is trained, use the testing script to apply the learned style to your clean dataset. To generate a large batch of images for the downstream task (e.g., 15,000 images), use the --num_test parameter:

```bash
python test.py \
    --dataroot ../cut_dataset_mixed \
    --name newspaper_style_run \
    --model cut \
    --CUT_mode CUT \
    --num_test 15000
```
The augmented images will be saved in the results/newspaper_style_run/ directory, ready to be used for training the downstream timm classifier.