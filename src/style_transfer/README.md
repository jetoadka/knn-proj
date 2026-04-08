# Style Transfer Component (CUT)

This module is responsible for transforming clean faces (Domain A) into newspaper-style historical photos (Domain B) using the Contrastive Unpaired Translation (CUT) architecture.

## Overview of contents

* `cut_model/` - The core CUT model repository (cloned and integrated).
* `prepare_cut_data.py` - Script to flatten and format nested datasets into the strict `trainA/` and `trainB/` structure required by the CUT PyTorch Dataloader.
* `data_*/` - Auto-generated folders containing the flattened datasets for experiments (ignored in git).
* `wandb/` - Local Weights & Biases sync logs.

## Quick Start

**1. Format the data:**
The CUT model cannot read nested directories (like `PersonName/01.jpg`). Use our prep script to flatten them safely (it automatically prevents naming collisions). You can pass multiple datasets to mix them!

```bash
python prepare_cut_data.py \
    --clean ../../data/webface4m \
    --noisy ../../data/people_gator/aligned_112/train ../../data/wiki_face_112 \
    --output cut_dataset_mixed
```

**2. Start Training:**
Navigate into the cut_model directory and start the training. We use 112x112 crop sizes to match the Face Recognition downstream requirements.

```bash
cd cut_model
python train.py --dataroot ../cut_dataset_mixed --name newspaper_style_run --model cut --load_size 112 --crop_size 112 --display_freq 100
```

(Note: Add --gpu_ids -1 if testing locally on a CPU)
