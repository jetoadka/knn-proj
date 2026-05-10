# Style Transfer for Face Recognition on Historical Newspapers

Face recognition degrades on scanned historical newspapers due to print raster noise. This project bridges the domain gap using **CUT** (Contrastive Unpaired Translation) to synthetically transform clean face photos into newspaper-style images, then fine-tunes a recognition model on the augmented data.

**Team:** xbuchm03 (Nadzeya Antsipenka, Adriana Buchmei, Rostislav Lán)

## Project Phases

1. **Preprocessing** -- Normalize datasets (SCRFD alignment, 112x112 crops, train/dev/test splits)
2. **Style Transfer** -- Train CUT to transform clean faces into newspaper-style historical photos (PatchNCE preserves geometry)
3. **Downstream** -- Fine-tune face recognition with [timm-face](https://github.com/gau-nernst/timm-face) on generated data
4. **Evaluation** -- Compare Rank-1 accuracy, FAR/FRR on held-out newspaper test set

## Datasets

Full data lives under `data/` (gitignored). A ~100-image subset is committed in `sample_data/`.

| Dataset | Role | Size | Resolution |
| ------- | ---- | ---- | ---------- |
| [WebFace4M](https://huggingface.co/datasets/gaunernst/webface4m-wds-gz) | Source domain (clean faces) | ~4.2M images, 205k identities | 112x112 |
| people_gator | Target domain (newspaper faces) | 16,120 aligned crops | 112x112 |
| wiki_face_112 | Supplementary (Wikipedia portraits) | 3,223 images, 1,538 identities | 112x112 |

## Data Preparation

All dataset preprocessing, filtering, and raw-to-filtered instructions were moved to:

- `src/data_prep/README.md`

## Style Transfer Pipeline

```bash
# 1. Prepare flat directory structure for CUT (combining target domains)
python src/style_transfer/prepare_cut_data.py \
    --clean sample_data/webface4m \
    --noisy sample_data/people_gator/aligned_112/train sample_data/wiki_face_112 \
    --output src/style_transfer/data_combined

# 2. Train the CUT model
cd src/style_transfer/cut_model
python train.py --dataroot ../data_combined --name exp_combined --model cut --load_size 112 --crop_size 112
```

## Project Structure

```text
src/
├── data_prep/          # Dataset preprocessing scripts
├── style_transfer/     # CUT training
├── downstream/         # timm-face fine-tuning
└── evaluation/         # Metrics and evaluation
configs/                # Training configs
sample_data/            # Committed ~100-image subsets (see sample_data/README.md)
```

## References

- [CUT](https://arxiv.org/abs/2007.15651) -- Park et al., ECCV 2020
- [timm-face](https://github.com/gau-nernst/timm-face) -- Face recognition training with timm
- [WebFace4M](https://huggingface.co/datasets/gaunernst/webface4m-wds-gz) -- Large-scale face dataset
- [SCRFD](https://arxiv.org/abs/2105.04714) -- Face detection for alignment
