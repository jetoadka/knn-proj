# Style Transfer for Face Recognition on Historical Newspapers

Face recognition degrades on scanned historical newspapers due to print raster noise. This project bridges the domain gap using **CUT** (Contrastive Unpaired Translation) to synthetically transform clean face photos into newspaper-style images, then fine-tunes a recognition model on the augmented data.

**Team:** xbuchm03 (Nadzeya Antsipenka, Adriana Buchmei, Rostislav Lán)
## Project Phases

1. **Preprocessing & Filtering** -- Normalize datasets (112x112 crops), apply a custom filtration pipeline (quality scoring, face detection certainty, reconstruction error), and use K-means clustering to separate target domains (e.g., colored vs. grayscale historical newspaper).
2. **Style Transfer** -- Train the Contrastive Unpaired Translation (CUT) architecture to transform clean faces into newspaper-style historical photos, utilizing PatchNCE loss to strictly preserve biometric geometry.
3. **Downstream Task** -- Fine-tune a face recognition classifier (ConvNeXt-Atto with CosFace loss) using the [timm](https://github.com/huggingface/pytorch-image-models) library on the synthetically augmented data.
4. **Evaluation** -- Compare Rank-1/5 accuracy, Precision@1, NDCG@10, and TAR@FAR=$10^{-4}$ on a strictly held-out historical newspaper test set.

## Datasets

Full data lives under the `data/` directory (ignored in git to save space). A small illustrative subset (~100 images) is committed in `sample_data/` to comply with upload limits and demonstrate the pipeline.

| Dataset | Role | Size | Resolution |
| ------- | ---- | ---- | ---------- |
| [WebFace4M](https://huggingface.co/datasets/gaunernst/webface4m-wds-gz) | Source domain (clean faces) | ~4.2M images, 205k identities | 112x112 |
| People Gator | Target domain (newspaper faces) | 16,120 aligned crops | 112x112 |
| WikiFace | Supplementary (historical portraits)| 3,223 images, 1,538 identities | 112x112 |

## Project Structure

Our source code is divided into logical modules. **Each subdirectory inside `src/` contains its own dedicated `README.md` file** with detailed instructions on how to execute the specific scripts, format the data, and reproduce our experiments.

```text
src/
├── data_prep/          # Scripts for WebDataset extraction, filtering, and K-means clustering
├── style_transfer/     # CUT architecture integration and generator training
├── downstream/         # Face recognition fine-tuning (ConvNeXt-Atto via timm)
└── evaluation/         # Metrics calculation and validation scripts
configs/                # Hyperparameter configurations for training
sample_data/            # Committed ~100-image subsets (see sample_data/README.md)
report.pdf              # Final project report detailing methodology, experiments, and results
```
