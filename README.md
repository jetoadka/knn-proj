# Style Transfer for Face Recognition on Historical Newspapers

Face recognition degrades on scanned historical newspapers due to print raster noise. This project bridges the domain gap using **CUT** (Contrastive Unpaired Translation) to synthetically transform clean face photos into newspaper-style images, then fine-tunes a recognition model on the augmented data.

**Team:** xbuchm03 (Nadzeya Antsipenka, Adriana Buchmei, Rostislav Lán)

## Project Phases

1. **Preprocessing** -- Normalize datasets (SCRFD alignment, 112x112 crops, train/dev/test splits)
2. **Style Transfer** -- Train CUT to transfer newspaper texture onto clean faces (PatchNCE preserves geometry)
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

```bash
pip install -r requirements.txt

# 1. Preprocess people_gator (extract, align, split)
python -m src.data_prep.preprocess_people_gator \
    --zip-path /path/to/people_gator__data_export.zip \
    --output-dir data/people_gator

# 1b. High-res re-extraction from page scans (recommended for curation)
python -m src.data_prep.preprocess_people_gator \
    --zip-path /path/to/people_gator__data_export.zip \
    --output-dir data/people_gator \
    --method realign \
    --target-size 224 224 \
    --metadata-path data/people_gator/metadata_224.jsonl

# 2. Download WebFace4M shards (~80 MB each)
python -m src.data_prep.download_webface4m \
    --output-dir data/webface4m --num-shards 2 --verify

# 3. Extract wiki_face_112
python -m src.data_prep.prepare_wiki_face \
    --zip-path /path/to/wiki_face_112_fin.zip \
    --output-dir data/wiki_face_112

# 4. Verify everything
python -m src.data_prep.dataset_stats --data-dir data
```

Use `--method realign` in step 1 for ArcFace-template alignment from page scans (slower, more precise).

### Diversity-Preserving Filtering (people_gator)

After generating high-res aligned crops (for example `aligned_224x224`), filter training data with quality metrics while preserving library/identity coverage:

```bash
# 5. Quality + diversity filtering (reject only worst samples per library)
python -m src.data_prep.filter_people_gator \
    --aligned-dir data/people_gator/aligned_224x224 \
    --output-dir data/people_gator/filter_run_224 \
    --splits train \
    --drop-ratio-per-library 0.15 \
    --min-per-identity 1

# 6. Compare pre/post stats and quality quantiles
python -m src.data_prep.dataset_stats \
    --data-dir data \
    --filter-report data/people_gator/filter_run_224/filter_report.json \
    --kept-manifest data/people_gator/filter_run_224/kept_manifest.jsonl \
    --rejected-manifest data/people_gator/filter_run_224/rejected_manifest.jsonl
```

Recommended defaults:

- Start with `--target-size 224 224`.
- Use `--drop-ratio-per-library 0.10` to `0.20` to remove only low-value outliers.
- Keep `--min-per-identity 1` to avoid dropping rare identities.
- Tune thresholds on a sample run first, then run full dataset.

### Optional Face-Detector Filtering Pass

After quality filtering, you can run a detector-based pass to remove the lowest-confidence face images (including many no-face samples).

```bash
python -m src.data_prep.filter_by_face_detector \
    --input-dataset-dir data/upload_ready/people_gator_full_filtered_drop05 \
    --output-dataset-dir data/upload_ready/people_gator_full_filtered_drop05_facedet \
    --drop-rate 0.05 \
    --splits train dev test \
    --copy-dropped
```

Notes:

- `--drop-rate` is user-controlled (e.g. `0.02`, `0.05`, `0.10`).
- Output keeps original `aligned_112/{train,dev,test}` structure.
- `corresponding_faces*.jsonl` are filtered to kept samples.
- Detector metadata is written into `_metadata/face_detector_filter_summary.json` and `_metadata/face_detector_scores.jsonl`.

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
