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

```bash
pip install -r requirements.txt

# 1. Preprocess people_gator (extract, align, split)
python -m src.data_prep.preprocess_people_gator \
    --zip-path /path/to/people_gator__data_export.zip \
    --output-dir /path/to/people_gator_output

# 1b. High-res re-extraction from page scans (recommended for curation)
python -m src.data_prep.preprocess_people_gator \
    --zip-path /path/to/people_gator__data_export.zip \
    --output-dir /path/to/people_gator_output \
    --method realign \
    --target-size 224 224 \
    --metadata-path /path/to/people_gator_output/metadata_224.jsonl

# 2. Download WebFace4M shards (~80 MB each)
python -m src.data_prep.download_webface4m \
    --output-dir /path/to/webface4m_output --num-shards 2 --verify

# 3. Extract wiki_face_112
python -m src.data_prep.prepare_wiki_face \
    --zip-path /path/to/wiki_face_112_fin.zip \
    --output-dir /path/to/wiki_face_output

# 4. Verify everything
python -m src.data_prep.dataset_stats --data-dir data
```

Use `--method realign` in step 1 for ArcFace-template alignment from page scans (slower, more precise).

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

After generating high-res aligned crops (for example `aligned_224x224`), filter training data with quality metrics while preserving library/identity coverage:

```bash
# 5. Quality + diversity filtering (reject only worst samples per library)
python -m src.data_prep.filter_people_gator \
    --aligned-dir /path/to/people_gator_output/aligned_224x224 \
    --output-dir /path/to/filter_run_224 \
    --splits train \
    --drop-ratio-per-library 0.15 \
    --min-per-identity 1

# 6. Compare pre/post stats and quality quantiles
python -m src.data_prep.dataset_stats \
    --data-dir /path/to/data_root \
    --filter-report /path/to/filter_run_224/filter_report.json \
    --kept-manifest /path/to/filter_run_224/kept_manifest.jsonl \
    --rejected-manifest /path/to/filter_run_224/rejected_manifest.jsonl
```

Recommended defaults:

- Start with `--target-size 224 224`.
- Use `--drop-ratio-per-library 0.10` to `0.20` to remove only low-value outliers.
- Keep `--min-per-identity 1` to avoid dropping rare identities.
- Tune thresholds on a small copy of the dataset first, then run full dataset.

### Optional Face-Detector Filtering Pass

After quality filtering, you can run a detector-based pass.
Current behavior is rule-first:

- drop no-face samples by default,
- drop below `--min-face-confidence` threshold,
- then optionally apply percentile trimming via `--drop-rate` on the remaining set.

```bash
python -m src.data_prep.filter_by_face_detector \
    --input-dataset-dir /path/to/input_dataset \
    --output-dataset-dir /path/to/face_filtered_output \
    --min-face-confidence 0.1 \
    --drop-rate 0.05 \
    --splits train dev test \
    --copy-dropped
```

Notes:

- `--drop-rate` is optional percentile trimming after rule-based drops.
- Use `--keep-no-face` to disable default no-face dropping.
- Output keeps original `aligned_112/{train,dev,test}` structure.
- `corresponding_faces*.jsonl` are filtered to kept samples.
- Detector metadata is written into `_metadata/face_detector_filter_summary.json` and `_metadata/face_detector_scores.jsonl`.

### Serial Master Filtering Pipeline (Config-Driven)

To run filters serially from a single config (quality -> face detector -> low detail), use:

```bash
python -m src.data_prep.run_filter_pipeline \
    --config /path/to/filter_pipeline_config.json
```

Config file:

- `input_dataset_dir`: source dataset root with `aligned_112/...`
- `output_root_dir`: where stage outputs are written
- `stages`: ordered list of filtering stages with per-stage parameters

Supported stage types:

- `quality` (uses `drop_ratio_per_library`)
- `face_detector` (uses `drop_rate`)
- `low_detail` (uses `drop_rate`)

### Final Filtering Config Used

The latest filtered dataset was generated with:

```bash
python -m src.data_prep.run_filter_pipeline \
    --config /path/to/filter_pipeline_config.json
```

This config applies:

- quality filtering (`5%` per-library drop),
- aggressive face-detector filtering (`5%` drop),
- low-detail filtering (`3%` drop, reconstruction-change metric).

## Raw To Filtered (Step-by-Step)

From raw `people_gator` export zip to final filtered dataset:

```bash
# 1) Preprocess raw archive -> split aligned dataset
python -m src.data_prep.preprocess_people_gator \
    --zip-path /path/to/people_gator_export.zip \
    --output-dir /path/to/people_gator_preprocessed

# 2) Run serial filtering pipeline
python -m src.data_prep.run_filter_pipeline \
    --config /path/to/filter_pipeline_config.json

# 3) Final filtered output is the last stage directory
# /path/to/pipeline_output

# 4) (Optional) Verify resulting structure/stats
python -m src.data_prep.dataset_stats --data-dir /path/to/pipeline_output
```

Resulting filtered dataset contains:

- kept samples under `aligned_112/{train,dev,test}`
- dropped samples under `_dropped/{train,dev,test}` (for stages with `copy_dropped: true`)
- filtered `corresponding_faces*.jsonl`

### Data Prep Script Index

Current `src/data_prep` scripts and their role:

- `preprocess_people_gator.py` - extract/align/split `people_gator` from export zip.
- `download_webface4m.py` - download selected WebFace4M shards.
- `prepare_wiki_face.py` - extract and organize `wiki_face_112_fin`.
- `dataset_stats.py` - report dataset and split statistics.
- `filter_people_gator.py` - quality/diversity filtering for aligned crops.
- `filter_by_face_detector.py` - drop lowest-confidence/no-face samples.
- `filter_by_low_detail.py` - drop lowest-detail samples via reconstruction-change score.
- `run_filter_pipeline.py` - serial config-driven runner (`quality -> face_detector -> low_detail`).
- `common_dataset_ops.py` - shared JSONL/dataset manifest and copy helpers.

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
