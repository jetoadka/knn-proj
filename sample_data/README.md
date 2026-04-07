# Sample Data

Committed subset (~100 images per domain) for initial experiments. Mirrors the `data/` directory layout.

## Contents

| Dataset | Count | Details |
|---------|-------|---------|
| `wiki_face_112/` | 102 images | 48 identities, 112x112 Wikipedia portraits (source domain) |
| `people_gator/aligned_112/train/` | 100 images | 15 libraries, newspaper face crops (target domain) |
| `people_gator/aligned_112/dev/` | 34 images | Validation with identity annotations |
| `people_gator/aligned_112/test/` | 35 images | Held-out evaluation with identity annotations |
| `webface4m/` | 100 images | 100 identities, 112x112 clean faces from WebFace4M |

Annotation files: `corresponding_faces_dev.jsonl` (48 records), `corresponding_faces_test.jsonl` (114 records).

## CUT Training Example

```bash
python train.py --dataroot_A sample_data/wiki_face_112 \
                --dataroot_B sample_data/people_gator/aligned_112/train \
                --name sample_cut_run
```

## Regenerating

```bash
# people_gator (--limit produces a subset)
python -m src.data_prep.preprocess_people_gator \
    --zip-path /path/to/people_gator__data_export.zip \
    --output-dir sample_data/people_gator --limit 100

# wiki_face_112
python -m src.data_prep.prepare_wiki_face \
    --zip-path /path/to/wiki_face_112_fin.zip \
    --output-dir sample_data/wiki_face_112 --limit 100
```

For full dataset preparation, see the [main README](../README.md#data-preparation).
