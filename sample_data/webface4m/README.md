# WebFace4M Sample Data

100 sample images (1 per identity, 112x112, RGB) extracted from shard `webface4m-0000.tar.gz`.

For the full dataset, download shards from HuggingFace:

```bash
python -m src.data_prep.download_webface4m --output-dir data/webface4m --num-shards 2
```
