"""Parse people_gator JSONL annotation files to build image→person_name mappings.

The people_gator dataset ships with ``corresponding_faces_*.jsonl`` files that
contain ground-truth person identities for each face crop.  Without these
annotations the directory hierarchy only provides *library* and *document*
information — **not** the actual person shown in the photo.

Usage::

    from src.evaluation.identity_mapping import load_identity_map

    id_map = load_identity_map("sample_data/people_gator/corresponding_faces_test.jsonl")
    # id_map["nlk/f70473e6-.../a699ac25-...__image_0__face_0.jpg"] == "Klement Gottwald"
"""

from __future__ import annotations

import json
from pathlib import Path


def load_identity_map(jsonl_path: str | Path) -> dict[str, str]:
    """Load a JSONL annotation file and return ``{face_relative_path: person_name}``.

    Each line in the JSONL file is expected to have at least:
    - ``"face"``: relative path of the face crop inside ``aligned_112/{split}/``
    - ``"person_name"``: ground-truth person identity

    Returns:
        Dictionary mapping the *face* path (as stored in the JSONL ``"face"``
        field) to the ``person_name`` string.
    """
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {jsonl_path}")

    id_map: dict[str, str] = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"  WARNING: Skipped line {line_no} in {jsonl_path.name}: {exc}")
                continue

            face_path = record.get("face")
            person_name = record.get("person_name")
            if face_path and person_name:
                id_map[face_path] = person_name

    return id_map


def load_identity_map_for_dir(
    data_dir: str | Path,
    jsonl_path: str | Path,
) -> dict[Path, str]:
    """Build ``{absolute_image_path: person_name}`` for all images in *data_dir*.

    This resolves the relative ``"face"`` paths from the JSONL against the
    actual directory tree so callers can look up identities by full path.

    Args:
        data_dir:   Root of the split, e.g. ``sample_data/people_gator/aligned_112/test``
        jsonl_path: Corresponding JSONL annotation file

    Returns:
        Dictionary mapping absolute ``Path`` objects to person names.
        Images present on disk but absent from the JSONL are silently skipped.
    """
    data_dir = Path(data_dir).resolve()
    face_to_person = load_identity_map(jsonl_path)

    result: dict[Path, str] = {}
    for rel_face, person in face_to_person.items():
        abs_path = data_dir / rel_face
        if abs_path.exists():
            result[abs_path] = person

    return result
