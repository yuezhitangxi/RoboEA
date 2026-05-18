#!/usr/bin/env python
"""Offline Full-MHN dataset preprocessor for RoboEA.

The script copies an original dataset directory and injects a fixed
Multi-source Heterogeneous Noise (Full-MHN) dataset into the copy. It does not
change training code and never adds noise dynamically in model forward passes.
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is expected in RoboEA, but keep parsing usable.
    torch = None


Triple = Tuple[int, int, int]
Link = Tuple[int, int]

COMMON_FEATURE_FILES = {
    "img": ["img_features.npy", "image_features.npy", "img_emb.npy", "vis_features.npy"],
    "att": ["att_features.npy", "attr_features.npy", "attribute_features.npy"],
    "name": ["name_features.npy", "name_emb.npy"],
    "char": ["char_features.npy", "char_emb.npy"],
}

LINK_CANDIDATES = ["train_ill", "train_links", "train.txt", "ref_ent_ids", "ill_ent_ids"]
RAW_MODALITY_MARKERS = {"att": "__raw_training_attrs__", "img": "__roboea_img_pkl__"}


def read_triples(path: Path | str) -> List[Triple]:
    """Read triples separated by spaces or tabs."""
    triples: List[Triple] = []
    with Path(path).open("r", encoding="utf-8") as fr:
        for line_no, line in enumerate(fr, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"{path}:{line_no} expected 3 columns, got {len(parts)}")
            try:
                triples.append((int(parts[0]), int(parts[1]), int(parts[2])))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no} contains non-integer triple ids: {line}") from exc
    return triples


def write_triples(path: Path | str, triples: Sequence[Triple]) -> None:
    with Path(path).open("w", encoding="utf-8") as fw:
        for h, r, t in triples:
            fw.write(f"{h}\t{r}\t{t}\n")


def read_links(path: Path | str) -> List[Link]:
    """Read aligned entity pairs separated by spaces or tabs."""
    links: List[Link] = []
    with Path(path).open("r", encoding="utf-8") as fr:
        for line_no, line in enumerate(fr, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"{path}:{line_no} expected 2 columns, got {len(parts)}")
            try:
                links.append((int(parts[0]), int(parts[1])))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no} contains non-integer link ids: {line}") from exc
    return links


def load_feature(path: Path | str) -> Any:
    """Load feature files in .npy, .pt/.pth, or .pkl format."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path, allow_pickle=True)
    if suffix in {".pt", ".pth"}:
        if torch is None:
            raise RuntimeError("torch is required to load .pt/.pth feature files")
        return torch.load(path, map_location="cpu")
    if suffix == ".pkl":
        with path.open("rb") as fr:
            return pickle.load(fr)
    raise ValueError(f"Unsupported feature format for {path}; expected .npy, .pt/.pth, or .pkl")


def save_feature(path: Path | str, feature: Any, original_path: Path | str) -> None:
    """Save features using the original file extension."""
    path = Path(path)
    suffix = Path(original_path).suffix.lower()
    path.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".npy":
        np.save(path, feature)
        return
    if suffix in {".pt", ".pth"}:
        if torch is None:
            raise RuntimeError("torch is required to save .pt/.pth feature files")
        torch.save(feature, path)
        return
    if suffix == ".pkl":
        with path.open("wb") as fw:
            pickle.dump(feature, fw, protocol=pickle.HIGHEST_PROTOCOL)
        return
    raise ValueError(f"Unsupported feature format for {original_path}; expected .npy, .pt/.pth, or .pkl")


def parse_feature_map(feature_map: str | None) -> Dict[str, str]:
    """Parse modality:file pairs, for example img:img_features.npy,att:att_features.npy."""
    if not feature_map:
        return {}
    parsed: Dict[str, str] = {}
    for item in feature_map.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid --feature_map entry '{item}', expected modality:path")
        modality, rel_path = item.split(":", 1)
        modality = modality.strip()
        rel_path = rel_path.strip()
        if not modality or not rel_path:
            raise ValueError(f"Invalid --feature_map entry '{item}', modality and path are required")
        parsed[modality] = rel_path
    return parsed


def parse_modalities(raw: str) -> List[str]:
    modalities = [item.strip() for item in raw.split(",") if item.strip()]
    if not modalities:
        raise ValueError("--modalities must contain at least one modality")
    return modalities


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def copy_feature(feature: Any) -> Any:
    if isinstance(feature, np.ndarray):
        return feature.copy()
    if torch is not None and isinstance(feature, torch.Tensor):
        return feature.clone()
    if isinstance(feature, dict):
        copied = {}
        for key, value in feature.items():
            if isinstance(value, np.ndarray):
                copied[key] = value.copy()
            elif torch is not None and isinstance(value, torch.Tensor):
                copied[key] = value.clone()
            else:
                copied[key] = pickle.loads(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
        return copied
    return pickle.loads(pickle.dumps(feature, protocol=pickle.HIGHEST_PROTOCOL))


def feature_has_entity(feature: Any, entity: int) -> bool:
    if isinstance(feature, np.ndarray):
        return 0 <= entity < len(feature)
    if torch is not None and isinstance(feature, torch.Tensor):
        return 0 <= entity < feature.shape[0]
    if isinstance(feature, Mapping):
        return entity in feature
    return False


def set_feature_zero(feature: Any, entity: int) -> None:
    if isinstance(feature, np.ndarray):
        feature[entity] = 0
    elif torch is not None and isinstance(feature, torch.Tensor):
        feature[entity] = 0
    elif isinstance(feature, MutableMapping):
        value = feature[entity]
        if isinstance(value, np.ndarray):
            feature[entity] = np.zeros_like(value)
        elif torch is not None and isinstance(value, torch.Tensor):
            feature[entity] = torch.zeros_like(value)
        elif isinstance(value, list):
            feature[entity] = []
        else:
            feature[entity] = 0
    else:
        raise TypeError(f"Unsupported feature object type: {type(feature).__name__}")


def assign_feature_from(feature: Any, original_feature: Any, entity: int, source_entity: int) -> None:
    if isinstance(feature, np.ndarray):
        feature[entity] = original_feature[source_entity]
    elif torch is not None and isinstance(feature, torch.Tensor):
        feature[entity] = original_feature[source_entity].clone()
    elif isinstance(feature, MutableMapping):
        value = original_feature[source_entity]
        if isinstance(value, np.ndarray):
            feature[entity] = value.copy()
        elif torch is not None and isinstance(value, torch.Tensor):
            feature[entity] = value.clone()
        elif isinstance(value, list):
            feature[entity] = list(value)
        else:
            feature[entity] = pickle.loads(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
    else:
        raise TypeError(f"Unsupported feature object type: {type(feature).__name__}")


def feature_entities(feature: Any, candidates: Iterable[int]) -> List[int]:
    return [entity for entity in candidates if feature_has_entity(feature, entity)]


def resolve_input_path(data_dir: Path, path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else data_dir / path


def resolve_output_path(data_dir: Path, output_dir: Path, input_path: Path) -> Path:
    try:
        return output_dir / input_path.resolve().relative_to(data_dir.resolve())
    except ValueError:
        # External feature paths can be useful for pkl-based image features. Keep
        # the generated copy self-contained by placing them in output_dir.
        return output_dir / input_path.name


def runpy_img_path(file_dir: Path, repo_root: Path) -> Path:
    """Mirror RoboEA/src/run.py:load_img_features_use_mean_img path selection."""
    file_dir_text = str(file_dir)
    if "V1" in file_dir_text:
        return repo_root / "data/pkls/dbpedia_wikidata_15k_norm_GA_id_img_feature_dict.pkl"
    if "V2" in file_dir_text:
        return repo_root / "data/pkls/dbpedia_wikidata_15k_dense_GA_id_img_feature_dict.pkl"
    if "FBDB15K" in file_dir_text:
        return repo_root / "data/mmkg/pkls/FBDB15K_id_img_feature_dict.pkl"
    split = file_dir.name
    return repo_root / "data/mmkg/pkls" / f"{split}_GA_id_img_feature_dict.pkl"


def raw_attrs_available(data_dir: Path) -> bool:
    return (data_dir / "training_attrs_1").exists() and (data_dir / "training_attrs_2").exists()


def discover_feature_files(data_dir: Path, output_dir: Path, modalities: Sequence[str], explicit_map: Dict[str, str]) -> Dict[str, Path | str]:
    discovered: Dict[str, Path | str] = {}
    repo_root = repo_root_from_script()
    for modality in modalities:
        if modality in explicit_map:
            candidate = resolve_input_path(data_dir, explicit_map[modality])
            if candidate.exists():
                discovered[modality] = candidate
            else:
                print(f"[warning] feature file for modality '{modality}' not found: {candidate}")
            continue

        for filename in COMMON_FEATURE_FILES.get(modality, []):
            candidate = data_dir / filename
            if candidate.exists():
                discovered[modality] = candidate
                break
        if modality not in discovered and modality == "att" and raw_attrs_available(data_dir):
            discovered[modality] = RAW_MODALITY_MARKERS["att"]
            continue
        if modality not in discovered and modality == "img":
            img_path = runpy_img_path(data_dir, repo_root)
            if img_path.exists():
                discovered[modality] = RAW_MODALITY_MARKERS["img"]
                continue
        if modality not in discovered:
            print(f"[warning] no feature file found for modality '{modality}', skip this modality")
    return discovered


def find_links_file(data_dir: Path, links_file: str | None) -> Path | None:
    if links_file:
        path = resolve_input_path(data_dir, links_file)
        if not path.exists():
            raise FileNotFoundError(f"--links_file does not exist: {path}")
        return path
    for filename in LINK_CANDIDATES:
        path = data_dir / filename
        if path.exists():
            return path
    return None


def read_ent_ids(path: Path) -> List[int]:
    ent_ids: List[int] = []
    with path.open("r", encoding="utf-8") as fr:
        for line_no, line in enumerate(fr, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                ent_ids.append(int(parts[0]))
            except (IndexError, ValueError) as exc:
                raise ValueError(f"{path}:{line_no} has invalid entity id line: {line}") from exc
    return ent_ids


def read_ent_id_map(path: Path) -> Dict[str, int]:
    ent2id: Dict[str, int] = {}
    with path.open("r", encoding="utf-8") as fr:
        for line_no, line in enumerate(fr, start=1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                raise ValueError(f"{path}:{line_no} has invalid ent_ids row: {line.rstrip()}")
            ent2id[parts[1]] = int(parts[0])
    return ent2id


def read_attr_rows(path: Path, ent2id: Mapping[str, int]) -> Tuple[Dict[int, List[str]], Dict[int, str]]:
    rows: Dict[int, List[str]] = {}
    labels: Dict[int, str] = {}
    with path.open("r", encoding="utf-8") as fr:
        for line_no, line in enumerate(fr, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if not parts or parts[0] not in ent2id:
                continue
            entity = int(ent2id[parts[0]])
            rows[entity] = parts[1:]
            labels[entity] = parts[0]
    return rows, labels


def write_attr_rows(path: Path, rows: Mapping[int, Sequence[str]], labels: Mapping[int, str]) -> None:
    with path.open("w", encoding="utf-8") as fw:
        for entity in sorted(rows):
            attrs = list(rows[entity])
            label = labels[entity]
            if attrs:
                fw.write(label + "\t" + "\t".join(attrs) + "\n")
            else:
                fw.write(label + "\n")


def get_kg_entities(data_dir: Path, triples_by_name: Mapping[str, Sequence[Triple]]) -> Tuple[List[int], List[int]]:
    ent_ids_1 = data_dir / "ent_ids_1"
    ent_ids_2 = data_dir / "ent_ids_2"
    if ent_ids_1.exists() and ent_ids_2.exists():
        return sorted(set(read_ent_ids(ent_ids_1))), sorted(set(read_ent_ids(ent_ids_2)))

    triples_1 = triples_by_name.get("triples_1", [])
    triples_2 = triples_by_name.get("triples_2", [])
    kg1 = sorted({h for h, _, _ in triples_1}.union({t for _, _, t in triples_1}))
    kg2 = sorted({h for h, _, _ in triples_2}.union({t for _, _, t in triples_2}))
    return kg1, kg2


def entities_for_side(kg1_entities: Sequence[int], kg2_entities: Sequence[int], kg_side: str) -> List[int]:
    if kg_side == "kg1":
        return list(kg1_entities)
    if kg_side == "kg2":
        return list(kg2_entities)
    return sorted(set(kg1_entities).union(kg2_entities))


def triples_names_for_side(kg_side: str) -> List[str]:
    if kg_side == "kg1":
        return ["triples_1"]
    if kg_side == "kg2":
        return ["triples_2"]
    return ["triples_1", "triples_2"]


def sample_without_replacement(rng: np.random.Generator, items: Sequence[Any], count: int) -> List[Any]:
    if count <= 0 or not items:
        return []
    count = min(count, len(items))
    indices = rng.choice(len(items), size=count, replace=False)
    return [items[int(index)] for index in indices]


def random_other_entity(rng: np.random.Generator, entities: Sequence[int], entity: int) -> int | None:
    if len(entities) < 2:
        return None
    while True:
        source = int(entities[int(rng.integers(0, len(entities)))])
        if source != entity:
            return source


def corrupt_features(
    args: argparse.Namespace,
    feature_files: Mapping[str, Path | str],
    output_dir: Path,
    kg1_entities: Sequence[int],
    kg2_entities: Sequence[int],
    aligned_pairs: Sequence[Link],
    rng: np.random.Generator,
    dry_run: bool,
) -> Tuple[Dict[str, Path], Dict[str, Any], Dict[str, Any]]:
    candidate_entities = entities_for_side(kg1_entities, kg2_entities, args.kg_side)
    feature_map_for_meta: Dict[str, Path] = {}
    log: Dict[str, Any] = {
        "modality_missing": {},
        "entity_attribute_mismatch": {},
        "attribute_attribute_inconsistency": {},
    }
    stats: Dict[str, Any] = {
        "modality_missing": {},
        "entity_attribute_mismatch": {},
        "attribute_attribute_inconsistency": {},
    }

    for modality in args.modalities:
        input_path = feature_files.get(modality)
        if input_path is None:
            continue

        resource_kind = "feature_file"
        if input_path == RAW_MODALITY_MARKERS["att"]:
            resource_kind = "training_attrs"
            ent2id_1 = read_ent_id_map(args.data_dir_path / "ent_ids_1")
            ent2id_2 = read_ent_id_map(args.data_dir_path / "ent_ids_2")
            rows_1, labels_1 = read_attr_rows(args.data_dir_path / "training_attrs_1", ent2id_1)
            rows_2, labels_2 = read_attr_rows(args.data_dir_path / "training_attrs_2", ent2id_2)
            feature = {**rows_1, **rows_2}
            original_feature = copy_feature(feature)
            output_path = output_dir / "training_attrs_1"
            feature_map_for_meta[modality] = Path("training_attrs_1,training_attrs_2")
        elif input_path == RAW_MODALITY_MARKERS["img"]:
            resource_kind = "roboea_img_pkl"
            repo_root = repo_root_from_script()
            original_img_path = runpy_img_path(args.data_dir_path, repo_root)
            output_img_path = runpy_img_path(output_dir, repo_root)
            if output_img_path == original_img_path:
                print(
                    "[warning] output_dir matches RoboEA's original image-pkl branch; "
                    "img noise would overwrite the original pkl, so img modality is skipped. "
                    "Use an output_dir whose path does not contain FBDB15K/V1/V2."
                )
                continue
            input_path = original_img_path
            output_path = output_img_path
            feature_map_for_meta[modality] = output_path
            feature = load_feature(input_path)
            original_feature = copy_feature(feature)
        else:
            input_path = Path(input_path)
            output_path = resolve_output_path(args.data_dir_path, output_dir, input_path)
            feature_map_for_meta[modality] = output_path
            feature = load_feature(input_path)
            original_feature = copy_feature(feature)

        valid_candidates = sorted(feature_entities(feature, candidate_entities))
        missing_count = int(args.noise_ratio * len(valid_candidates))
        missing_entities = [int(e) for e in sample_without_replacement(rng, valid_candidates, missing_count)]
        for entity in missing_entities:
            set_feature_zero(feature, entity)
        log["modality_missing"][modality] = missing_entities
        stats["modality_missing"][modality] = {"count": len(missing_entities), "denominator": len(valid_candidates)}

        # Use the clean original_feature as source to avoid chained pollution.
        mismatch_entities = [int(e) for e in sample_without_replacement(rng, valid_candidates, missing_count)]
        mismatch_log = []
        for entity in mismatch_entities:
            source = random_other_entity(rng, valid_candidates, entity)
            if source is None:
                continue
            assign_feature_from(feature, original_feature, entity, source)
            mismatch_log.append({"entity": int(entity), "source_entity": int(source)})
        log["entity_attribute_mismatch"][modality] = mismatch_log
        stats["entity_attribute_mismatch"][modality] = {
            "count": len(mismatch_log),
            "denominator": len(valid_candidates),
        }

        valid_pairs = [
            (int(l), int(r))
            for l, r in aligned_pairs
            if feature_has_entity(feature, l) and feature_has_entity(feature, r)
        ]
        aa_count = int(args.noise_ratio * len(valid_pairs))
        selected_pairs = sample_without_replacement(rng, valid_pairs, aa_count)
        aa_log = []
        all_feature_entities = sorted(feature_entities(feature, sorted(set(kg1_entities).union(kg2_entities))))
        for left, right in selected_pairs:
            if args.aa_side == "left":
                corrupted_entity = left
            elif args.aa_side == "right":
                corrupted_entity = right
            else:
                corrupted_entity = left if int(rng.integers(0, 2)) == 0 else right
            source = random_other_entity(rng, all_feature_entities, corrupted_entity)
            if source is None:
                continue
            assign_feature_from(feature, original_feature, corrupted_entity, source)
            aa_log.append(
                {
                    "pair": [int(left), int(right)],
                    "corrupted_entity": int(corrupted_entity),
                    "source_entity": int(source),
                }
            )
        log["attribute_attribute_inconsistency"][modality] = aa_log
        stats["attribute_attribute_inconsistency"][modality] = {
            "count": len(aa_log),
            "denominator": len(valid_pairs),
        }

        if not dry_run:
            if resource_kind == "training_attrs":
                out_rows_1 = {entity: feature[entity] for entity in rows_1}
                out_rows_2 = {entity: feature[entity] for entity in rows_2}
                write_attr_rows(output_dir / "training_attrs_1", out_rows_1, labels_1)
                write_attr_rows(output_dir / "training_attrs_2", out_rows_2, labels_2)
            else:
                save_feature(output_path, feature, input_path)

    return feature_map_for_meta, log, stats


def add_wrong_edges(
    rng: np.random.Generator,
    original_triples: Sequence[Triple],
    remaining_triples: Sequence[Triple],
    entity_set: Sequence[int],
    add_num: int,
) -> List[Triple]:
    if add_num <= 0:
        return []
    relations = sorted({r for _, r, _ in original_triples})
    entities = list(entity_set)
    if not entities:
        print("[warning] cannot add edges because the KG entity set is empty")
        return []
    if not relations:
        print("[warning] cannot add edges because the KG relation set is empty")
        return []

    existing = set(original_triples).union(set(remaining_triples))
    max_possible = len(entities) * len(entities) * len(relations) - len(existing)
    if max_possible <= 0:
        print("[warning] cannot add edges because all possible triples already exist")
        return []
    if add_num > max_possible:
        print(f"[warning] requested {add_num} added edges but only {max_possible} unique triples are possible; cap it")
        add_num = max_possible

    added: List[Triple] = []
    added_set = set()
    max_attempts = max(1000, add_num * 100)
    attempts = 0
    while len(added) < add_num and attempts < max_attempts:
        attempts += 1
        h = int(entities[int(rng.integers(0, len(entities)))])
        r = int(relations[int(rng.integers(0, len(relations)))])
        t = int(entities[int(rng.integers(0, len(entities)))])
        triple = (h, r, t)
        if triple in existing or triple in added_set:
            continue
        added.append(triple)
        added_set.add(triple)

    if len(added) < add_num:
        # Deterministic fallback for dense graphs.
        for h in entities:
            for r in relations:
                for t in entities:
                    triple = (int(h), int(r), int(t))
                    if triple in existing or triple in added_set:
                        continue
                    added.append(triple)
                    added_set.add(triple)
                    if len(added) == add_num:
                        return added
    return added


def corrupt_edges(
    args: argparse.Namespace,
    output_dir: Path,
    triples_by_name: Mapping[str, Sequence[Triple]],
    kg1_entities: Sequence[int],
    kg2_entities: Sequence[int],
    rng: np.random.Generator,
    dry_run: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    log = {
        "edge_dropout": {"triples_1": [], "triples_2": []},
        "edge_addition": {"triples_1": [], "triples_2": []},
    }
    stats = {
        "edge_dropout": {},
        "edge_addition": {},
        "final_triples": {},
    }

    for name in triples_names_for_side(args.kg_side):
        triples = list(triples_by_name.get(name, []))
        original_num_edges = len(triples)
        drop_num = int(args.noise_ratio * original_num_edges)
        drop_indices = set(int(i) for i in rng.choice(original_num_edges, size=drop_num, replace=False)) if drop_num else set()
        dropped = [triples[i] for i in sorted(drop_indices)]
        remaining = [triple for i, triple in enumerate(triples) if i not in drop_indices]

        entity_set = kg1_entities if name == "triples_1" else kg2_entities
        add_num = int(args.noise_ratio * original_num_edges)
        added = add_wrong_edges(rng, triples, remaining, entity_set, add_num)
        final_triples = remaining + added

        log["edge_dropout"][name] = [[int(h), int(r), int(t)] for h, r, t in dropped]
        log["edge_addition"][name] = [[int(h), int(r), int(t)] for h, r, t in added]
        stats["edge_dropout"][name] = len(dropped)
        stats["edge_addition"][name] = len(added)
        stats["final_triples"][name] = len(final_triples)

        if not dry_run:
            write_triples(output_dir / name, final_triples)

    for name in {"triples_1", "triples_2"} - set(triples_names_for_side(args.kg_side)):
        triples = list(triples_by_name.get(name, []))
        stats["edge_dropout"][name] = 0
        stats["edge_addition"][name] = 0
        stats["final_triples"][name] = len(triples)

    return log, stats


def print_stats(output_dir: Path, noise_ratio: float, feature_stats: Mapping[str, Any], edge_stats: Mapping[str, Any]) -> None:
    print("\nFull-MHN preprocessing summary")
    print(f"output_dir: {output_dir}")
    print(f"noise_ratio: {noise_ratio}")

    for section, title in [
        ("modality_missing", "Modality Missing"),
        ("entity_attribute_mismatch", "Entity-Attribute Mismatch"),
        ("attribute_attribute_inconsistency", "Attribute-Attribute Inconsistency"),
    ]:
        section_stats = feature_stats.get(section, {})
        if not section_stats:
            print(f"{title}: no available modalities")
            continue
        for modality, values in section_stats.items():
            count = int(values["count"])
            denominator = int(values["denominator"])
            ratio = count / denominator if denominator else 0.0
            unit = "pairs" if section == "attribute_attribute_inconsistency" else "entities"
            print(f"{title} [{modality}]: {count}/{denominator} {unit} ({ratio:.4f})")

    for name in ["triples_1", "triples_2"]:
        print(f"{name} dropped edges: {edge_stats['edge_dropout'].get(name, 0)}")
        print(f"{name} added edges: {edge_stats['edge_addition'].get(name, 0)}")
        print(f"{name} final triples: {edge_stats['final_triples'].get(name, 0)}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a fixed Full-MHN noisy dataset for RoboEA.")
    parser.add_argument("--data_dir", type=str, required=True, help="Original dataset directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output noisy dataset directory")
    parser.add_argument("--noise_ratio", type=float, required=True, help="Noise ratio, e.g. 0.2, 0.4, 0.6")
    parser.add_argument("--modalities", type=str, default="img,att", help="Modalities to corrupt")
    parser.add_argument("--feature_map", type=str, default=None, help="Explicit modality:file mapping")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    parser.add_argument("--kg_side", choices=["both", "kg1", "kg2"], default="both")
    parser.add_argument("--aa_side", choices=["right", "left", "random"], default="right")
    parser.add_argument("--links_file", type=str, default=None, help="Aligned pairs file")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output_dir if it exists")
    parser.add_argument("--dry_run", action="store_true", help="Only print statistics; do not copy or write files")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.data_dir_path = Path(args.data_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not 0.0 <= args.noise_ratio <= 1.0:
        raise ValueError("--noise_ratio must be in [0, 1]")
    if not args.data_dir_path.exists() or not args.data_dir_path.is_dir():
        raise FileNotFoundError(f"--data_dir does not exist or is not a directory: {args.data_dir_path}")

    args.modalities = parse_modalities(args.modalities)
    explicit_feature_map = parse_feature_map(args.feature_map)

    triples_paths = {name: args.data_dir_path / name for name in ["triples_1", "triples_2"]}
    for name, path in triples_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Required triples file not found: {path}")
    triples_by_name = {name: read_triples(path) for name, path in triples_paths.items()}
    kg1_entities, kg2_entities = get_kg_entities(args.data_dir_path, triples_by_name)
    if not kg1_entities:
        raise ValueError("KG1 entity set is empty; cannot generate Full-MHN")
    if not kg2_entities:
        raise ValueError("KG2 entity set is empty; cannot generate Full-MHN")

    links_path = find_links_file(args.data_dir_path, args.links_file)
    if links_path is None:
        print("[warning] no aligned pairs file found; Attribute-Attribute Inconsistency will be skipped")
        aligned_pairs: List[Link] = []
    else:
        aligned_pairs = read_links(links_path)

    feature_files = discover_feature_files(args.data_dir_path, output_dir, args.modalities, explicit_feature_map)
    rng = np.random.default_rng(args.seed)

    if not args.dry_run:
        if output_dir.exists():
            if not args.overwrite:
                raise FileExistsError(f"output_dir already exists: {output_dir}. Use --overwrite to replace it.")
            shutil.rmtree(output_dir)
        shutil.copytree(args.data_dir_path, output_dir)

    feature_map_for_meta, feature_log, feature_stats = corrupt_features(
        args,
        feature_files,
        output_dir,
        kg1_entities,
        kg2_entities,
        aligned_pairs,
        rng,
        args.dry_run,
    )
    edge_log, edge_stats = corrupt_edges(
        args,
        output_dir,
        triples_by_name,
        kg1_entities,
        kg2_entities,
        rng,
        args.dry_run,
    )

    noise_log = {
        **feature_log,
        **edge_log,
    }
    meta = {
        "noise_name": "Full-MHN",
        "noise_ratio": args.noise_ratio,
        "modalities": args.modalities,
        "feature_map": {modality: str(path) for modality, path in feature_map_for_meta.items()},
        "seed": args.seed,
        "kg_side": args.kg_side,
        "aa_side": args.aa_side,
        "data_dir": str(args.data_dir_path),
        "output_dir": str(output_dir),
    }

    if not args.dry_run:
        with (output_dir / "full_mhn_meta.json").open("w", encoding="utf-8") as fw:
            json.dump(meta, fw, indent=2, ensure_ascii=False)
        with (output_dir / "full_mhn_noise_log.json").open("w", encoding="utf-8") as fw:
            json.dump(noise_log, fw, indent=2, ensure_ascii=False)

    print_stats(output_dir, args.noise_ratio, feature_stats, edge_stats)
    if args.dry_run:
        print("dry_run: no files were copied or written")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1)
