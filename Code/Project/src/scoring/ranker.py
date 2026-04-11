from typing import Dict, List, Tuple

import numpy as np

from src.scoring.similarity import cosine_similarity_matrix, euclidean_similarity_matrix


def build_internal_to_semantic_mapping(
    phase1_rows: List[dict],
    phase1_embeddings: np.ndarray,
    prototypes: Dict[str, np.ndarray],
    similarity: str = "cosine",
) -> Dict[str, str]:
    proto_labels = sorted(prototypes.keys())
    proto_matrix = np.stack([prototypes[l] for l in proto_labels], axis=0)

    if similarity == "cosine":
        sims = cosine_similarity_matrix(phase1_embeddings, proto_matrix)
    elif similarity == "euclidean":
        sims = euclidean_similarity_matrix(phase1_embeddings, proto_matrix)
    else:
        raise ValueError("similarity must be one of: cosine, euclidean")

    internal_votes: Dict[str, List[str]] = {}

    for row, sim_vec in zip(phase1_rows, sims):
        internal_pred = row.get("pred_name", row.get("target_name", "unknown"))
        best_idx = int(np.argmax(sim_vec))
        semantic_label = proto_labels[best_idx]
        internal_votes.setdefault(internal_pred, []).append(semantic_label)

    mapping = {}
    for internal_pred, labels in internal_votes.items():
        counts: Dict[str, int] = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
        best_label = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0]
        mapping[internal_pred] = best_label

    return mapping


def correct_labels(
    phase1_rows: List[dict],
    phase1_embeddings: np.ndarray,
    prototypes: Dict[str, np.ndarray],
    similarity: str = "cosine",
    confidence_threshold: float = 0.0,
    correction_margin: float = 0.0,
) -> Tuple[List[dict], Dict[str, float]]:
    proto_labels = sorted(prototypes.keys())
    proto_matrix = np.stack([prototypes[l] for l in proto_labels], axis=0)

    if similarity == "cosine":
        sim_matrix = cosine_similarity_matrix(phase1_embeddings, proto_matrix)
    elif similarity == "euclidean":
        sim_matrix = euclidean_similarity_matrix(phase1_embeddings, proto_matrix)
    else:
        raise ValueError("similarity must be one of: cosine, euclidean")

    corrected_rows = []
    num_changed = 0

    for row, sim_vec in zip(phase1_rows, sim_matrix):
        best_idx = int(np.argmax(sim_vec))
        best_label = proto_labels[best_idx]
        best_score = float(sim_vec[best_idx])

        second_score = float(np.partition(sim_vec, -2)[-2]) if len(sim_vec) > 1 else best_score
        margin = best_score - second_score

        original_pred = row.get("pred_name", row.get("target_name", "unknown"))
        phase1_conf = float(row.get("confidence", 1.0))

        should_correct = True
        if phase1_conf < confidence_threshold:
            should_correct = False
        if margin < correction_margin:
            should_correct = False

        corrected_label = best_label if should_correct else original_pred

        if corrected_label != original_pred:
            num_changed += 1

        new_row = dict(row)
        new_row["prototype_best_label"] = best_label
        new_row["prototype_best_score"] = best_score
        new_row["prototype_margin"] = margin
        new_row["corrected_label"] = corrected_label
        new_row["correction_applied"] = int(corrected_label != original_pred)
        corrected_rows.append(new_row)

    summary = {
        "total_samples": float(len(phase1_rows)),
        "num_changed": float(num_changed),
        "change_ratio": float(num_changed / max(1, len(phase1_rows))),
    }
    return corrected_rows, summary


def score_candidates(
    candidates,
    target_label: str,
    prototypes: Dict[str, np.ndarray],
    similarity_mode: str = "cosine",
    alpha: float = 1.0,
    beta: float = 0.002,
):
    if target_label is None:
        for c in candidates:
            c.similarity = 0.0
            c.score = float(-beta * c.distance_to_origin)
        return sorted(candidates, key=lambda c: c.score, reverse=True)

    if target_label not in prototypes:
        raise KeyError(f"Prototype not found for label: {target_label}")

    prototype = prototypes[target_label]

    for c in candidates:
        emb = c.embedding.astype(np.float32)

        if similarity_mode == "cosine":
            emb_n = emb / (np.linalg.norm(emb) + 1e-12)
            proto_n = prototype / (np.linalg.norm(prototype) + 1e-12)
            sim = float(np.dot(emb_n, proto_n))
        elif similarity_mode == "euclidean":
            sim = float(-np.linalg.norm(emb - prototype))
        else:
            raise ValueError("similarity must be one of: cosine, euclidean")

        c.similarity = sim
        c.score = float(alpha * sim - beta * c.distance_to_origin)

    return sorted(candidates, key=lambda c: c.score, reverse=True)