import math
from dataclasses import dataclass
from typing import List


@dataclass
class Candidate:
    x: float
    y: float
    distance_to_origin: float
    embedding: object = None
    similarity: float = None
    score: float = None


def generate_candidate_grid(
    origin_x: float,
    origin_y: float,
    radius_px: int,
    step_px: int,
) -> List[Candidate]:
    candidates = []
    for dy in range(-radius_px, radius_px + 1, step_px):
        for dx in range(-radius_px, radius_px + 1, step_px):
            x = origin_x + dx
            y = origin_y + dy
            distance = math.sqrt(dx * dx + dy * dy)
            candidates.append(Candidate(x=x, y=y, distance_to_origin=distance))
    return candidates