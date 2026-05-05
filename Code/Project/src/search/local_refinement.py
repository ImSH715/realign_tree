import math
from typing import List

from src.search.candidate_grid import Candidate, generate_candidate_grid


def build_refinement_candidates(
    coarse_x: float,
    coarse_y: float,
    original_x: float,
    original_y: float,
    radius_px: int,
    step_px: int,
) -> List[Candidate]:
    candidates = generate_candidate_grid(
        origin_x=coarse_x,
        origin_y=coarse_y,
        radius_px=radius_px,
        step_px=step_px,
    )

    for c in candidates:
        c.distance_to_origin = math.sqrt((c.x - original_x) ** 2 + (c.y - original_y) ** 2)

    return candidates