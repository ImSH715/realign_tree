from dataclasses import dataclass
from typing import List, Optional

from tqdm import tqdm

from src.search.bounded_region import build_bounded_region
from src.search.candidate_grid import generate_candidate_grid
from src.search.local_refinement import build_refinement_candidates
from src.scoring.ranker import score_candidates


@dataclass
class RefinementResult:
    point_id: str
    image_path: str
    original_x: float
    original_y: float
    refined_x: float
    refined_y: float
    target_label: Optional[str]
    best_score: float
    best_similarity: float
    coarse_x: float
    coarse_y: float
    coarse_score: float
    status: str


class FeatureGuidedBoundedSearchPipeline:
    def __init__(
        self,
        encoder,
        prototypes,
        patch_extractor,
        similarity_mode: str,
        search_radius_px: int,
        coarse_step_px: int,
        refine_radius_px: int,
        refine_step_px: int,
        alpha: float,
        beta: float,
        batch_size: int,
    ):
        self.encoder = encoder
        self.prototypes = prototypes
        self.patch_extractor = patch_extractor
        self.similarity_mode = similarity_mode
        self.search_radius_px = search_radius_px
        self.coarse_step_px = coarse_step_px
        self.refine_radius_px = refine_radius_px
        self.refine_step_px = refine_step_px
        self.alpha = alpha
        self.beta = beta
        self.batch_size = batch_size

    def _embed_candidates(self, image_path: str, candidates):
        patches = [
            self.patch_extractor.extract(
                image_path=image_path,
                center_x=c.x,
                center_y=c.y,
            )
            for c in candidates
        ]

        embeddings = self.encoder.encode_batch(patches, batch_size=self.batch_size)

        for c, emb in zip(candidates, embeddings):
            c.embedding = emb

        return candidates

    def _coarse_search(self, point):
        _ = build_bounded_region(
            center_x=point.x,
            center_y=point.y,
            radius_px=self.search_radius_px,
        )

        candidates = generate_candidate_grid(
            origin_x=point.x,
            origin_y=point.y,
            radius_px=self.search_radius_px,
            step_px=self.coarse_step_px,
        )
        candidates = self._embed_candidates(point.image_path, candidates)
        candidates = score_candidates(
            candidates=candidates,
            target_label=point.target_label,
            prototypes=self.prototypes,
            similarity_mode=self.similarity_mode,
            alpha=self.alpha,
            beta=self.beta,
        )
        return candidates[0]

    def _refine_search(self, point, coarse_best):
        candidates = build_refinement_candidates(
            coarse_x=coarse_best.x,
            coarse_y=coarse_best.y,
            original_x=point.x,
            original_y=point.y,
            radius_px=self.refine_radius_px,
            step_px=self.refine_step_px,
        )
        candidates = self._embed_candidates(point.image_path, candidates)
        candidates = score_candidates(
            candidates=candidates,
            target_label=point.target_label,
            prototypes=self.prototypes,
            similarity_mode=self.similarity_mode,
            alpha=self.alpha,
            beta=self.beta,
        )
        return candidates[0]

    def run_point(self, point):
        coarse_best = self._coarse_search(point)
        final_best = self._refine_search(point, coarse_best)

        return RefinementResult(
            point_id=point.point_id,
            image_path=point.image_path,
            original_x=point.x,
            original_y=point.y,
            refined_x=final_best.x,
            refined_y=final_best.y,
            target_label=point.target_label,
            best_score=float(final_best.score),
            best_similarity=float(final_best.similarity),
            coarse_x=float(coarse_best.x),
            coarse_y=float(coarse_best.y),
            coarse_score=float(coarse_best.score),
            status="ok",
        )

    def run(self, points: List):
        results = []
        for point in tqdm(points, desc="Refining points", dynamic_ncols=True):
            try:
                results.append(self.run_point(point))
            except Exception as e:
                results.append(
                    RefinementResult(
                        point_id=point.point_id,
                        image_path=point.image_path,
                        original_x=point.x,
                        original_y=point.y,
                        refined_x=point.x,
                        refined_y=point.y,
                        target_label=point.target_label,
                        best_score=float("nan"),
                        best_similarity=float("nan"),
                        coarse_x=point.x,
                        coarse_y=point.y,
                        coarse_score=float("nan"),
                        status=f"failed: {e}",
                    )
                )
        return results