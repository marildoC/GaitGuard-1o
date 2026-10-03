
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

from face.multiview_types import MultiViewConfig
from face.multiview_gallery_view import MultiViewGalleryView

logger = logging.getLogger(__name__)


try:
    import faiss
except Exception as exc:
    faiss = None
    _faiss_error = exc
else:
    _faiss_error = None


MatchStrength = Literal["strong", "weak", "none"]


@dataclass
class MultiViewCandidate:
    """
    One candidate match from the multi-view database.
    """

    person_id: str
    distance: float
    score: float
    pose_bin: str
    centroid_quality: float


@dataclass
class MultiViewMatchResult:
    """
    Result of matching a single live face observation against the pseudo-3D
    head database.

    If strength == "none", `best` will be None and `candidates` may be empty.
    """

    strength: MatchStrength
    best: Optional[MultiViewCandidate]
    candidates: List[MultiViewCandidate]
    pose_bin_used: str
    considered_bins: List[str]
    query_quality: float
    yaw: float
    pitch: float




@dataclass
class _RowMeta:
    person_id: str
    pose_bin: str
    centroid_quality: float


class _BinIndex:
    """
    Simple container for one pose-bin index (or the global index).

    Holds:
      - embeddings: (N, dim) float32, L2-normalised.
      - rows: per-row metadata.
      - faiss_index: optional; if FAISS is unavailable we use NumPy.
    """

    def __init__(self) -> None:
        self.embeddings: Optional[np.ndarray] = None
        self.rows: List[_RowMeta] = []
        self.faiss_index: Any = None

    @property
    def size(self) -> int:
        if self.embeddings is None:
            return 0
        return int(self.embeddings.shape[0])




class MultiViewMatcher:
    """
    Pose-aware matcher over MultiViewGalleryView.

    Typical usage (per process):

        mv_view = MultiViewGalleryView(gallery, multiview_cfg)
        matcher = MultiViewMatcher(mv_view, multiview_cfg)

        # After any enrollment/reset:
        mv_view.mark_dirty()
        matcher.mark_dirty()

        # For each live face sample:
        result = matcher.match(
            embedding=emb, yaw=yaw, pitch=pitch, quality=quality
        )

    Design notes:

      - This matcher is *stateless* between calls, except for its cached
        FAISS/NumPy indices. Temporal smoothing is handled by an identity
        engine on top of this.
      - It never modifies the gallery; if anything goes wrong here, the
        classic flat gallery pipeline still works unchanged.
    """

    def __init__(
        self,
        mv_gallery: MultiViewGalleryView,
        multiview_cfg: Optional[MultiViewConfig] = None,
    ) -> None:
        self._mv_gallery = mv_gallery
        self._cfg: MultiViewConfig = multiview_cfg or MultiViewConfig()

        self._bin_indices: Dict[str, _BinIndex] = {}

        self._global_index: _BinIndex = _BinIndex()

        self._dirty: bool = True

        self._metric: str = "cosine"

        self._dim: int = int(getattr(self._cfg, "dim", 512))

        if hasattr(self._cfg, "min_quality_for_match"):
            self._min_quality_for_match: float = float(
                getattr(self._cfg, "min_quality_for_match")
            )
        else:
            self._min_quality_for_match = float(
                getattr(self._cfg, "min_quality", 0.3)
            )

        if hasattr(self._cfg, "strong_distance"):
            self._strong_distance: float = float(
                getattr(self._cfg, "strong_distance")
            )
        else:
            self._strong_distance = float(
                getattr(self._cfg, "strong_dist", 0.35)
            )

        if hasattr(self._cfg, "weak_distance"):
            self._weak_distance: float = float(getattr(self._cfg, "weak_distance"))
        else:
            self._weak_distance = float(getattr(self._cfg, "weak_dist", 0.55))

        self._enable_matching: bool = bool(
            getattr(self._cfg, "enable_matching", True)
        )

        logger.info(
            "MultiViewMatcher initialised | dim=%d, q_match>=%.3f, "
            "strong_d=%.3f, weak_d=%.3f, enable_matching=%s",
            self._dim,
            self._min_quality_for_match,
            self._strong_distance,
            self._weak_distance,
            self._enable_matching,
        )


    def mark_dirty(self) -> None:
        """
        Mark internal indices as stale.

        Call this *whenever* templates in the underlying FaceGallery
        change (enrollment, reset, load, etc.) along with
        mv_gallery.mark_dirty().
        """
        self._dirty = True

    def _ensure_indices(self) -> None:
        """
        Lazily rebuild bin/global indices if marked dirty.
        """
        if not self._dirty:
            return

        self._mv_gallery.refresh_if_needed()

        models = self._mv_gallery.get_all_models()
        if not models:
            self._bin_indices.clear()
            self._global_index = _BinIndex()
            self._dirty = False
            logger.info("MultiViewMatcher: no models available, indices cleared.")
            return

        per_bin_vectors: Dict[str, List[np.ndarray]] = {}
        per_bin_meta: Dict[str, List[_RowMeta]] = {}
        global_vectors: List[np.ndarray] = []
        global_meta: List[_RowMeta] = []

        for pid, model in models.items():
            for bin_name, bin_state in model.bins.items():
                try:
                    num = int(bin_state.num_samples())
                except TypeError:
                    logger.warning(
                        "MultiViewMatcher: bin_state.num_samples not callable for "
                        "pid=%s bin=%s; skipping bin.",
                        pid,
                        bin_name,
                    )
                    continue

                if bin_state.centroid is None or num <= 0:
                    continue

                centroid = np.asarray(
                    bin_state.centroid, dtype=np.float32
                ).reshape(-1)

                if centroid.size != self._dim:
                    logger.warning(
                        "MultiViewMatcher: centroid dim mismatch for pid=%s bin=%s "
                        "(expected %d, got %d); skipping this bin.",
                        pid,
                        bin_name,
                        self._dim,
                        centroid.size,
                    )
                    continue

                norm = float(np.linalg.norm(centroid))
                if norm > 1e-6:
                    centroid = centroid / norm
                else:
                    logger.debug(
                        "MultiViewMatcher: degenerate centroid for pid=%s bin=%s; "
                        "skipping.",
                        pid,
                        bin_name,
                    )
                    continue

                meta = _RowMeta(
                    person_id=pid,
                    pose_bin=str(bin_name),
                    centroid_quality=float(getattr(bin_state, "avg_quality", 0.0)),
                )

                bin_key = str(bin_name)
                per_bin_vectors.setdefault(bin_key, []).append(centroid)
                per_bin_meta.setdefault(bin_key, []).append(meta)

                global_vectors.append(centroid)
                global_meta.append(meta)

        self._bin_indices = {}
        for bin_name, vecs in per_bin_vectors.items():
            idx = _BinIndex()
            idx.embeddings = np.stack(vecs, axis=0)
            idx.rows = per_bin_meta[bin_name]
            idx.faiss_index = self._build_faiss_index(idx.embeddings)
            self._bin_indices[bin_name] = idx

        self._global_index = _BinIndex()
        if global_vectors:
            self._global_index.embeddings = np.stack(global_vectors, axis=0)
            self._global_index.rows = global_meta
            self._global_index.faiss_index = self._build_faiss_index(
                self._global_index.embeddings
            )

        self._dirty = False

        logger.info(
            "MultiViewMatcher: indices rebuilt | persons=%d, centroids=%d, bins=%d",
            len(models),
            self._global_index.size,
            len(self._bin_indices),
        )

    def _build_faiss_index(self, embeddings: np.ndarray) -> Any:
        """
        Build a FAISS index for the given embeddings, or return None if
        FAISS is unavailable.

        Embeddings MUST be float32 and L2-normalised if metric == "cosine".
        """
        if embeddings is None or embeddings.size == 0:
            return None

        if faiss is None:
            if _faiss_error is not None:
                logger.debug(
                    "MultiViewMatcher: FAISS not available (%s); using NumPy search.",
                    _faiss_error,
                )
            return None

        dim = int(embeddings.shape[1])
        if dim != self._dim:
            logger.warning(
                "MultiViewMatcher: FAISS index dim mismatch (expected %d, got %d); "
                "proceeding but this should be investigated.",
                self._dim,
                dim,
            )

        if self._metric == "cosine":
            index = faiss.IndexFlatIP(dim)
        else:
            index = faiss.IndexFlatL2(dim)

        index.add(embeddings.astype(np.float32))
        return index


    def match(
        self,
        embedding: np.ndarray,
        *,
        yaw: float,
        pitch: float,
        quality: float,
        top_k: int = 5,
    ) -> MultiViewMatchResult:
        """
        Match a single live face sample to the multi-view gallery.

        Parameters
        ----------
        embedding:
            Raw 1-D face embedding (will be float32 + L2-normalised here).
        yaw, pitch:
            Pose angles (degrees) estimated by the face route.
        quality:
            Scalar quality estimate from the face quality module.
        top_k:
            How many candidates to return in the diagnostics list.
        """
        self._ensure_indices()

        if self._global_index.size == 0:
            return MultiViewMatchResult(
                strength="none",
                best=None,
                candidates=[],
                pose_bin_used="none",
                considered_bins=[],
                query_quality=float(quality),
                yaw=float(yaw),
                pitch=float(pitch),
            )

        if quality < self._min_quality_for_match:
            return MultiViewMatchResult(
                strength="none",
                best=None,
                candidates=[],
                pose_bin_used="low_quality_skip",
                considered_bins=[],
                query_quality=float(quality),
                yaw=float(yaw),
                pitch=float(pitch),
            )

        q = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if q.size != self._dim:
            raise ValueError(
                f"MultiViewMatcher: embedding dim mismatch "
                f"(expected {self._dim}, got {q.size})."
            )
        norm = float(np.linalg.norm(q))
        if norm > 1e-6:
            q = q / norm
        else:
            return MultiViewMatchResult(
                strength="none",
                best=None,
                candidates=[],
                pose_bin_used="degenerate_embedding",
                considered_bins=[],
                query_quality=float(quality),
                yaw=float(yaw),
                pitch=float(pitch),
            )

        if hasattr(self._cfg, "choose_bin"):
            primary_bin = str(self._cfg.choose_bin(yaw=yaw, pitch=pitch))
        else:
            primary_bin = "FRONT"

        considered_bins: List[str] = [primary_bin]

        bin_idx = self._bin_indices.get(primary_bin)
        if bin_idx is None or bin_idx.size == 0:
            candidates = self._search_index(self._global_index, q, top_k=top_k)
            pose_bin_used = "global_fallback"
        else:
            candidates = self._search_index(bin_idx, q, top_k=top_k)
            if not candidates:
                candidates = self._search_index(self._global_index, q, top_k=top_k)
                pose_bin_used = "global_fallback"
            else:
                pose_bin_used = primary_bin

        if not candidates:
            return MultiViewMatchResult(
                strength="none",
                best=None,
                candidates=[],
                pose_bin_used=pose_bin_used,
                considered_bins=considered_bins,
                query_quality=float(quality),
                yaw=float(yaw),
                pitch=float(pitch),
            )

        best = candidates[0]

        strength = self._classify_strength(best.distance, best.score)

        if strength == "none":
            return MultiViewMatchResult(
                strength="none",
                best=None,
                candidates=candidates,
                pose_bin_used=pose_bin_used,
                considered_bins=considered_bins,
                query_quality=float(quality),
                yaw=float(yaw),
                pitch=float(pitch),
            )

        return MultiViewMatchResult(
            strength=strength,
            best=best,
            candidates=candidates,
            pose_bin_used=pose_bin_used,
            considered_bins=considered_bins,
            query_quality=float(quality),
            yaw=float(yaw),
            pitch=float(pitch),
        )


    def _search_index(
        self,
        index: _BinIndex,
        q: np.ndarray,
        *,
        top_k: int,
    ) -> List[MultiViewCandidate]:
        """
        Search one index (pose bin or global) and return top_k candidates.
        """
        if index.size == 0 or index.embeddings is None:
            return []

        k = max(1, min(top_k, index.size))
        E = index.embeddings

        if index.faiss_index is not None:
            q_batch = q.reshape(1, -1)
            if self._metric == "cosine":
                sims, idxs = index.faiss_index.search(q_batch, k)
                sims = sims[0]
                idxs = idxs[0]
                results: List[MultiViewCandidate] = []
                for sim, row_idx in zip(sims, idxs):
                    row = index.rows[int(row_idx)]
                    distance = 1.0 - float(sim)
                    score = max(0.0, float(sim))
                    results.append(
                        MultiViewCandidate(
                            person_id=row.person_id,
                            distance=distance,
                            score=score,
                            pose_bin=row.pose_bin,
                            centroid_quality=row.centroid_quality,
                        )
                    )
            else:
                dists, idxs = index.faiss_index.search(q_batch, k)
                dists = dists[0]
                idxs = idxs[0]
                results = []
                for dist, row_idx in zip(dists, idxs):
                    row = index.rows[int(row_idx)]
                    distance = float(dist)
                    score = 1.0 / (1.0 + distance)
                    results.append(
                        MultiViewCandidate(
                            person_id=row.person_id,
                            distance=distance,
                            score=score,
                            pose_bin=row.pose_bin,
                            centroid_quality=row.centroid_quality,
                        )
                    )
        else:
            if self._metric == "cosine":
                sims = (E @ q.reshape(-1, 1)).reshape(-1)
                order = np.argsort(-sims)[:k]
                results = []
                for row_idx in order:
                    sim = float(sims[row_idx])
                    row = index.rows[int(row_idx)]
                    distance = 1.0 - sim
                    score = max(0.0, sim)
                    results.append(
                        MultiViewCandidate(
                            person_id=row.person_id,
                            distance=distance,
                            score=score,
                            pose_bin=row.pose_bin,
                            centroid_quality=row.centroid_quality,
                        )
                    )
            else:
                diffs = E - q.reshape(1, -1)
                dists = np.sum(diffs * diffs, axis=1)
                order = np.argsort(dists)[:k]
                results = []
                for row_idx in order:
                    dist = float(dists[row_idx])
                    row = index.rows[int(row_idx)]
                    distance = dist
                    score = 1.0 / (1.0 + distance)
                    results.append(
                        MultiViewCandidate(
                            person_id=row.person_id,
                            distance=distance,
                            score=score,
                            pose_bin=row.pose_bin,
                            centroid_quality=row.centroid_quality,
                        )
                    )

        return results


    def _classify_strength(self, distance: float, score: float) -> MatchStrength:
        """
        Map numeric distance/score into one of three discrete classes:
        "strong", "weak", or "none".

        Uses thresholds from MultiViewConfig, defined in terms of distance
        (smaller is better). For cosine we typically have:
            strong if d <= strong_dist
            weak   if d <= weak_dist
            none   otherwise
        """
        if not self._enable_matching:
            return "none"

        strong_d = self._strong_distance
        weak_d = self._weak_distance

        if distance <= strong_d:
            return "strong"
        if distance <= weak_d:
            return "weak"
        return "none"
