
from __future__ import annotations

import logging
from typing import Iterable, List, Optional

import numpy as np

from .config import FaceConfig, default_face_config

logger = logging.getLogger(__name__)


class FaceEmbedder:
    """
    Compatibility / utility wrapper for face embeddings.

    Responsibilities now:

    - Provide a stable .dim property (usually 512).
    - Provide embed(...) / embed_many(...) that accept *embedding-like*
      inputs (1-D arrays) and:
        * cast to float32,
        * flatten to 1-D,
        * L2-normalise,
        * adjust dim if necessary (with a warning).

    It NO LONGER performs model inference on raw images. If you pass in
    an HxWx3 array, it will raise a RuntimeError and remind you that
    embeddings must come from FaceDetectorAligner (buffalo_l).
    """

    def __init__(self, cfg: Optional[FaceConfig] = None) -> None:
        self.cfg = cfg or default_face_config()
        self._dim = int(self.cfg.gallery.dim)

        logger.info(
            "FaceEmbedder initialised in passthrough mode "
            "(expects precomputed embeddings, dim=%d).",
            self._dim,
        )


    @property
    def dim(self) -> int:
        """
        Current expected embedding dimensionality.

        Note: This may be updated at runtime if we observe a different
        dimension in incoming vectors. This mirrors how the rest of the
        system is tolerant to model swaps (e.g. different ArcFace variants).
        """
        return self._dim


    def warmup(self) -> None:
        """
        Kept for API compatibility. Does nothing in passthrough mode.
        """
        return

    def _handle_dim_mismatch(self, observed_dim: int) -> None:
        """
        Handle the case where an incoming embedding has a different
        dimensionality than currently expected.

        For robustness, we:
          - log a warning,
          - update internal _dim to the new observed dimension.

        This avoids hard crashes if the embedding backend is changed
        (e.g. to a 256-D model) while keeping the rest of the system
        able to adapt.
        """
        if observed_dim == self._dim:
            return

        logger.warning(
            "FaceEmbedder received embedding with dim=%d, expected %d. "
            "Updating internal dim to %d. Make sure gallery/config are "
            "consistent with the embedding backend.",
            observed_dim,
            self._dim,
            observed_dim,
        )
        self._dim = observed_dim

    def _ensure_embedding_1d(self, vector: np.ndarray) -> np.ndarray:
        """
        Ensure vector is a 1-D float32 embedding, L2-normalised.

        If the size doesn't match cfg.gallery.dim, we log a warning and
        update self._dim to the new observed size, then continue.
        """
        emb = np.asarray(vector, dtype=np.float32).reshape(-1)

        if emb.size != self._dim:
            self._handle_dim_mismatch(emb.size)

        norm = float(np.linalg.norm(emb))
        if norm > 1e-6:
            emb /= norm
        else:
            emb[:] = 0.0

        return emb


    def embed(self, vector_or_image: np.ndarray) -> np.ndarray:
        """
        Normalise a precomputed embedding.

        Parameters
        ----------
        vector_or_image : np.ndarray
            Expected to be a 1-D or (N, 1)/(1, N) array containing a
            face embedding. If an HxWx3 image is passed, this method
            will raise an error because model inference is no longer
            supported here.

        Returns
        -------
        np.ndarray
            1-D float32, L2-normalised embedding.
        """
        arr = np.asarray(vector_or_image)

        if arr.ndim == 3 and arr.shape[2] == 3:
            raise RuntimeError(
                "FaceEmbedder.embed no longer supports raw image inputs. "
                "Embeddings must be produced by FaceDetectorAligner "
                "(buffalo_l) and then passed here only if you need "
                "normalisation."
            )

        return self._ensure_embedding_1d(arr)

    def embed_many(self, vectors: Iterable[np.ndarray]) -> np.ndarray:
        """
        Normalise multiple precomputed embeddings.

        Parameters
        ----------
        vectors : Iterable[np.ndarray]
            Iterable of embedding-like arrays (1-D).

        Returns
        -------
        np.ndarray
            Array of shape (num_vectors, dim), each row L2-normalised.
            If no vectors are provided, returns an empty array with
            shape (0, dim).
        """
        embs: List[np.ndarray] = []

        for idx, v in enumerate(vectors):
            try:
                embs.append(self.embed(v))
            except Exception as exc:
                logger.warning(
                    "Failed to normalise one embedding in embed_many (index=%d): %s",
                    idx,
                    exc,
                )
                embs.append(np.zeros(self._dim, dtype=np.float32))

        if not embs:
            return np.zeros((0, self._dim), dtype=np.float32)

        return np.stack(embs, axis=0)


    def is_compatible_dim(self, vector: np.ndarray) -> bool:
        """
        Quick check: does this vector have the same dimensionality as
        the current embedder?

        This is a non-critical helper that can be used by other parts
        of the system (e.g. diagnostics) to verify that stored gallery
        embeddings and runtime embeddings agree on dim.
        """
        arr = np.asarray(vector).reshape(-1)
        return int(arr.size) == int(self._dim)
