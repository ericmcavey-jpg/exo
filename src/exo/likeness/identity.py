"""Measure whether an edited image still looks like you.

A face-recognition embedding is computed for each image and compared, by cosine
similarity, with an averaged reference built from your own photos. The gate in
`judge_identity` turns "still looks like me" into a pass/fail rule per edit.

The InsightFace embedder is optional and loaded lazily; run the toolkit with
scripts/likeness adds it to the commands that need it. Its pretrained models are licensed
for non-commercial use, which covers personal use of your own likeness.
"""

import importlib
import math
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast, final

from exo.utils.pydantic_ext import FrozenModel

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

REFERENCE_OUTLIER_THRESHOLD: Final = 0.35
# Faces that fill the frame are missed by the detector unless the image is padded.
DETECTION_PADDING: Final = 0.25


class IdentityPolicy(FrozenModel):
    enforce: bool
    minimum_similarity: float = 0.45
    maximum_drop: float = 0.10
    # Restoration should only sharpen, so it gets a tighter tolerance than edits.
    maximum_restoration_drop: float = 0.05
    require_face: bool = True


class IdentityVerdict(FrozenModel):
    accepted: bool
    similarity: float | None
    reason: str


class FaceEmbedder(Protocol):
    @property
    def name(self) -> str: ...

    def embed_largest_face(self, image_path: Path) -> list[float] | None: ...


def normalize(vector: Sequence[float]) -> list[float]:
    length = math.sqrt(sum(value * value for value in vector))
    if length == 0:
        return [0.0 for _ in vector]
    return [value / length for value in vector]


def cosine_similarity(first: Sequence[float], second: Sequence[float]) -> float:
    if len(first) != len(second):
        raise ValueError(f"Embedding sizes differ: {len(first)} vs {len(second)}")
    return sum(
        left * right
        for left, right in zip(normalize(first), normalize(second), strict=True)
    )


def _mean(vectors: Sequence[Sequence[float]]) -> list[float]:
    return normalize(
        [sum(column) / len(vectors) for column in zip(*vectors, strict=True)]
    )


def build_reference(
    embeddings: Sequence[Sequence[float]],
    outlier_threshold: float = REFERENCE_OUTLIER_THRESHOLD,
) -> tuple[list[float], int]:
    """Average the embeddings after dropping faces that are not you (mis-tagged in Photos).

    Returns the reference embedding and how many faces were discarded.
    """
    if not embeddings:
        raise ValueError("No faces were found in the task's photos.")
    first_pass = _mean(embeddings)
    kept = [
        embedding
        for embedding in embeddings
        if cosine_similarity(embedding, first_pass) >= outlier_threshold
    ]
    if not kept:
        return first_pass, 0
    return _mean(kept), len(embeddings) - len(kept)


def required_similarity(policy: IdentityPolicy, input_similarity: float) -> float:
    """The similarity an edit must reach, given how well the unedited photo matches you."""
    if input_similarity >= policy.minimum_similarity:
        return max(policy.minimum_similarity, input_similarity - policy.maximum_drop)
    # The source is already a weak match (odd angle, harsh light): only block further drift.
    return input_similarity - policy.maximum_drop


def judge_identity(
    policy: IdentityPolicy,
    input_similarity: float | None,
    output_similarity: float | None,
) -> IdentityVerdict:
    if not policy.enforce:
        return IdentityVerdict(
            accepted=True,
            similarity=output_similarity,
            reason="identity check is report-only for this recipe",
        )
    if input_similarity is None:
        return IdentityVerdict(
            accepted=True,
            similarity=output_similarity,
            reason="no face found in the source image, so identity was not checked",
        )
    if output_similarity is None:
        return IdentityVerdict(
            accepted=not policy.require_face,
            similarity=None,
            reason="no face found in the result",
        )
    required = required_similarity(policy, input_similarity)
    if output_similarity < required:
        return IdentityVerdict(
            accepted=False,
            similarity=output_similarity,
            reason=(
                f"looks less like you: similarity {output_similarity:.2f} "
                f"is below the required {required:.2f} (source photo: {input_similarity:.2f})"
            ),
        )
    return IdentityVerdict(
        accepted=True,
        similarity=output_similarity,
        reason=f"identity kept: similarity {output_similarity:.2f} (source photo: {input_similarity:.2f})",
    )


@final
class ReferenceIdentityScorer:
    """Scores images against a stored reference embedding."""

    def __init__(
        self, embedder: FaceEmbedder, reference_embedding: list[float]
    ) -> None:
        self._embedder = embedder
        self._reference = reference_embedding

    def similarity(self, image_path: Path) -> float | None:
        embedding = self._embedder.embed_largest_face(image_path)
        if embedding is None:
            return None
        return round(cosine_similarity(embedding, self._reference), 4)


class _DetectedFace(Protocol):
    @property
    def bbox(self) -> "npt.NDArray[np.float32]": ...

    @property
    def normed_embedding(self) -> "npt.NDArray[np.float32]": ...


class _FaceAnalyzer(Protocol):
    def prepare(self, ctx_id: int, det_size: tuple[int, int]) -> None: ...

    def get(self, img: "npt.NDArray[np.uint8]") -> list[_DetectedFace]: ...


class _FaceAnalyzerFactory(Protocol):
    def __call__(self, *, name: str, providers: list[str]) -> _FaceAnalyzer: ...


def _face_area(face: _DetectedFace) -> float:
    left, top, right, bottom = cast(list[float], face.bbox[:4].astype(float).tolist())
    return max(0.0, right - left) * max(0.0, bottom - top)


@final
class InsightFaceEmbedder:
    """ArcFace embeddings from InsightFace's buffalo_l pack (downloaded on first use)."""

    def __init__(self, model_pack: str = "buffalo_l") -> None:
        self._model_pack = model_pack
        self._analyzer: _FaceAnalyzer | None = None

    @property
    def name(self) -> str:
        return f"insightface/{self._model_pack}"

    def _load(self) -> _FaceAnalyzer:
        if self._analyzer is None:
            try:
                module = importlib.import_module("insightface.app")
                # insightface calls a scikit-image API that is deprecated but still works.
                warnings.filterwarnings(
                    "ignore", category=FutureWarning, module=r"insightface\."
                )
            except ImportError as error:
                raise RuntimeError(
                    "InsightFace is not installed. Run the toolkit through "
                    "scripts/likeness, which adds it."
                ) from error
            factory = cast(_FaceAnalyzerFactory, module.FaceAnalysis)
            analyzer = factory(
                name=self._model_pack, providers=["CPUExecutionProvider"]
            )
            analyzer.prepare(ctx_id=0, det_size=(640, 640))
            self._analyzer = analyzer
        return self._analyzer

    def embed_largest_face(self, image_path: Path) -> list[float] | None:
        # Imported here so the scoring rules above stay usable without Pillow installed.
        import numpy as np
        from PIL import Image, ImageOps

        with Image.open(image_path) as opened:
            upright = ImageOps.exif_transpose(opened).convert("RGB")
        border = round(max(upright.size) * DETECTION_PADDING)
        upright = ImageOps.expand(upright, border=border, fill=(0, 0, 0))
        # InsightFace expects OpenCV channel order (BGR).
        pixels = np.ascontiguousarray(np.asarray(upright, dtype=np.uint8)[:, :, ::-1])
        faces = self._load().get(pixels)
        if not faces:
            return None
        largest = max(faces, key=_face_area)
        return cast(list[float], largest.normed_embedding.astype(np.float64).tolist())
