"""Reproducible compact-feature classifier and its raw-image inference adapter.

The legacy cache combines native-resolution SH descriptors with 256px regional
highlight/color descriptors. Preserve that explicit recipe when reusing it.
Untrained normal features and their confidence are excluded entirely.
"""
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from ..data.preprocessor import validate_and_load_image
from ..extractors.illumination_sh import IlluminationSHExtractor
from ..extractors.corneal_optics import CornealOpticsExtractor
from ..extractors.chromatic_shadow import ChromaticShadowExtractor


FEATURE_COLUMNS = [0, 1, 2, 3, 4, 5, 6, 7, 12, 13]
CONFIDENCE_COLUMNS = [0, 3]
FEATURE_SCHEMA = {
    "version": "compact_reproducible_v1",
    "feature_columns": FEATURE_COLUMNS,
    "confidence_columns": CONFIDENCE_COLUMNS,
    "input_dimensions": [14, 4],
    "sh_resolution": "native",
    "sh_openmp_thread_limit": 6,
    "sh_blas_thread_limit": 12,
    "regional_resolution": [256, 256],
    "resize_interpolation": "INTER_AREA",
    "normal_features": "excluded: legacy prediction head was untrained",
    "highlight_profile_and_confidence": "excluded: historical cache differs from current implementation",
    "modalities": ["SH intensity-gradient proxy", "regional highlight descriptor", "bright-dark chromatic descriptor"],
    "physical_validity": "physics-inspired proxies; not verified optical invariants",
}


def as_numpy(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def feature_matrix(features, confidences):
    features, confidences = as_numpy(features), as_numpy(confidences)
    if features.ndim == 1:
        features = features[None, :]
    if confidences.ndim == 1:
        confidences = confidences[None, :]
    if features.ndim != 2 or confidences.ndim != 2 or features.shape[1] != 14 or confidences.shape != (len(features), 4):
        raise ValueError("Expected aligned [N,14] features and [N,4] confidences")
    # Only the selected columns define this model; ignored normal columns cannot affect predictions.
    selected = np.column_stack([features[:, FEATURE_COLUMNS], confidences[:, CONFIDENCE_COLUMNS]])
    if not np.isfinite(selected).all():
        raise ValueError("Selected features must be finite")
    conf = confidences[:, CONFIDENCE_COLUMNS]
    if ((conf < 0) | (conf > 1)).any():
        raise ValueError("Confidences must be in [0, 1]")
    return selected


def legacy_cache_matrix(features, confidences):
    """Return all historical cache columns for the accuracy-focused model.

    This schema cannot be reconstructed reliably from a new raw image because
    the historical normal head was not saved and two highlight columns changed.
    It is therefore intentionally available only for existing/versioned caches.
    """
    features, confidences = as_numpy(features), as_numpy(confidences)
    if features.ndim == 1:
        features = features[None, :]
    if confidences.ndim == 1:
        confidences = confidences[None, :]
    if features.shape != (len(features), 14) or confidences.shape != (len(features), 4):
        raise ValueError("Expected aligned [N,14] legacy features and [N,4] confidences")
    matrix = np.column_stack([features, confidences])
    if not np.isfinite(matrix).all() or ((confidences < 0) | (confidences > 1)).any():
        raise ValueError("Legacy cached inputs must be finite with confidences in [0,1]")
    return matrix


def probability_logits(probabilities):
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-7, 1 - 1e-7)
    return np.log(p) - np.log1p(-p)


class MeanProbabilityEnsemble:
    """Average already-fitted estimators without refitting on held-out data."""
    def __init__(self, estimators):
        if not estimators:
            raise ValueError("At least one estimator is required")
        self.estimators = list(estimators)

    def predict_proba(self, features):
        return np.mean([estimator.predict_proba(features) for estimator in self.estimators], axis=0)


class ReproducibleFeatureExtractor:
    """Reconstruct only the deterministic columns used by the improved model.

    Fixed highlight regions are explicitly regional descriptors, not detected eyes.
    No network downloads, neural normal head, or random initialization is used.
    """
    def __init__(self):
        self.sh = IlluminationSHExtractor()
        self.highlight = CornealOpticsExtractor()
        self.chromatic = ChromaticShadowExtractor()

    def extract(self, image):
        original = validate_and_load_image(image)
        resized = cv2.resize(original, (256, 256), interpolation=cv2.INTER_AREA)
        # The legacy cache was produced with six numerical threads. Its small
        # spherical k-means can change assignments at exact ties under a
        # different reduction order, so pin the recipe until caches are rebuilt.
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(FEATURE_SCHEMA["sh_openmp_thread_limit"])
        try:
            with threadpool_limits(limits={
                "openmp": FEATURE_SCHEMA["sh_openmp_thread_limit"],
                "blas": FEATURE_SCHEMA["sh_blas_thread_limit"],
            }):
                light, c_light = self.sh(original)
        finally:
            torch.set_num_threads(previous_threads)
        highlight, c_highlight = self.highlight(resized)
        chromatic, c_chromatic = self.chromatic(resized)
        features = torch.cat([light, highlight, torch.zeros(3), chromatic])
        confidences = torch.tensor([c_light, c_highlight, 0., c_chromatic], dtype=torch.float32)
        return features, confidences


class ImprovedPhysicsClassifier:
    def __init__(self, artifact_path):
        # Model artifacts are executable Python serialization: load only trusted local runs.
        self.artifact = joblib.load(Path(artifact_path))
        if self.artifact.get("feature_schema") != FEATURE_SCHEMA:
            raise ValueError("Model feature schema differs from the installed extractor")
        self.estimator = self.artifact["estimator"]
        self.policy = self.artifact["policy"]
        self.extractor = ReproducibleFeatureExtractor()

    def predict_features(self, features, confidences):
        from ..utils.decision_policy import apply_policy_calibration
        matrix = feature_matrix(features, confidences)
        raw = self.estimator.predict_proba(matrix)[:, 1]
        probabilities = apply_policy_calibration(probability_logits(raw), self.policy)
        conf = as_numpy(confidences)
        if conf.ndim == 1:
            conf = conf[None, :]
        observability = conf[:, CONFIDENCE_COLUMNS].sum(1)
        threshold = self.policy["decision_threshold"]
        results = []
        for p, obs in zip(probabilities, observability):
            binary = int(p >= threshold)
            decided = obs >= self.policy["obs_threshold"] and (p < self.policy["tau_low"] or p >= self.policy["tau_high"])
            results.append({
                "probability_fake": float(p),
                "binary_prediction": binary,
                "decision_threshold": threshold,
                "verdict": ("SYNTHETIC" if binary else "AUTHENTIC") if decided else "INDETERMINATE",
                "observability_score": float(obs),
                "is_decided": bool(decided),
            })
        return results

    def predict_image(self, image):
        features, confidences = self.extractor.extract(image)
        result = self.predict_features(features, confidences)[0]
        result["features"] = features.tolist()
        result["confidences"] = confidences.tolist()
        result["feature_schema"] = FEATURE_SCHEMA["version"]
        return result


class ImprovedCachedClassifier:
    """Accuracy-focused ensemble for the audited historical cache schema."""
    def __init__(self, artifact_path, legacy_checkpoint=None):
        from .cross_gen_gated import TransformerPhysicsCrossGenHead
        self.artifact_path = Path(artifact_path)
        self.artifact = joblib.load(self.artifact_path)
        if self.artifact.get("feature_schema") != "legacy_cache_14_plus_4_v1":
            raise ValueError("Artifact is not a compatible legacy-cache model")
        self.estimator = self.artifact["estimator"]
        self.policy = self.artifact["policy"]
        self.blend_weight = float(self.artifact["blend_weight"])
        checkpoint = Path(legacy_checkpoint) if legacy_checkpoint else self.artifact_path.parent.parent / "gated_cross_gen_40k_best.pt"
        expected = self.artifact["metadata"]["legacy_checkpoint_sha256"]
        import hashlib
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != expected:
            raise ValueError("Legacy checkpoint does not match the frozen ensemble")
        self.legacy = TransformerPhysicsCrossGenHead().eval()
        self.legacy.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))

    def predict_features(self, features, confidences):
        from ..utils.decision_policy import apply_decision_policy
        features_t = torch.as_tensor(features, dtype=torch.float32)
        confidences_t = torch.as_tensor(confidences, dtype=torch.float32)
        if features_t.ndim == 1:
            features_t = features_t.unsqueeze(0)
        if confidences_t.ndim == 1:
            confidences_t = confidences_t.unsqueeze(0)
        cached_probability = self.estimator.predict_proba(legacy_cache_matrix(features_t, confidences_t))[:, 1]
        with torch.inference_mode():
            legacy_probability = torch.sigmoid(self.legacy(features_t, confidences_t)[0]).numpy()
        blended = self.blend_weight * cached_probability + (1.0 - self.blend_weight) * legacy_probability
        return apply_decision_policy(probability_logits(blended), confidences_t.numpy(), self.policy)
