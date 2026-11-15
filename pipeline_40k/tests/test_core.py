import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from src.data.dataset import PhysicsFeatureDataset
from src.models.cross_gen_gated import FocalLoss, SupConLoss, TransformerPhysicsCrossGenHead
from src.models.improved import (
    FEATURE_SCHEMA,
    ImprovedCachedClassifier,
    feature_matrix,
    probability_logits,
)
from src.models.multi_entity import (
    MultiEntityCachedClassifier,
    MultiEntityConsistencyHead,
    fit_feature_standardizer,
    multi_entity_loss,
    standardize_features,
)
from src.extractors.surface_normals import SurfaceNormalsExtractor
from src.extractors.regional_physics import REGION_NAMES, RegionalPhysicsExtractor, quadrant_boxes
from src.extractors.corneal_optics import CornealOpticsExtractor
from src.extractors.perspective_vp import GEOMETRY_DIM, PerspectiveVPExtractor
from src.models.regional_multi_physics import (
    RegionalMultiPhysicsHead,
    fit_regional_standardizer,
    regional_multi_physics_loss,
    standardize_regional,
)
from src.utils.decision_policy import apply_decision_policy, fit_decision_policy
from src.utils.metrics import evaluate_predictions, evaluate_observability_subsets
from scripts.train_generator_invariant_v6 import _GradientReverse


class ModelTests(unittest.TestCase):
    def test_focal_loss_handles_singleton(self):
        value = FocalLoss()(torch.tensor([0.0]), torch.tensor([1.0]))
        self.assertEqual(value.ndim, 0)
        self.assertTrue(torch.isfinite(value))

    def test_supcon_singleton_has_differentiable_zero(self):
        projection = torch.randn(1, 8, requires_grad=True)
        value = SupConLoss()(projection, torch.tensor([0]))
        self.assertEqual(value.item(), 0.0)
        value.backward()
        self.assertIsNotNone(projection.grad)

    def test_transformer_validates_schema_and_confidence(self):
        model = TransformerPhysicsCrossGenHead().eval()
        with self.assertRaises(ValueError):
            model(torch.zeros(2, 13), torch.zeros(2, 4))
        with self.assertRaises(ValueError):
            model(torch.zeros(2, 14), torch.full((2, 4), 1.1))
        with torch.inference_mode():
            logits, projection = model(torch.zeros(1, 14), torch.zeros(1, 4))
        self.assertEqual(tuple(logits.shape), (1,))
        self.assertEqual(tuple(projection.shape), (1, 64))

    def test_dataset_rejects_invalid_cache_values(self):
        with self.assertRaises(ValueError):
            PhysicsFeatureDataset(torch.zeros(2, 14), torch.zeros(2, 4), torch.tensor([0.0, 2.0]))

    def test_surface_normals_never_use_random_head_implicitly(self):
        first = SurfaceNormalsExtractor()
        second = SurfaceNormalsExtractor()
        image = np.full((32, 32, 3), 128, dtype=np.uint8)
        first_features, first_confidence = first(image)
        second_features, second_confidence = second(image)
        torch.testing.assert_close(first_features, second_features)
        self.assertEqual(first_confidence, second_confidence)
        with self.assertRaises(ValueError):
            SurfaceNormalsExtractor(use_foundation_backbone=True)

    def test_official_dsine_checkpoint_is_finite_when_available(self):
        root = Path(__file__).resolve().parents[1]
        checkpoint = root / "models/normal_estimators/dsine/exp002_kappa/dsine.pt"
        if not checkpoint.exists():
            self.skipTest("official DSINE checkpoint is not installed")
        extractor = SurfaceNormalsExtractor(normal_backend="dsine", dsine_checkpoint=checkpoint)
        features, confidence = extractor(np.full((64, 64, 3), 128, dtype=np.uint8))
        self.assertEqual(tuple(features.shape), (3,))
        self.assertTrue(torch.isfinite(features).all())
        self.assertGreaterEqual(confidence, 0.05)
        self.assertLessEqual(confidence, 1.0)
        self.assertEqual(extractor.provenance()["backend"], "DSINE_v02_kappa")

    def test_regional_physics_uses_aligned_five_region_schema(self):
        image = np.zeros((64, 80, 3), dtype=np.uint8)
        image[:, :40, 0] = 210
        image[:32, :, 1] = 120
        extractor = RegionalPhysicsExtractor(SurfaceNormalsExtractor(normal_backend="sobel"))
        features, confidences = extractor.extract(image)
        self.assertEqual(tuple(features.shape), (len(REGION_NAMES), 14))
        self.assertEqual(tuple(confidences.shape), (len(REGION_NAMES), 4))
        self.assertTrue(torch.isfinite(features).all())
        self.assertTrue(torch.all((confidences >= 0) & (confidences <= 1)))
        self.assertEqual(quadrant_boxes(64, 80)[0], (0, 64, 0, 80))

    def test_general_specular_stats_preserve_normalized_float_scale(self):
        extractor = CornealOpticsExtractor(region_mode="general")
        crop_uint8 = np.arange(64, dtype=np.uint8).reshape(8, 8) * 4
        uint8_stats = extractor._extract_glint_stats(crop_uint8)
        float_stats = extractor._extract_glint_stats(crop_uint8.astype(np.float32) / 255.0)
        for left, right in zip(uint8_stats[:3], float_stats[:3]):
            torch.testing.assert_close(left, right)
        self.assertAlmostEqual(uint8_stats[3], float_stats[3], places=6)

    def test_projective_geometry_is_finite_and_confidence_gated(self):
        extractor = PerspectiveVPExtractor()
        blank, blank_confidence = extractor(np.zeros((96, 96, 3), dtype=np.uint8))
        self.assertEqual(tuple(blank.shape), (GEOMETRY_DIM,))
        self.assertEqual(blank_confidence, 0.02)
        scene = np.zeros((160, 200, 3), dtype=np.uint8)
        for offset in range(10, 190, 24):
            cv2.line(scene, (100, 8), (offset, 152), (255, 255, 255), 2)
        geometry, confidence = extractor(scene)
        self.assertTrue(torch.isfinite(geometry).all())
        self.assertGreaterEqual(confidence, 0.02)
        self.assertLessEqual(confidence, 1.0)

    def test_multi_entity_head_uses_all_four_confidence_aware_entities(self):
        model = MultiEntityConsistencyHead().eval()
        features = torch.randn(3, 14)
        confidences = torch.tensor([[1.0, 0.8, 0.4, 0.2]]).repeat(3, 1)
        output = model(features, confidences, return_details=True)
        self.assertEqual(tuple(output["logits"].shape), (3,))
        self.assertEqual(tuple(output["entity_logits"].shape), (3, 4))
        self.assertEqual(tuple(output["pair_logits"].shape), (3, 6))
        torch.testing.assert_close(output["pair_attention"].sum(1), torch.ones(3))
        loss = multi_entity_loss({key: value for key, value in output.items() if key != "pair_confidence"}, torch.tensor([0., 1., 0.]))
        self.assertTrue(torch.isfinite(loss["total"]))

    def test_multi_entity_standardizer_is_train_only_and_schema_checked(self):
        train = np.arange(42, dtype=np.float32).reshape(3, 14)
        standardizer = fit_feature_standardizer(train)
        standardized = standardize_features(torch.tensor(train), standardizer)
        self.assertAlmostEqual(float(standardized.mean()), 0.0, places=6)
        with self.assertRaises(ValueError):
            standardize_features(torch.zeros(2, 13), standardizer)

    def test_regional_model_uses_all_regions_and_entities(self):
        features = torch.randn(3, 5, 14)
        confidences = torch.rand(3, 5, 4)
        standardizer = fit_regional_standardizer(features.numpy())
        values = standardize_regional(features, standardizer)
        model = RegionalMultiPhysicsHead().eval()
        output = model(values, confidences, return_details=True)
        self.assertEqual(tuple(output["logits"].shape), (3,))
        self.assertEqual(tuple(output["entity_logits"].shape), (3, 4))
        self.assertEqual(tuple(output["region_logits"].shape), (3, 5))
        self.assertEqual(tuple(output["embedding"].shape), (3, 128))
        torch.testing.assert_close(output["region_attention"].sum(1), torch.ones(3))
        torch.testing.assert_close(output["pair_attention"].sum(2), torch.ones(3, 5))
        loss = regional_multi_physics_loss(output, torch.tensor([0., 1., 0.]))
        self.assertTrue(torch.isfinite(loss["total"]))

        reliability_model = RegionalMultiPhysicsHead(use_entity_attention=True).eval()
        reliability_output = reliability_model(values, confidences, return_details=True)
        self.assertEqual(tuple(reliability_output["entity_attention"].shape), (3, 5, 4))
        torch.testing.assert_close(reliability_output["entity_attention"].sum(2), torch.ones(3, 5))
        self.assertTrue(torch.isfinite(reliability_output["entity_attention"]).all())

        masked_confidences = confidences.clone()
        masked_confidences[:, :, 0] = 0.0
        changed_illumination = values.clone()
        changed_illumination[:, :, :5] += 1000.0
        torch.testing.assert_close(
            model(values, masked_confidences),
            model(changed_illumination, masked_confidences),
        )

    def test_regional_model_accepts_geometry_as_a_fifth_entity(self):
        entity_dims = (5, 4, 3, 2, GEOMETRY_DIM)
        features = torch.randn(2, 5, sum(entity_dims))
        confidences = torch.rand(2, 5, len(entity_dims))
        standardizer = fit_regional_standardizer(features.numpy(), entity_dims)
        values = standardize_regional(features, standardizer)
        model = RegionalMultiPhysicsHead(entity_dims=entity_dims).eval()
        output = model(values, confidences, return_details=True)
        self.assertEqual(tuple(output["entity_logits"].shape), (2, 5))
        self.assertEqual(tuple(output["pair_attention"].shape), (2, 5, 10))
        torch.testing.assert_close(output["entity_attention"].sum(2), torch.ones(2, 5))

    def test_generator_adversary_reverses_feature_gradient(self):
        values = torch.ones(4, requires_grad=True)
        _GradientReverse.apply(values, 0.25).sum().backward()
        torch.testing.assert_close(values.grad, torch.full((4,), -0.25))


class ImprovedFeatureTests(unittest.TestCase):
    def test_ignored_normal_and_legacy_mismatch_columns_cannot_change_input(self):
        features = np.zeros((2, 14), dtype=float)
        confidence = np.zeros((2, 4), dtype=float)
        base = feature_matrix(features, confidence)
        features[:, [8, 9, 10, 11]] = 1000
        confidence[:, [1, 2]] = 1
        np.testing.assert_array_equal(base, feature_matrix(features, confidence))
        self.assertEqual(base.shape[1], 12)
        self.assertEqual(FEATURE_SCHEMA["version"], "compact_reproducible_v1")

    def test_probability_logits_are_finite(self):
        result = probability_logits([0.0, 0.5, 1.0])
        self.assertTrue(np.isfinite(result).all())
        self.assertAlmostEqual(result[1], 0.0)


class EvaluationTests(unittest.TestCase):
    def test_policy_is_fit_without_test_inputs(self):
        logits = np.array([-3, -2, -1, 1, 2, 3], dtype=float)
        labels = np.array([0, 0, 0, 1, 1, 1])
        policy = fit_decision_policy(logits, labels, obs_threshold=0.5, abstention_margin=0.1)
        self.assertEqual(policy["selection_split"], "calibration")
        decisions = apply_decision_policy(logits, np.ones((6, 2)), policy)
        self.assertEqual(decisions["predictions"].tolist(), labels.tolist())
        self.assertEqual(decisions["selective_predictions"].tolist(), labels.tolist())

    def test_single_class_metrics_are_explicitly_undefined(self):
        result = evaluate_predictions(np.zeros(3), np.array([0.1, 0.2, 0.3]))
        self.assertIsNone(result["auc_roc"])
        self.assertIsNone(result["eer"])
        subset = evaluate_observability_subsets(
            np.array([0, 1]), np.array([0.1, 0.9]), np.zeros((2, 2)), obs_threshold=1.0
        )
        self.assertEqual(subset["observable"]["sample_count"], 0)
        self.assertIsNone(subset["observable"]["accuracy"])


class IntegrityArtifactTests(unittest.TestCase):
    def test_existing_integrity_split_is_disjoint(self):
        root = Path(__file__).resolve().parents[1]
        path = root / "models/improved_v1/integrity/split_indices.npz"
        manifest_path = root / "models/improved_v1/integrity/split_manifest.json"
        if not path.exists():
            self.skipTest("integrity audit has not been generated")
        with np.load(path) as splits:
            names = ("train_indices", "development_indices", "calibration_indices")
            sets = [set(splits[name].tolist()) for name in names]
            self.assertFalse(sets[0] & sets[1])
            self.assertFalse(sets[0] & sets[2])
            self.assertFalse(sets[1] & sets[2])
            self.assertEqual(len(splits["test_indices"]), 7980)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertFalse(manifest["test_labels_used_for_selection"])
        self.assertEqual(manifest["removed_test_overlap_rows"], 1)

    def test_promoted_artifact_reproduces_saved_probability(self):
        root = Path(__file__).resolve().parents[1]
        artifact = root / "models/improved_v2/model.joblib"
        predictions_path = root / "models/improved_v2/test_predictions.npz"
        if not artifact.exists():
            self.skipTest("promoted artifact has not been generated")
        with np.load(predictions_path) as saved:
            row = int(saved["row_indices"][0])
            expected = float(saved["improved_probability"][0])
        cache = torch.load(root / "data/cache_scaled/test_scaled_features.pt", map_location="cpu", weights_only=True)
        result = ImprovedCachedClassifier(artifact).predict_features(
            cache["features"][row], cache["confidences"][row]
        )
        self.assertAlmostEqual(float(result["probabilities"][0]), expected, places=12)

    def test_multi_entity_artifact_reproduces_saved_probability(self):
        root = Path(__file__).resolve().parents[1]
        artifact = root / "models/multi_entity_v3/model.pt"
        predictions_path = root / "models/multi_entity_v3/test_predictions.npz"
        if not artifact.exists():
            self.skipTest("multi-entity candidate has not been generated")
        with np.load(predictions_path) as saved:
            row = int(saved["row_indices"][0])
            expected = float(saved["candidate_probability"][0])
        cache = torch.load(root / "data/cache_scaled/test_scaled_features.pt", map_location="cpu", weights_only=True)
        result = MultiEntityCachedClassifier(artifact).predict_features(cache["features"][row], cache["confidences"][row])
        self.assertAlmostEqual(float(result["probabilities"][0]), expected, places=6)


if __name__ == "__main__":
    unittest.main()
