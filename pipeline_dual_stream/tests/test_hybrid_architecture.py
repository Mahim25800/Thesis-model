"""Unit tests for the Dual-Stream Hybrid Architecture."""

import unittest
import torch

from src.models.dinov2_stream import DINOv2Stream
from src.models.physics_stream import RegionalPhysicsStream
from src.models.gated_fusion import GatedCrossAttentionFusion, dual_stream_hybrid_loss
from src.models.hybrid_detector import DualStreamHybridDetector


class TestDualStreamArchitecture(unittest.TestCase):
    """Test suite for dual-stream components and end-to-end integration."""

    def setUp(self):
        self.batch_size = 4
        self.img_size = 224
        self.phys_dim = 64
        self.proj_dim = 128

        # Dummy inputs
        self.images = torch.rand(self.batch_size, 3, self.img_size, self.img_size)
        self.physics_features = torch.randn(self.batch_size, 5, 14)
        self.physics_confidences = torch.rand(self.batch_size, 5, 4).clamp(0.01, 0.99)
        self.labels = torch.randint(0, 2, (self.batch_size,)).float()

    def test_dinov2_quadrant_pooling(self):
        """Verify quadrant pooling logic on synthetic patch tokens."""
        stream = DINOv2Stream(pretrained=False, img_size=self.img_size)
        # 16 x 16 patches
        patch_tokens = torch.randn(self.batch_size, 256, 768)
        quadrants = stream.pool_quadrants(patch_tokens)
        self.assertEqual(quadrants.shape, (self.batch_size, 4, 768))

    def test_physics_stream_shapes(self):
        """Verify regional physics stream encoding and shapes."""
        stream = RegionalPhysicsStream(d_model=self.phys_dim)
        out = stream(self.physics_features, self.physics_confidences)

        self.assertEqual(out["regional_tokens"].shape, (self.batch_size, 5, self.phys_dim))
        self.assertEqual(out["global_summary"].shape, (self.batch_size, 2 * self.phys_dim))
        self.assertEqual(out["physics_logits"].shape, (self.batch_size, 1))
        self.assertEqual(out["observability"].shape, (self.batch_size, 5))

    def test_gated_cross_attention_fusion(self):
        """Verify cross-attention, trust gating bounds, and discrepancy map."""
        fusion = GatedCrossAttentionFusion(
            sem_dim=768,
            phys_dim=self.phys_dim,
            proj_dim=self.proj_dim,
        )

        sem_tokens = torch.randn(self.batch_size, 5, 768)
        sem_global = torch.randn(self.batch_size, 768)
        phys_tokens = torch.randn(self.batch_size, 5, self.phys_dim)
        phys_global = torch.randn(self.batch_size, 2 * self.phys_dim)
        phys_conf = self.physics_confidences
        physics_logits = torch.randn(self.batch_size, 1)

        out = fusion(
            sem_tokens=sem_tokens,
            sem_global=sem_global,
            phys_tokens=phys_tokens,
            phys_global=phys_global,
            phys_conf=phys_conf,
            physics_logits=physics_logits,
        )

        # Check shapes
        self.assertEqual(out["logits"].shape, (self.batch_size,))
        self.assertEqual(out["sem_logits"].shape, (self.batch_size,))
        self.assertEqual(out["phys_logits"].shape, (self.batch_size,))
        self.assertEqual(out["joint_logits"].shape, (self.batch_size,))
        self.assertEqual(out["alpha"].shape, (self.batch_size,))
        self.assertEqual(out["quad_discrepancy"].shape, (self.batch_size, 4))

        # Check alpha is strictly bounded in [0, 1]
        self.assertTrue((out["alpha"] >= 0.0).all() and (out["alpha"] <= 1.0).all())

    def test_end_to_end_detector_and_loss(self):
        """Verify complete detector with loss and gradient backpropagation."""
        detector = DualStreamHybridDetector(
            load_pretrained_dinov2=False,  # faster test execution without weight download
            phys_dim=self.phys_dim,
            proj_dim=self.proj_dim,
        )

        # Forward pass with simulated precomputed tokens (fast path)
        dinov2_cls = torch.randn(self.batch_size, 768)
        dinov2_reg = torch.randn(self.batch_size, 5, 768)

        out = detector(
            physics_features=self.physics_features,
            physics_confidences=self.physics_confidences,
            dinov2_cls=dinov2_cls,
            dinov2_regional=dinov2_reg,
        )

        self.assertEqual(out["prob_final"].shape, (self.batch_size,))
        self.assertEqual(out["prob_semantic"].shape, (self.batch_size,))
        self.assertEqual(out["prob_physics"].shape, (self.batch_size,))

        # Multi-task loss calculation
        losses = dual_stream_hybrid_loss(out, self.labels)
        self.assertIn("total_loss", losses)
        self.assertGreater(losses["total_loss"].item(), 0.0)

        # Backpropagation
        losses["total_loss"].backward()

        # Check gradients exist in fusion head and physics stream
        for p in detector.fusion_head.parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad)
        for p in detector.physics_stream.parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad)


if __name__ == "__main__":
    unittest.main()
