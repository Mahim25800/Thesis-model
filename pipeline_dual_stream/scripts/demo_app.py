"""Clean & Streamlined Web Application for AI Image Detection with Explainability Heatmaps.
Displays a clear Verdict (AI-Generated vs Real), Confidence Percentage,
and Spatial Multimodal Anomaly Heatmaps + 3D Surface Geometry.
Runs on http://127.0.0.1:7865
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import gradio as gr
import torch
from PIL import Image

from src.inference.predict import DualStreamPredictor


def create_demo(predictor: DualStreamPredictor):
    def analyze_image(img: Image.Image):
        if img is None:
            return (
                "<div style='padding: 20px; text-align: center; color: #6b7280; font-size: 1.2rem;'>Please upload an image to analyze.</div>",
                None,
                None,
                None,
                None,
            )

        result = predictor.predict_image(img, return_heatmaps=True)
        prob_fake = result["final_prob_fake"]
        prob_real = 1.0 - prob_fake

        is_fake = prob_fake >= 0.5
        percentage = (prob_fake if is_fake else prob_real) * 100.0

        sensor_details = result.get("sensor_noise_details", {})
        is_cam = sensor_details.get("camera_sensor_verified", False)
        is_bokeh = sensor_details.get("bokeh_blur_detected", False)
        finding = result.get("forensic_finding", "Multimodal Consensus")

        sensor_badge = ""
        if is_cam and not is_fake:
            extra = " (Optical Bokeh Detected)" if is_bokeh else ""
            sensor_badge = f"""
            <div style="margin-top: 12px; display: inline-block; padding: 6px 16px; background-color: #dcfce7; color: #15803d; border-radius: 9999px; font-size: 0.95rem; font-weight: 700; border: 1px solid #86efac;">
                📸 Hardware CMOS Sensor Verified{extra}
            </div>
            """

        if is_fake:
            verdict_html = f"""
            <div style="padding: 28px; border-radius: 16px; background-color: #fef2f2; border: 2px solid #ef4444; text-align: center; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);">
                <div style="font-size: 3rem; margin-bottom: 8px;">⚠️</div>
                <h1 style="color: #dc2626; margin: 0; font-size: 2.2rem; font-weight: 800; letter-spacing: -0.025em;">
                    AI-Generated Image
                </h1>
                <div style="margin-top: 14px; font-size: 2.5rem; font-weight: 900; color: #b91c1c;">
                    {percentage:.1f}% AI-Generated
                </div>
                <p style="color: #991b1b; margin-top: 8px; font-size: 1.1rem; font-weight: 500;">
                    (Probability Real: {prob_real*100:.1f}%)
                </p>
                <div style="margin-top: 10px; font-size: 0.95rem; color: #7f1d1d; font-weight: 600;">
                    Finding: {finding}
                </div>
            </div>
            """
        else:
            verdict_html = f"""
            <div style="padding: 28px; border-radius: 16px; background-color: #f0fdf4; border: 2px solid #22c55e; text-align: center; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);">
                <div style="font-size: 3rem; margin-bottom: 8px;">✅</div>
                <h1 style="color: #16a34a; margin: 0; font-size: 2.2rem; font-weight: 800; letter-spacing: -0.025em;">
                    Real Camera Photo
                </h1>
                <div style="margin-top: 14px; font-size: 2.5rem; font-weight: 900; color: #15803d;">
                    {percentage:.1f}% Real
                </div>
                <p style="color: #166534; margin-top: 8px; font-size: 1.1rem; font-weight: 500;">
                    (Probability AI-Generated: {prob_fake*100:.1f}%)
                </p>
                {sensor_badge}
            </div>
            """

        confidence_chart = {
            "AI-Generated": prob_fake,
            "Real Camera": prob_real,
        }

        heatmap_img = result.get("forensic_heatmap")
        normal_img = result.get("surface_normal_map")

        tech_details = {
            "Final Decision": "AI-Generated" if is_fake else "Real",
            "Confidence": f"{percentage:.2f}%",
            "Forensic Finding": finding,
            "DINOv2 Semantic Score": f"{result['semantic_prob_fake']*100:.1f}% fake",
            "Physics Consistency Score": f"{result['physics_prob_fake']*100:.1f}% fake",
            "Dynamic Trust Weight": result["trust_interpretation"],
            "Evidential Temperature T(x)": result.get("evidential_temperature", 1.0),
            "Quadrant Inconsistencies": result.get("quadrant_inconsistencies", {}),
            "Sensor Noise Residual": sensor_details.get("sensor_finding", "N/A"),
            "Hardware Sensor Verified": "Yes" if is_cam else "No",
            "Optical / Bokeh Blur": "Yes" if is_bokeh else "No",
            "Portrait Subject Detected": "Yes" if sensor_details.get("portrait_subject_detected", False) else "No",
            "Directional Lighting Verified": "Yes" if sensor_details.get("directional_lighting_verified", False) else "No",
            "Directional Correlation (r)": f"{sensor_details.get('directional_correlation', 0.0):+.2f}",
        }

        return verdict_html, confidence_chart, heatmap_img, normal_img, tech_details

    ckpt_label = getattr(predictor, "checkpoint_label", "universal_v5_disagreement_gate/best_model.pt")
    gate_mode_label = getattr(predictor.model.fusion_head, "gate_mode", "v4_disagreement")

    with gr.Blocks(title="AI Image Detector - Universal v5 (Disagreement-Calibrated)") as demo:
        gr.Markdown(
            f"""
            # 🔬 Universal AI Image Detector (v5 Disagreement-Calibrated)
            ### Dual-Stream Physics & Semantic Cross-Attention Network with Dynamic Evidential Gating

            <div style="background: #f0fdf4; border: 1px solid #86efac; border-radius: 8px; padding: 10px 16px; margin-bottom: 12px; font-size: 0.95rem; color: #166534;">
                🟢 <b>Active Checkpoint:</b> <code>models/{ckpt_label}</code> &nbsp;|&nbsp; 
                ⚙️ <b>Gate Mode:</b> <code>{gate_mode_label}</code> &nbsp;|&nbsp; 
                ⚡ <b>Compute Device:</b> <code>{predictor.device.upper()}</code>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(type="pil", label="Upload Image to Test", sources=["upload", "clipboard"])
                analyze_btn = gr.Button("🔍 Detect & Explain Image", variant="primary", size="lg")

                with gr.Accordion("ℹ️ How to Interpret the Visual Heatmaps", open=True):
                    gr.Markdown(
                        """
                        * 🟦 **Cool Blue / Cyan**: **Consistent Camera Physics** — Matches authentic hardware camera sensors, continuous 3D surface geometry, and realistic optical light fall-off.
                        * 🟥 **Warm Orange / Red**: **AI Anomaly Detected** — Localized neural generative artifacts, disrupted noise frequency spectra, or non-physical surface normal inversions.
                        """
                    )

            with gr.Column(scale=1):
                verdict_output = gr.HTML(
                    value="<div style='padding: 40px; text-align: center; color: #9ca3af; font-size: 1.2rem; border: 2px dashed #e5e7eb; border-radius: 16px;'>Upload an image and click <b>Detect & Explain Image</b> to view the verdict and explainability heatmap.</div>"
                )
                confidence_bar = gr.Label(label="Confidence Distribution")

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Tab("🔍 Forensic Anomaly Heatmap"):
                    heatmap_output = gr.Image(type="pil", label="Spatial Anomaly Heatmap Overlay", interactive=False)
                    gr.Markdown(
                        "*Overlay Map: Highlights spatial regions driving the model's verdict. Red/yellow regions indicate generative defects or non-physical lighting discrepancies.*"
                    )
                with gr.Tab("🌐 3D Surface Normal Geometry"):
                    normal_output = gr.Image(type="pil", label="Reconstructed 3D Surface Normals (DSINE)", interactive=False)
                    gr.Markdown(
                        "*DSINE Normal Vectors: Physical camera photos maintain smooth, continuous curvature conforming to optical lenses. Generative models often exhibit severe geometric planar warping.*"
                    )

            with gr.Column(scale=1):
                with gr.Accordion("📊 Technical Diagnostics & Stream Metrics", open=True):
                    tech_output = gr.JSON(label="Detailed Dual-Stream Forensic Metrics")

        analyze_btn.click(
            fn=analyze_image,
            inputs=[input_image],
            outputs=[verdict_output, confidence_bar, heatmap_output, normal_output, tech_output],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Launch Dual-Stream Gradio UI")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="G:/Thesis/pipeline_dual_stream/models/universal_v5_disagreement_gate/best_model.pt",
    )
    parser.add_argument("--port", type=int, default=7865)
    parser.add_argument("--share", action="store_true", help="Create public Gradio share link")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Launching Clean AI Detector UI on port {args.port} using checkpoint {args.checkpoint}...")
    predictor = DualStreamPredictor(checkpoint_path=args.checkpoint, device=args.device)
    demo = create_demo(predictor)
    demo.launch(server_name="127.0.0.1", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
