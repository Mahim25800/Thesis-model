"""Clean & Streamlined Web Application for AI Image Detection.
Displays a clear Verdict (AI-Generated vs Real) and the exact Confidence Percentage.
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
            )

        result = predictor.predict_image(img)
        prob_fake = result["final_prob_fake"]
        prob_real = 1.0 - prob_fake

        is_fake = prob_fake >= 0.5
        percentage = (prob_fake if is_fake else prob_real) * 100.0

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
            </div>
            """

        confidence_chart = {
            "AI-Generated": prob_fake,
            "Real Camera": prob_real,
        }

        tech_details = {
            "Final Decision": "AI-Generated" if is_fake else "Real",
            "Confidence": f"{percentage:.2f}%",
            "DINOv2 Semantic Score": f"{result['semantic_prob_fake']*100:.1f}% fake",
            "Physics Consistency Score": f"{result['physics_prob_fake']*100:.1f}% fake",
            "Dynamic Trust Weight": result["trust_interpretation"],
        }

        return verdict_html, confidence_chart, tech_details

    with gr.Blocks(title="AI Image Detector") as demo:
        gr.Markdown(
            """
            # 🔍 AI Image Detector
            ### Upload an image to verify whether it is AI-generated or an authentic camera photo.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(type="pil", label="Upload Image", sources=["upload", "clipboard"])
                analyze_btn = gr.Button("Detect Image", variant="primary", size="lg")

            with gr.Column(scale=1):
                verdict_output = gr.HTML(
                    value="<div style='padding: 40px; text-align: center; color: #9ca3af; font-size: 1.2rem; border: 2px dashed #e5e7eb; border-radius: 16px;'>Upload an image and click <b>Detect Image</b> to see the verdict.</div>"
                )
                confidence_bar = gr.Label(label="Confidence Distribution")

                with gr.Accordion("Technical Diagnostics", open=False):
                    tech_output = gr.JSON(label="Stream Details")

        analyze_btn.click(
            fn=analyze_image,
            inputs=[input_image],
            outputs=[verdict_output, confidence_bar, tech_output],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Launch Dual-Stream Gradio UI")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="G:/Thesis/pipeline_dual_stream/models/dual_stream_v1/best_model.pt",
    )
    parser.add_argument("--port", type=int, default=7865)
    parser.add_argument("--share", action="store_true", help="Create public Gradio share link")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Launching Clean AI Detector UI on port {args.port}...")
    predictor = DualStreamPredictor(checkpoint_path=args.checkpoint, device=args.device)
    demo = create_demo(predictor)
    demo.launch(server_name="127.0.0.1", server_port=args.port, share=args.share)


if __name__ == "__main__":
    import torch
    main()
