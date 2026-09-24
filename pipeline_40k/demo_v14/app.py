"""Local Gradio demonstration for the frozen physics ensemble v14.

Run from ``pipeline_40k`` with:
    .venv\\Scripts\\python.exe demo_v14\\app.py --device cpu

Use ``--device cuda`` only when the GPU is not occupied by a training run.
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def build_demo(device: str):
    import gradio as gr
    from src.inference.v14 import V14PhysicsPredictor

    @lru_cache(maxsize=1)
    def predictor() -> V14PhysicsPredictor:
        return V14PhysicsPredictor(device=device)

    def detect(image):
        if image is None:
            raise gr.Error("Choose a PNG, JPEG, or WebP image first.")
        try:
            result = predictor().predict(image)
        except Exception as error:
            raise gr.Error(f"Inference could not complete: {error}") from error
        probability = result["fake_probability"]
        label = {
            "AI-generated": probability,
            "camera-origin": 1.0 - probability,
        }
        summary = (
            f"<h3>{result['decision'].capitalize()}</h3>"
            f"<p><b>AI-generated probability:</b> {probability:.1%}</p>"
            f"<p>The uncertainty band is {result['uncertainty_band'][0]:.1%}–"
            f"{result['uncertainty_band'][1]:.1%}. This research result is not proof of origin.</p>"
        )
        return label, result, summary

    with gr.Blocks(title="Physics Ensemble v14") as demo:
        gr.Markdown(
            "# Physics Ensemble v14\n"
            "Upload one image to inspect the frozen research detector’s calibrated score and the four "
            "physics-entity observability values. The detector is an academic artifact; it is not a forensic "
            "or universal authenticity decision tool."
        )
        with gr.Row():
            image = gr.Image(type="pil", label="Image", sources=["upload"])
            with gr.Column():
                result = gr.Label(label="Prediction", num_top_classes=2)
                summary = gr.HTML()
        run = gr.Button("Analyze image", variant="primary")
        diagnostics = gr.JSON(label="Physics diagnostics")
        run.click(detect, inputs=image, outputs=[result, diagnostics, summary])
        gr.Markdown(
            "**Interpretation:** the probability is calibrated on the project’s internal calibration split. "
            "It can vary across unseen generator families, image compression, and scene content."
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Request a temporary Gradio share URL.")
    args = parser.parse_args()
    if not (1 <= args.port <= 65535):
        raise ValueError("Port must be in 1..65535")
    build_demo(args.device).launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
