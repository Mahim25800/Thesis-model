"""Universal AI Image Detector — Minimalist & Modern Forensic Suite.
Built with Dual-Stream Physics & DINOv2 Semantic Cross-Attention Network.
Integrates Opinionated Minimalist Bento Grid Architecture, Micro-Animations, and Explainable Forensic Telemetry.
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


# Custom Modern Styling & Animation CSS
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg-primary: #090D16;
    --bg-card: rgba(15, 23, 42, 0.75);
    --border-subtle: rgba(255, 255, 255, 0.08);
    --border-hover: rgba(99, 102, 241, 0.35);
    --accent-emerald: #10B981;
    --accent-rose: #F43F5E;
    --accent-indigo: #6366F1;
    --accent-sky: #38BDF8;
    --text-primary: #F8FAFC;
    --text-secondary: #94A3B8;
    --text-muted: #64748B;
}

body, .gradio-container {
    background-color: var(--bg-primary) !important;
    font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif !important;
    color: var(--text-primary) !important;
}

/* Subtle Ambient Glow in Background */
.gradio-container::before {
    content: '';
    position: fixed;
    top: -10%;
    left: 20%;
    width: 60vw;
    height: 40vh;
    background: radial-gradient(circle, rgba(99, 102, 241, 0.06) 0%, rgba(9, 13, 22, 0) 70%);
    pointer-events: none;
    z-index: 0;
}

/* Bento Box Container Cards */
.bento-card {
    background: var(--bg-card) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 14px !important;
    padding: 24px !important;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
    position: relative;
    overflow: hidden;
}

.bento-card:hover {
    border-color: var(--border-hover) !important;
    box-shadow: 0 8px 30px -4px rgba(0, 0, 0, 0.35), 0 0 15px -2px rgba(99, 102, 241, 0.12) !important;
    transform: translateY(-1.5px);
}

/* Shimmer Scanning Animation Effect */
@keyframes scanSweep {
    0% { transform: translateY(-100%); opacity: 0; }
    50% { opacity: 0.8; }
    100% { transform: translateY(300%); opacity: 0; }
}

.scanning-active::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 25%;
    background: linear-gradient(180deg, rgba(99, 102, 241, 0) 0%, rgba(56, 189, 248, 0.35) 50%, rgba(99, 102, 241, 0) 100%);
    animation: scanSweep 2.2s infinite ease-in-out;
    pointer-events: none;
}

/* Pulsing Status Dot */
@keyframes pulseDot {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.5); }
    70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}

.pulse-beacon {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: var(--accent-emerald);
    display: inline-block;
    animation: pulseDot 2s infinite cubic-bezier(0.4, 0, 0.6, 1);
    margin-right: 6px;
}

/* Monospace Pill Badges */
.mono-pill {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 9999px;
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.12);
    color: var(--text-secondary);
    display: inline-flex;
    align-items: center;
    letter-spacing: 0.04em;
}

/* Primary Analysis Button */
.cta-button {
    background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%) !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35) !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

.cta-button:hover {
    box-shadow: 0 6px 20px rgba(37, 99, 235, 0.5) !important;
    transform: translateY(-1px);
}

.cta-button:active {
    transform: scale(0.98);
}

/* Tabs and Output Controls */
.gr-tab-item {
    background: transparent !important;
    border: none !important;
}

.tabs button.selected {
    border-bottom: 2px solid var(--accent-sky) !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
}

/* Accordion Minimalist Styling */
.gr-accordion {
    border: 1px solid var(--border-subtle) !important;
    background: var(--bg-card) !important;
    border-radius: 12px !important;
}
"""

CUSTOM_HEAD = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<script src="https://cdnjs.cloudflare.com/ajax/libs/animejs/3.2.1/anime.min.js"></script>
"""


def create_demo(predictor: DualStreamPredictor):
    def analyze_image(img: Image.Image):
        if img is None:
            empty_html = """
            <div class="bento-card" style="text-align: center; padding: 48px 24px;">
                <div style="font-size: 2.2rem; margin-bottom: 12px; opacity: 0.5;">📁</div>
                <div style="font-size: 1.1rem; font-weight: 600; color: #94A3B8;">Awaiting Image Input</div>
                <p style="font-size: 0.85rem; color: #64748B; margin-top: 6px;">
                    Upload an authentic camera photograph or an AI-generated synthesis to inspect dual-stream physical invariants.
                </p>
            </div>
            """
            return empty_html, "", None, None, None

        result = predictor.predict_image(img, return_heatmaps=True)
        prob_fake = result["final_prob_fake"]
        prob_real = 1.0 - prob_fake

        is_fake = prob_fake >= 0.5
        percentage = (prob_fake if is_fake else prob_real) * 100.0

        sensor_details = result.get("sensor_noise_details", {})
        is_cam = sensor_details.get("camera_sensor_verified", False)
        is_bokeh = sensor_details.get("bokeh_blur_detected", False)
        finding = result.get("forensic_finding", "Multimodal Consensus")

        sem_fake_pct = result["semantic_prob_fake"] * 100.0
        phys_fake_pct = result["physics_prob_fake"] * 100.0
        alpha = result["trust_alpha"]
        temp = result.get("evidential_temperature", 1.0)
        quads = result.get("quadrant_inconsistencies", {})

        # Hardware Sensor Verification Badge
        hardware_pill = ""
        if is_cam and not is_fake:
            extra = " · Optical Bokeh Intact" if is_bokeh else " · High-Frequency PRNU Intact"
            hardware_pill = f"""
            <div style="margin-top: 14px; display: inline-flex; align-items: center; gap: 8px; padding: 6px 16px; background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.35); border-radius: 9999px; color: #34D399; font-size: 0.85rem; font-weight: 700;">
                <span style="font-size: 1rem;">📸</span> Hardware CMOS Sensor Verified{extra}
            </div>
            """
        elif is_fake:
            hardware_pill = f"""
            <div style="margin-top: 14px; display: inline-flex; align-items: center; gap: 8px; padding: 6px 16px; background: rgba(244, 63, 94, 0.12); border: 1px solid rgba(244, 63, 94, 0.35); border-radius: 9999px; color: #FB7185; font-size: 0.85rem; font-weight: 700;">
                <span style="font-size: 1rem;">⚠️</span> Non-Physical Generative Discontinuity Flagged
            </div>
            """

        # Color schemes depending on verdict
        theme_color = "#F43F5E" if is_fake else "#10B981"
        theme_bg = "rgba(244, 63, 94, 0.08)" if is_fake else "rgba(16, 185, 129, 0.08)"
        theme_border = "rgba(244, 63, 94, 0.35)" if is_fake else "rgba(16, 185, 129, 0.35)"
        verdict_title = "SYNTHETIC AI GENERATION" if is_fake else "AUTHENTIC CAMERA PHOTOGRAPH"
        verdict_sub = "Deep generative neural fingerprints or physical geometry anomalies identified." if is_fake else "3D surface curvature, optical shading, and sensor noise conform to physical camera capture."

        verdict_html = f"""
        <div class="bento-card" style="background: {theme_bg}; border: 1px solid {theme_border}; text-align: center; padding: 32px 24px;">
            <div class="mono-pill" style="margin-bottom: 12px; color: {theme_color}; border-color: {theme_border};">
                FORENSIC VERDICT // CONFIDENCE {percentage:.1f}%
            </div>
            
            <h2 style="font-size: 1.85rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.02em; margin: 0;">
                {verdict_title}
            </h2>
            
            <div style="font-size: 3.2rem; font-weight: 900; color: {theme_color}; line-height: 1.1; margin: 14px 0 6px 0; font-family: 'JetBrains Mono', monospace;">
                {percentage:.1f}%
            </div>
            
            <p style="font-size: 0.9rem; color: #94A3B8; max-width: 480px; margin: 0 auto; line-height: 1.5;">
                {verdict_sub}
            </p>

            {hardware_pill}

            <!-- Dual-Stream Probability Meter -->
            <div style="margin-top: 24px; padding-top: 20px; border-top: 1px solid rgba(255, 255, 255, 0.08);">
                <div style="display: flex; justify-content: space-between; font-size: 0.8rem; font-weight: 600; margin-bottom: 8px;">
                    <span style="color: #34D399;">AUTHENTIC CAMERA: {prob_real*100:.1f}%</span>
                    <span style="color: #FB7185;">AI SYNTHETIC: {prob_fake*100:.1f}%</span>
                </div>
                <div style="height: 10px; border-radius: 9999px; background: rgba(255, 255, 255, 0.1); overflow: hidden; display: flex;">
                    <div style="width: {prob_real*100:.1f}%; background: linear-gradient(90deg, #059669, #10B981); transition: width 0.8s ease;"></div>
                    <div style="width: {prob_fake*100:.1f}%; background: linear-gradient(90deg, #F43F5E, #E11D48); transition: width 0.8s ease;"></div>
                </div>
            </div>
        </div>
        """

        # 4-Card Multi-Stream Telemetry Bento
        def quad_val(k):
            return f"{quads.get(k, 0.0)*100:.1f}%" if quads else "N/A"

        telemetry_html = f"""
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; margin-top: 16px;">
            <!-- Card 1: DINOv2 Semantics -->
            <div class="bento-card" style="padding: 18px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span class="mono-pill">STREAM A // SEMANTIC</span>
                    <span style="font-size: 0.85rem; color: #A855F7; font-weight: 700;">ViT-B/14</span>
                </div>
                <div style="font-size: 1.6rem; font-weight: 800; color: #FFFFFF; font-family: 'JetBrains Mono', monospace;">
                    {sem_fake_pct:.1f}%
                </div>
                <div style="font-size: 0.78rem; color: #94A3B8; margin-top: 4px;">Predicted AI Probability</div>
                <div style="height: 5px; border-radius: 9999px; background: rgba(255,255,255,0.08); margin-top: 10px; overflow: hidden;">
                    <div style="width: {sem_fake_pct:.1f}%; background: #A855F7; height: 100%;"></div>
                </div>
            </div>

            <!-- Card 2: DSINE Physics -->
            <div class="bento-card" style="padding: 18px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span class="mono-pill">STREAM B // PHYSICAL</span>
                    <span style="font-size: 0.85rem; color: #38BDF8; font-weight: 700;">DSINE + SH</span>
                </div>
                <div style="font-size: 1.6rem; font-weight: 800; color: #FFFFFF; font-family: 'JetBrains Mono', monospace;">
                    {phys_fake_pct:.1f}%
                </div>
                <div style="font-size: 0.78rem; color: #94A3B8; margin-top: 4px;">Physics Inconsistency Score</div>
                <div style="height: 5px; border-radius: 9999px; background: rgba(255,255,255,0.08); margin-top: 10px; overflow: hidden;">
                    <div style="width: {phys_fake_pct:.1f}%; background: #38BDF8; height: 100%;"></div>
                </div>
            </div>

            <!-- Card 3: Dynamic Trust Gate -->
            <div class="bento-card" style="padding: 18px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span class="mono-pill">GATE // ALLOCATION</span>
                    <span style="font-size: 0.85rem; color: #F59E0B; font-weight: 700;">α(x) = {alpha:.2f}</span>
                </div>
                <div style="font-size: 1.15rem; font-weight: 700; color: #FFFFFF;">
                    {result['trust_interpretation'].split('(')[0]}
                </div>
                <div style="font-size: 0.78rem; color: #94A3B8; margin-top: 4px;">Dynamic Trust Parameter</div>
                <div style="height: 5px; border-radius: 9999px; background: rgba(255,255,255,0.08); margin-top: 10px; overflow: hidden;">
                    <div style="width: {alpha*100:.1f}%; background: #F59E0B; height: 100%;"></div>
                </div>
            </div>

            <!-- Card 4: Evidential Temperature -->
            <div class="bento-card" style="padding: 18px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span class="mono-pill">CALIBRATION // EVIDENTIAL</span>
                    <span style="font-size: 0.85rem; color: #10B981; font-weight: 700;">T ≥ 1.0</span>
                </div>
                <div style="font-size: 1.6rem; font-weight: 800; color: #FFFFFF; font-family: 'JetBrains Mono', monospace;">
                    {temp:.2f}
                </div>
                <div style="font-size: 0.78rem; color: #94A3B8; margin-top: 4px;">Epistemic Temperature T(x)</div>
                <div style="height: 5px; border-radius: 9999px; background: rgba(255,255,255,0.08); margin-top: 10px; overflow: hidden;">
                    <div style="width: {min(100.0, (temp/2.5)*100):.1f}%; background: #10B981; height: 100%;"></div>
                </div>
            </div>
        </div>

        <!-- Regional Quadrant Matrix (5-Region Decomposition) -->
        <div class="bento-card" style="margin-top: 14px; padding: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <span style="font-size: 0.95rem; font-weight: 700; color: #F8FAFC;">
                    🗺️ Multi-Instance Regional Anomaly Matrix (MIL Quadrant Decomposition)
                </span>
                <span class="mono-pill">5-REGION SPATIAL POOLING</span>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 10px; text-align: center;">
                    <div style="font-size: 0.75rem; color: #94A3B8; font-weight: 600;">QUADRANT 1 (TOP-LEFT)</div>
                    <div style="font-size: 1.2rem; font-weight: 800; color: #38BDF8; font-family: 'JetBrains Mono', monospace;">{quad_val('top_left')}</div>
                </div>
                <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 10px; text-align: center;">
                    <div style="font-size: 0.75rem; color: #94A3B8; font-weight: 600;">QUADRANT 2 (TOP-RIGHT)</div>
                    <div style="font-size: 1.2rem; font-weight: 800; color: #38BDF8; font-family: 'JetBrains Mono', monospace;">{quad_val('top_right')}</div>
                </div>
                <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 10px; text-align: center;">
                    <div style="font-size: 0.75rem; color: #94A3B8; font-weight: 600;">QUADRANT 3 (BOTTOM-LEFT)</div>
                    <div style="font-size: 1.2rem; font-weight: 800; color: #38BDF8; font-family: 'JetBrains Mono', monospace;">{quad_val('bottom_left')}</div>
                </div>
                <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 10px; text-align: center;">
                    <div style="font-size: 0.75rem; color: #94A3B8; font-weight: 600;">QUADRANT 4 (BOTTOM-RIGHT)</div>
                    <div style="font-size: 1.2rem; font-weight: 800; color: #38BDF8; font-family: 'JetBrains Mono', monospace;">{quad_val('bottom_right')}</div>
                </div>
            </div>
        </div>
        """

        heatmap_img = result.get("forensic_heatmap")
        normal_img = result.get("surface_normal_map")

        tech_details = {
            "Final Decision": "AI-Generated" if is_fake else "Real",
            "Confidence Percentage": f"{percentage:.2f}%",
            "Forensic Finding": finding,
            "Semantic Stream (DINOv2)": f"{sem_fake_pct:.2f}% fake",
            "Physical Stream (DSINE + SH)": f"{phys_fake_pct:.2f}% fake",
            "Dynamic Trust Weight alpha(x)": f"{alpha:.4f} ({result['trust_interpretation']})",
            "Evidential Temperature T(x)": f"{temp:.4f}",
            "Regional Quadrant Inconsistencies": quads,
            "Hardware Sensor Verified": "Yes" if is_cam else "No",
            "Optical Bokeh Blur Verified": "Yes" if is_bokeh else "No",
            "Directional Correlation (r)": f"{sensor_details.get('directional_correlation', 0.0):+.3f}",
        }

        return verdict_html, telemetry_html, heatmap_img, normal_img, tech_details

    ckpt_label = getattr(predictor, "checkpoint_label", "universal_v5_disagreement_gate/best_model.pt")
    gate_mode_label = getattr(predictor.model.fusion_head, "gate_mode", "v4_disagreement")

    # Sample images for instant testing
    sample_images = [
        "pipeline_dual_stream/data/Chameleon/Chameleon/test/0_real/002654cd-49e7-4229-b91a-82fa1aeea3d0.jpg",
        "pipeline_dual_stream/data/Chameleon/Chameleon/test/0_real/00089005-f5ae-44ab-bf8a-73dbd92ec2f8.jpg",
        "pipeline_dual_stream/data/Chameleon/Chameleon/test/1_fake/06b830f0-6026-4e48-9bd3-275938280df1.jpg",
        "pipeline_dual_stream/data/Chameleon/Chameleon/test/1_fake/06ce3cef-919e-4800-81c2-0dd843cccf55.jpg",
    ]
    valid_samples = [p for p in sample_images if Path(p).exists()]

    with gr.Blocks(title="AURA FORENSICS // Dual-Stream AI Detector") as demo:
        # ----------------- TOP NAVBAR / SYSTEM STATUS -----------------
        gr.HTML(
            f"""
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 18px 24px; border-bottom: 1px solid rgba(255, 255, 255, 0.08); margin-bottom: 24px;">
                <div style="display: flex; align-items: center; gap: 14px;">
                    <div style="width: 36px; height: 36px; border-radius: 10px; background: linear-gradient(135deg, #3B82F6, #6366F1); display: flex; align-items: center; justify-content: center; font-size: 1.2rem; font-weight: 800; color: white;">
                        ✦
                    </div>
                    <div>
                        <div style="font-size: 1.15rem; font-weight: 800; color: #F8FAFC; letter-spacing: -0.02em;">
                            AURA FORENSICS <span style="font-weight: 400; color: #64748B;">// DUAL-STREAM SUITE</span>
                        </div>
                        <div style="font-size: 0.78rem; color: #94A3B8;">
                            Dense Spatial Cross-Attention + Deterministic 3D Surface Invariants
                        </div>
                    </div>
                </div>

                <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
                    <div class="mono-pill">
                        <span class="pulse-beacon"></span> SYSTEM ACTIVE
                    </div>
                    <div class="mono-pill">
                        ⚡ {predictor.device.upper()} ACCELERATED
                    </div>
                    <div class="mono-pill" style="color: #38BDF8; border-color: rgba(56, 189, 248, 0.3);">
                        GATE: {gate_mode_label.upper()}
                    </div>
                </div>
            </div>
            """
        )

        # ----------------- MAIN WORKSPACE (BENTO GRID) -----------------
        with gr.Row():
            # LEFT: Input Dropzone & Controls
            with gr.Column(scale=5):
                with gr.Group():
                    input_image = gr.Image(
                        type="pil",
                        label="Drop or Upload Target Image",
                        sources=["upload", "clipboard"],
                        elem_classes=["bento-card"],
                    )
                    analyze_btn = gr.Button(
                        "⚡ Run Dual-Stream Forensic Analysis",
                        variant="primary",
                        size="lg",
                        elem_classes=["cta-button"],
                    )

                # Quick-load sample test buttons
                if valid_samples:
                    gr.Markdown("#### 🧪 Benchmark Verification Samples")
                    gr.Examples(
                        examples=valid_samples,
                        inputs=input_image,
                        label="Click any benchmark sample to inspect instantaneously",
                    )

            # RIGHT: Hero Verdict Card
            with gr.Column(scale=7):
                verdict_output = gr.HTML(
                    value="""
                    <div class="bento-card" style="text-align: center; padding: 50px 24px; min-height: 280px; display: flex; flex-direction: column; justify-content: center; align-items: center;">
                        <div style="width: 52px; height: 52px; border-radius: 50%; background: rgba(255, 255, 255, 0.04); border: 1px solid rgba(255, 255, 255, 0.08); display: flex; align-items: center; justify-content: center; font-size: 1.5rem; margin-bottom: 14px;">
                            🔍
                        </div>
                        <div style="font-size: 1.15rem; font-weight: 700; color: #F8FAFC;">Awaiting Forensic Inspection</div>
                        <p style="font-size: 0.85rem; color: #64748B; max-width: 420px; margin-top: 6px;">
                            Upload an image to compute 3D surface normals, spherical harmonic lighting residuals, and multi-instance patch anomalies.
                        </p>
                    </div>
                    """
                )

        # ----------------- STREAM BREAKDOWN & REGIONAL TELEMETRY -----------------
        telemetry_output = gr.HTML()

        # ----------------- EXPLAINABILITY VISUALIZERS (TABS) -----------------
        gr.Markdown("### 🔬 Spatial Explanations & Multimodal Geometry")
        with gr.Tabs(elem_classes=["bento-card"]):
            with gr.Tab("🗺️ Spatial Anomaly Heatmap"):
                heatmap_output = gr.Image(
                    type="pil",
                    label="Pixel-Level Forensic Anomaly Heatmap Overlay",
                    interactive=False,
                )
                gr.Markdown(
                    "*Warm red/yellow contours indicate localized generative synthesis flaws or broken physical lighting constraints.*"
                )

            with gr.Tab("🌐 3D Surface Normal Geometry (DSINE)"):
                normal_output = gr.Image(
                    type="pil",
                    label="Reconstructed 3D Normal Vectors",
                    interactive=False,
                )
                gr.Markdown(
                    "*DSINE surface normal vector map: Authentic camera photos conform to smooth continuous lens curvature; generative syntheses show severe planar breaks.*"
                )

            with gr.Tab("📊 Raw Forensic JSON Telemetry"):
                tech_output = gr.JSON(label="Complete Dual-Stream Invariant Telemetry")

        # Event Trigger
        analyze_btn.click(
            fn=analyze_image,
            inputs=[input_image],
            outputs=[verdict_output, telemetry_output, heatmap_output, normal_output, tech_output],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Launch Dual-Stream Modern Forensic UI")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="G:/Thesis/pipeline_dual_stream/models/universal_v5_disagreement_gate/best_model.pt",
    )
    parser.add_argument("--port", type=int, default=7865)
    parser.add_argument("--share", action="store_true", help="Create public Gradio share link")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Launching Upgraded Minimalist UI on port {args.port} using checkpoint {args.checkpoint}...")
    predictor = DualStreamPredictor(checkpoint_path=args.checkpoint, device=args.device)
    demo = create_demo(predictor)
    demo.launch(server_name="127.0.0.1", server_port=args.port, share=args.share, css=CUSTOM_CSS, head=CUSTOM_HEAD)


if __name__ == "__main__":
    main()
