import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.inference.predict import DualStreamPredictor

p = DualStreamPredictor(checkpoint_path="models/universal_v3/best_model.pt")

imgs = [
    ("Asian Woman Outdoor (Real)", "data/test_asian_woman_outdoor.jpg"),
    ("Toddler Photo (Real)", r"C:/Users/laptop villa/.gemini/antigravity/brain/b7cb5a33-a10e-4028-a138-fa394175e4ff/.user_uploaded/media_1790960672770.jpg"),
    ("DSLR Blur (Real)", r"C:\Users\laptop villa\AppData\Local\Temp\gradio\e87b25f52193471154a16b20d382713c8b7e983c30a39fa25a88af2e3db9b751\DSC03398.JPG"),
    ("Smartphone Portrait (Real)", "data/test_smartphone_portrait_clean.png"),
    ("Catwoman (AI)", "data/catwoman_extracted.png"),
    ("Blue Bedroom PNG (AI)", "data/test_blue_bedroom.png"),
    ("Blue Bedroom JPG (AI)", r"C:\Users\laptop villa\AppData\Local\Temp\gradio\895daf81d3c3f075701e5a23ef1d4273ea6f3836016275e73f65ba6f3b05d2f8\r485s41njdrh1.jpg"),
]

for name, path in imgs:
    res = p.predict_image(path)
    print(f"=== {name} ===")
    print(f"Verdict: {res['verdict']} ({res['confidence']})")
    print(f"Final: {res['final_prob_fake']:.4f} | Sem: {res['semantic_prob_fake']:.4f} | Phys: {res['physics_prob_fake']:.4f} | Alpha: {res['trust_alpha']:.4f}")
    print(f"Finding: {res['forensic_finding']}")
    print(f"Sensor: {res['sensor_noise_details']}")
    print()
