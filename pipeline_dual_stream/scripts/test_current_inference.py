import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.inference.predict import DualStreamPredictor

p = DualStreamPredictor(checkpoint_path='models/dual_stream_v2/best_model.pt')
imgs = [
    ('Smartphone Clean', 'data/test_smartphone_portrait_clean.png'),
    ('Smartphone Crop', 'data/test_smartphone_portrait.png'),
    ('Blue Bedroom Anime', 'data/test_blue_bedroom.png'),
    ('Catwoman AI', 'data/catwoman_extracted.png'),
    ('Catwoman Fence', 'data/test_catwoman_fence.png'),
]

for name, path in imgs:
    res = p.predict_image(path)
    print(f"=== {name} ===")
    print(f"Verdict: {res['verdict']} ({res['confidence']})")
    print(f"Final Prob Fake: {res['final_prob_fake']:.4f}")
    print(f"Semantic Prob Fake: {res['semantic_prob_fake']:.4f}")
    print(f"Physics Prob Fake: {res['physics_prob_fake']:.4f}")
    print(f"Alpha: {res['trust_alpha']:.4f}")
    print(f"Finding: {res['forensic_finding']}")
    print()
