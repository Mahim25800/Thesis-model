# Physics Ensemble v14 Demo

This local interface runs the frozen `physics_ensemble_v14` artifact on one
uploaded image. It computes the same five-region, four-entity physics feature
schema used in the thesis, validates the stored branch hashes, and applies the
frozen calibration policy.

## Run locally

Install the demo dependency once from the `pipeline_40k` directory:

```powershell
.venv\Scripts\python.exe -m pip install -r demo_v14\requirements.txt
```

While model training is using the NVIDIA GPU, run the interface on CPU:

```powershell
.venv\Scripts\python.exe demo_v14\app.py --device cpu
```

After the training run finishes, use CUDA for faster DSINE extraction:

```powershell
.venv\Scripts\python.exe demo_v14\app.py --device cuda
```

Open `http://127.0.0.1:7860` in a browser. The interface keeps uploads in
memory for inference; it does not save them as a dataset.

## Public deployment

The app needs the following release assets together: the v14 ensemble manifest,
its referenced branch models, the DSINE checkpoint, the vendored DSINE source
and runtime packages, plus `src`, `scripts`, and `demo_v14`. Do not publish raw
training, development, or external evaluation datasets.

For a temporary demonstration link, run with `--share`. For a durable public
demo, create a private Hugging Face Docker Space or another container host,
upload only the required release assets, and set the container command to:

```text
python demo_v14/app.py --host 0.0.0.0 --port 7860 --device cpu
```

The model card’s limitations still apply: this is a research interface, not a
forensic, legal, moderation, or autonomous authenticity system.
