# Local integration patch

This vendored DSINE checkout is pinned to `ef0c2afa32b4dd19cb8ca4567c652802cd92591c`.

`models/dsine/v02_kappa.py` has one local compatibility change: `pixel_coords` remains on CPU at construction and the pipeline adapter moves it to the selected device after checkpoint loading. Upstream's `.to(0)` required CUDA even for CPU inference. The DSINE architecture and checkpoint weights are unchanged.
