"""
train_attention.py
===================
Runs the exact same training entrypoint as train.py, but monkeypatches
GeometricFeatureProcessor -> AttentionGeometricFeatureProcessor before the
model is built, so GeoTransolver's own ball-query local feature path uses
attention-weighted neighbor aggregation instead of concatenate+MLP.

Does not modify any physicsnemo core files or train.py itself.
"""

import sys
import os
import runpy

sys.path.insert(0, os.path.dirname(__file__))
# appended (not inserted at 0) so it never shadows this directory's own train.py
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "full_body_crash"))

import physicsnemo.experimental.models.geotransolver.context_projector as cp
from context_projector_attention import AttentionGeometricFeatureProcessor

cp.GeometricFeatureProcessor = AttentionGeometricFeatureProcessor
print(">>> Monkeypatched GeometricFeatureProcessor -> AttentionGeometricFeatureProcessor", flush=True)

if __name__ == "__main__":
    # run_path executes train.py with __name__ == "__main__" and __file__ set
    # to train.py's own path, so Hydra's config_path resolution behaves
    # identically to `python train.py ...` directly.
    train_py = os.path.join(os.path.dirname(__file__), "train.py")
    runpy.run_path(train_py, run_name="__main__")
