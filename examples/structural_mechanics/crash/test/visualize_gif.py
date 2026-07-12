# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to generate an animated GIF showing top and side views of the full car crash simulation."""

import os
import sys
from unittest.mock import MagicMock

# Mock NVIDIA warp and s3fs modules before importing anything from physicsnemo
sys.modules['warp'] = MagicMock()
sys.modules['s3fs'] = MagicMock()

# Ensure crash package modules and physicsnemo package are importable
CRASH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CRASH_DIR)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, CRASH_DIR)

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from datapipe import CrashPointCloudDataset
from vtkhdf_reader_stripped import Reader

def main():
    data_dir = "/Users/victorracu/MLProjects/simulations"
    
    print("Initializing Custom Reader and Dataset...")
    reader = Reader(front_x_threshold=3000.0)
    
    def test_reader(data_dir, num_samples, split=None, **kwargs):
        return reader(data_dir, num_samples, split=None, **kwargs)

    dataset = CrashPointCloudDataset(
        reader=test_reader,
        data_dir=data_dir,
        global_features_filepath="dummy",
        global_features=["velocity_kmh", "front_support_scale", "lower_rail_subframe_scale"],
        split="train",
        num_samples=1,
        num_steps=17,
        sample_type="all_time_steps"
    )
    
    print("Retrieving the first SimSample...")
    sample = dataset[0]
    
    # Reconstruct raw (denormalized) sequence
    pos_t0 = sample.node_features["coords"]  # [N, 3] (normalized)
    rollout_pos = sample.node_target         # [N, T-1, 3] (normalized)
    T = rollout_pos.shape[1] + 1
    
    pos_mean = torch.tensor(dataset.node_stats["pos_mean"])
    pos_std = torch.tensor(dataset.node_stats["pos_std"])
    
    full_seq_norm = torch.cat([pos_t0.unsqueeze(1), rollout_pos], dim=1)  # [N, T, 3]
    full_seq_raw = (full_seq_norm * pos_std.view(1, 1, 3)) + pos_mean.view(1, 1, 3)
    coords_t = full_seq_raw.permute(1, 0, 2).numpy()  # [T, N, 3]
    
    # Calculate deformation displacement over time (distance from t=0)
    displacements = np.linalg.norm(coords_t - coords_t[0:1], axis=-1)  # [T, N]
    max_disp = float(displacements.max()) or 1.0
    
    # Fixed viewport bounds for consistency
    x = coords_t[..., 0]
    y = coords_t[..., 1]
    z = coords_t[..., 2]
    xlim = (x.min(), x.max())
    ylim = (y.min(), y.max())
    zlim = (z.min(), z.max())
    
    print("Generating frame images...")
    frames = []
    
    for t in range(T):
        fig, (ax_top, ax_side) = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
        
        # 1. Top View (X-Y plane)
        sc_top = ax_top.scatter(
            coords_t[t, :, 0], 
            coords_t[t, :, 1], 
            c=displacements[t], 
            s=0.5, 
            cmap="jet", 
            vmin=0, 
            vmax=max_disp
        )
        ax_top.set_title(f"Top View (X-Y plane)", fontsize=12, fontweight="bold")
        ax_top.set_xlim(xlim)
        ax_top.set_ylim(ylim)
        ax_top.set_aspect("equal")
        ax_top.set_xlabel("X (mm)")
        ax_top.set_ylabel("Y (mm)")
        
        # 2. Side View (X-Z plane)
        sc_side = ax_side.scatter(
            coords_t[t, :, 0], 
            coords_t[t, :, 2], 
            c=displacements[t], 
            s=0.5, 
            cmap="jet", 
            vmin=0, 
            vmax=max_disp
        )
        ax_side.set_title(f"Side View (X-Z plane)", fontsize=12, fontweight="bold")
        ax_side.set_xlim(xlim)
        ax_side.set_ylim(zlim)
        ax_side.set_aspect("equal")
        ax_side.set_xlabel("X (mm)")
        ax_side.set_ylabel("Z (mm)")
        
        # Add single colorbar for both subplots
        cbar = fig.colorbar(sc_side, ax=[ax_top, ax_side], shrink=0.8, label="Deformation Displacement (mm)")
        
        fig.suptitle(
            f"Frontal Car Crash Simulation - Step {t}/{T - 1} (Velocity: {sample.global_features['velocity_kmh']} km/h)", 
            fontsize=14, 
            fontweight="bold"
        )
        
        # Convert Matplotlib figure to PIL Image
        fig.canvas.draw()
        rgba_buffer = fig.canvas.buffer_rgba()
        img = Image.fromarray(np.asarray(rgba_buffer))
        frames.append(img)
        
        plt.close(fig)
        print(f"Rendered frame {t + 1}/{T}")
        
    out_path = os.path.join(os.path.dirname(__file__), "frontal_crash_simulation.gif")
    print(f"Compiling GIF and saving to {out_path}...")
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=200,  # 200ms per frame
        loop=0
    )
    print("Successfully compiled and saved GIF!")
    
    # Also save copy to workspace root
    workspace_dir = os.path.abspath(os.path.join(CRASH_DIR, "..", "..", "..", ".."))
    dest_path = os.path.join(workspace_dir, "frontal_crash_simulation.gif")
    import shutil
    shutil.copy(out_path, dest_path)
    print(f"Saved GIF copy to workspace root: {dest_path}")

    # Also save copy to the main artifact directory if available
    artifact_dir = "/Users/victorracu/.gemini/antigravity-cli/brain/b874884e-14e1-4db5-815b-6f61327f8767"
    if os.path.isdir(artifact_dir):
        dest_path = os.path.join(artifact_dir, "frontal_crash_simulation.gif")
        shutil.copy(out_path, dest_path)
        print(f"Copied GIF to artifacts directory: {dest_path}")

if __name__ == "__main__":
    main()
