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

"""Quick visualization of what the custom stripped VTKHDF dataloader produces."""

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

from datapipe import CrashPointCloudDataset
from vtkhdf_reader_stripped import Reader

def main():
    # Path where the full car simulation is stored
    data_dir = "/Users/victorracu/MLProjects/simulations"
    
    print("Initializing Custom Reader and Dataset...")
    # Using the default threshold of 3000.0 to capture only the frontal part of the car
    reader = Reader(front_x_threshold=3000.0)
    
    # We wrap the reader to ignore the split parameter for this visualization script
    def test_reader(data_dir, num_samples, split=None, **kwargs):
        return reader(data_dir, num_samples, split=None, **kwargs)

    dataset = CrashPointCloudDataset(
        reader=test_reader,
        data_dir=data_dir,
        global_features_filepath="dummy",  # satisfy datapipe.py assertion
        global_features=["velocity_kmh", "front_support_scale", "lower_rail_subframe_scale"],
        split="train",  # computes stats on-the-fly
        num_samples=1,
        num_steps=17,
        sample_type="all_time_steps"
    )
    
    print("Retrieving the first SimSample...")
    sample = dataset[0]  # SimSample
    
    # 1. Print General Mesh Size and Feature Information
    pos_t0 = sample.node_features["coords"]  # [N, 3] (normalized)
    rollout_pos = sample.node_target         # [N, T-1, 3] (normalized)
    N = pos_t0.shape[0]
    T = rollout_pos.shape[1] + 1
    
    print("\n" + "="*50)
    print("               MESH AND SAMPLE STATS")
    print("="*50)
    print(f"Sample:           {sample}")
    print(f"Frontal Nodes:    {N:,}")
    print(f"Time steps:       {T}")
    print(f"Global features:  {sample.global_features}")
    
    # Kinematics stats
    pos_mean = torch.tensor(dataset.node_stats["pos_mean"])
    pos_std = torch.tensor(dataset.node_stats["pos_std"])
    print(f"Position Mean:    {pos_mean.tolist()}")
    print(f"Position Std:     {pos_std.tolist()}")
    
    # 2. Reconstruct raw (denormalized) sequence for plotting
    # Combine pos_t0 and rollout_pos into [N, T, 3]
    full_seq_norm = torch.cat([pos_t0.unsqueeze(1), rollout_pos], dim=1)  # [N, T, 3]
    # Denormalize to raw spatial coordinates (mm)
    full_seq_raw = (full_seq_norm * pos_std.view(1, 1, 3)) + pos_mean.view(1, 1, 3)
    coords_t = full_seq_raw.permute(1, 0, 2).numpy()  # [T, N, 3]
    
    # Calculate deformation displacement over time (distance from t=0)
    # displacement: [T, N]
    displacements = np.linalg.norm(coords_t - coords_t[0:1], axis=-1)
    
    # 3. Plotting the 2D deformation scatter plot over time (Top View: X-Y plane)
    print("\nGenerating 2D deformation visualization...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    
    # Select 6 frames evenly distributed over time
    frames = np.linspace(0, T - 1, 6).astype(int)
    max_disp = float(displacements.max()) or 1.0
    
    # Fixed viewport bounds for consistency
    x = coords_t[..., 0]
    y = coords_t[..., 1]
    xlim = (x.min(), x.max())
    ylim = (y.min(), y.max())
    
    sc = None
    for ax, t in zip(axes.ravel(), frames):
        sc = ax.scatter(
            coords_t[t, :, 0], 
            coords_t[t, :, 1], 
            c=displacements[t], 
            s=0.5, 
            cmap="jet", 
            vmin=0, 
            vmax=max_disp
        )
        ax.set_title(f"t = {t}/{T - 1}")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        
    fig.colorbar(sc, ax=axes, shrink=0.7, label="Deformation Displacement (mm)")
    title = f"Frontal Car Crash Simulation - Point Cloud Top View (Colored by Deformation Displacement)"
    fig.suptitle(title, fontsize=14, fontweight="bold")
    
    out_path = os.path.join(os.path.dirname(__file__), "frontal_point_cloud_visual.png")
    fig.savefig(out_path, dpi=110)
    plt.close()
    print(f"Successfully saved visualization plot to: {out_path}")
    
    # Also save to the main workspace directory for easy retrieval
    workspace_dir = os.path.abspath(os.path.join(CRASH_DIR, "..", "..", "..", ".."))
    dest_path = os.path.join(workspace_dir, "frontal_point_cloud_visual.png")
    import shutil
    shutil.copy(out_path, dest_path)
    print(f"Saved plot copy to workspace root: {dest_path}")

    # Also save to the main artifact directory if available
    artifact_dir = "/Users/victorracu/.gemini/antigravity-cli/brain/b874884e-14e1-4db5-815b-6f61327f8767"
    if os.path.isdir(artifact_dir):
        dest_path = os.path.join(artifact_dir, "frontal_point_cloud_visual.png")
        shutil.copy(out_path, dest_path)
        print(f"Copied plot to artifacts directory: {dest_path}")

if __name__ == "__main__":
    main()
