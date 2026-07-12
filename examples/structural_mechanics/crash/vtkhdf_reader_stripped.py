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

"""Custom Geometry-Only and Spatial-Filtering VTKHDF Reader.

This reader loads unstructured mesh geometry from VTKHDF PartitionedDataSetCollection
format files, filters out the non-frontal nodes of the car based on X coordinate threshold,
and parses simulation run parameter configurations dynamically from local params.txt files.
"""

import os
import re
import numpy as np
import h5py

def load_params(txt_path):
    """Parse key-value options from params.txt."""
    params = {}
    if not os.path.isfile(txt_path):
        return params
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            parts = line.split("=", 1)
            key, val = parts[0].strip(), parts[1].strip()
            if val.lower() == "true":
                val = True
            elif val.lower() == "false":
                val = False
            else:
                try:
                    val = float(val)
                    if val.is_integer():
                        val = int(val)
                except ValueError:
                    pass
            params[key] = val
    return params

def find_vtkhdf_in_dir(directory):
    """Find any .vtkhdf file in a given directory."""
    if not os.path.isdir(directory):
        return None
    for name in ("field_trajectory.vtkhdf", "full_car_sim_sample.vtkhdf"):
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            return path
    for entry in sorted(os.listdir(directory)):
        if entry.lower().endswith(".vtkhdf"):
            return os.path.join(directory, entry)
    return None

def find_sim_dirs(data_dir):
    """Find directories containing simulation run data."""
    if not os.path.isdir(data_dir):
        return []
    
    # Check if data_dir itself contains a vtkhdf file
    if find_vtkhdf_in_dir(data_dir) is not None:
        return [data_dir]
        
    # Otherwise check subdirectories
    dirs = []
    for entry in sorted(os.listdir(data_dir)):
        sub = os.path.join(data_dir, entry)
        if os.path.isdir(sub):
            if find_vtkhdf_in_dir(sub) is not None:
                dirs.append(sub)
    return dirs

def _step_slices(part, dataset_len_key, offsets_key):
    """Yield step indexing slices."""
    counts = np.asarray(part[dataset_len_key])
    offsets = np.asarray(part[f"Steps/{offsets_key}"])
    for t in range(len(counts)):
        yield t, int(offsets[t]), int(counts[t])

def read_part_geometry_only(part_group):
    """Load points coordinates and connectivity cell indices for a single mesh part."""
    n_steps = len(part_group["NumberOfPoints"])
    n_pts = int(part_group["NumberOfPoints"][0])

    # Points coordinates over all steps
    points = np.asarray(part_group["Points"], dtype=np.float32)
    pos = np.empty((n_steps, n_pts, 3), dtype=np.float32)
    for t, start, count in _step_slices(part_group, "NumberOfPoints", "PointOffsets"):
        assert count == n_pts, "Non-constant point count across steps not supported."
        pos[t] = points[start : start + count]

    # Cell element connectivity
    n_cells = int(part_group["NumberOfCells"][0])
    offsets = np.asarray(part_group["Offsets"][: n_cells + 1])
    conn = np.asarray(part_group["Connectivity"][: int(part_group["NumberOfConnectivityIds"][0])])
    cells = [conn[offsets[i] : offsets[i + 1]].tolist() for i in range(n_cells)]

    return pos, cells

def get_geometry_part_names(root):
    """Return all valid geometry part names in natural order."""
    names = []
    for name, obj in root.items():
        if name in ("Assembly", "Steps") or not isinstance(obj, h5py.Group):
            continue
        if "Points" in obj and "Connectivity" in obj:
            names.append(name)
            
    def _natural_key(name):
        return [
            int(s) if s.isdigit() else s.lower()
            for s in re.findall(r"\d+|\D+", os.path.basename(str(name)))
        ]
    return sorted(names, key=_natural_key)

def load_geometry_file(path):
    """Load and merge all parts of the mesh geometry."""
    with h5py.File(path, "r") as f:
        root = f["VTKHDF"]
        part_names = get_geometry_part_names(root)
        if not part_names:
            raise ValueError(f"No geometry parts found in {path}")
            
        pos_list = []
        all_cells = []
        offset = 0
        n_steps_ref = None
        
        for name in part_names:
            pos, cells = read_part_geometry_only(root[name])
            if n_steps_ref is None:
                n_steps_ref = pos.shape[0]
            elif pos.shape[0] != n_steps_ref:
                raise ValueError(f"Part '{name}' has {pos.shape[0]} steps, expected {n_steps_ref}")
            
            pos_list.append(pos)
            for cell in cells:
                all_cells.append([idx + offset for idx in cell])
            offset += pos.shape[1]
            
    coords = np.concatenate(pos_list, axis=1)  # [T, N, 3]
    return coords, all_cells

def build_edges_from_cells(cells):
    """Build unique undirected edge indices from cells."""
    edges = set()
    for cell in cells:
        m = len(cell)
        for i in range(m):
            a, b = cell[i], cell[(i + 1) % m]
            if a != b:
                edges.add((a, b) if a < b else (b, a))
    return edges

def filter_frontal_mesh(coords, cells, front_x_threshold=3000.0):
    """Keep only nodes/edges located in the frontal part of the car."""
    # Retrieve coordinates at the initial time step (t=0)
    coords_t0 = coords[0]
    keep_mask = coords_t0[:, 0] >= front_x_threshold
    
    kept_indices = np.where(keep_mask)[0]
    if len(kept_indices) == 0:
        raise ValueError(f"No nodes found satisfying X >= {front_x_threshold}")
        
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(kept_indices)}
    
    # Filter points
    filtered_coords = coords[:, keep_mask, :]
    
    # Filter cell list to keep only elements whose vertices are kept
    filtered_cells = []
    for cell in cells:
        if all(node in old_to_new for node in cell):
            filtered_cells.append([old_to_new[node] for node in cell])
            
    # Rebuild edge indices
    edge_set = build_edges_from_cells(filtered_cells)
    
    if edge_set:
        edge_arr = np.array(sorted(edge_set), dtype=np.int64)
        src, dst = edge_arr[:, 0], edge_arr[:, 1]
    else:
        src = dst = np.zeros((0,), dtype=np.int64)
        
    return filtered_coords, src, dst


class Reader:
    """Modular, stripped geometry reader with spatial frontal filtering."""

    def __init__(self, front_x_threshold=3000.0, **kwargs):
        self.front_x_threshold = float(front_x_threshold)

    def __call__(
        self,
        data_dir,
        num_samples,
        split=None,
        global_features_filepath=None,
        logger=None,
        **kwargs,
    ):
        dirs = find_sim_dirs(data_dir)
        if not dirs:
            msg = f"No simulation directories with VTKHDF found in {data_dir}"
            if logger:
                logger.error(msg)
            raise FileNotFoundError(msg)
            
        # Optional split filtering based on params.txt
        if split:
            want_split = "val" if split in ("val", "validation") else split.strip().lower()
            filtered_dirs = []
            for sim_dir in dirs:
                params_path = os.path.join(sim_dir, "params.txt")
                if os.path.isfile(params_path):
                    params = load_params(params_path)
                    sim_split = str(params.get("split", "")).strip().lower()
                    mapped_sim_split = "val" if sim_split in ("val", "validation") else sim_split
                    if mapped_sim_split == want_split:
                        filtered_dirs.append(sim_dir)
            dirs = filtered_dirs

        srcs, dsts, point_data_all, global_features_all = [], [], [], []
        
        for sim_dir in dirs[:num_samples]:
            run_id = os.path.basename(sim_dir)
            if logger:
                logger.info(f"Processing {run_id} ...")
                
            vtkhdf_path = find_vtkhdf_in_dir(sim_dir)
            coords, cells = load_geometry_file(vtkhdf_path)
            
            # Filter geometry spatially to keep only the front
            filtered_coords, src, dst = filter_frontal_mesh(
                coords, cells, front_x_threshold=self.front_x_threshold
            )
            
            # Point data is empty (only mesh positions are modeled)
            record = {"coords": filtered_coords, "point_data": {}}
            
            # Load global feature options
            params_path = os.path.join(sim_dir, "params.txt")
            global_features = load_params(params_path)
            
            srcs.append(src)
            dsts.append(dst)
            point_data_all.append(record)
            global_features_all.append(global_features)
            
        return srcs, dsts, point_data_all, global_features_all
