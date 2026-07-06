"""
Reader for the Dodge Neon full-body crash dataset (VTKHDF format).

Each simulation lives in a folder like:
    neon_full_frontal_ml_0002/
        field_trajectory.vtkhdf   (~1.3 GB)
        params.txt                (velocity, split, ...)
        qc_iso.gif / qc_side.gif  (pre-made quality-check animations)

The VTKHDF file contains ~1035 named parts (car components).  Each part is a
transient UnstructuredGrid with 17 timesteps.  This reader merges all
structural parts (those that carry Von Mises + Plastic Strain cell fields) into
a single point cloud and returns a dict that downstream code can use directly.

Quick usage
-----------
    from vtkhdf_reader import load_simulation, find_simulations

    sims = find_simulations("/mnt/1t/mit-project/Dataset/full_body")
    data = load_simulation(sims[0])

    data["positions"]          # [T, N, 3]  absolute node positions
    data["displacement"]       # [T, N, 3]  per-node displacement
    data["velocity"]           # [T, N, 3]  per-node velocity
    data["stress_vm"]          # [T, N]     von Mises stress (cell -> point averaged)
    data["plastic_strain"]     # [T, N]     effective plastic strain
    data["specific_energy"]    # [T, N]     specific internal energy
    data["part_id"]            # [N]        part ID per node (from t=0)
    data["node_id"]            # [N]        global node ID
    data["params"]             # dict       from params.txt (velocity_kmh, split, ...)
    data["sim_name"]           # str        e.g. "neon_full_frontal_ml_0002"
    data["n_parts_merged"]     # int        number of structural parts merged
"""

import os
import re

import h5py
import numpy as np


# CellData field name in file  ->  output key
_CELL_TARGET_MAP = {
    "Von Mises_2D":       "stress_vm",
    "Plastic Strain_2D":  "plastic_strain",
    "Specific Energy_2D": "specific_energy",
}

_POINT_TARGET_MAP = {
    "Displacement": "displacement",
    "Velocity":     "velocity",
}


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def find_simulations(data_dir: str, split: str = None) -> list:
    """Return sorted list of field_trajectory.vtkhdf paths under data_dir.

    Parameters
    ----------
    data_dir:
        Root directory containing one sub-folder per simulation.
    split:
        Optional filter: "train", "val", "test".  Reads params.txt to decide.
    """
    paths = []
    for entry in sorted(os.listdir(data_dir)):
        sub = os.path.join(data_dir, entry)
        if not os.path.isdir(sub):
            continue
        vtkhdf = os.path.join(sub, "field_trajectory.vtkhdf")
        if not os.path.isfile(vtkhdf):
            continue
        if split is not None:
            params = _read_params(sub)
            if params.get("split", "").lower() != split.lower():
                continue
        paths.append(vtkhdf)
    return paths


def _read_params(sim_dir: str) -> dict:
    """Parse params.txt -> dict of str->str (best-effort numeric conversion)."""
    p = os.path.join(sim_dir, "params.txt")
    out = {}
    if not os.path.isfile(p):
        return out
    with open(p) as f:
        for line in f:
            line = line.strip()
            if "=" not in line:
                continue
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip()
            # try numeric conversion
            for cast in (int, float):
                try:
                    v = cast(v)
                    break
                except ValueError:
                    pass
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Low-level part reader
# ---------------------------------------------------------------------------

def _is_structural(part_grp: h5py.Group) -> bool:
    """True if the part carries both Von Mises and Plastic Strain cell fields."""
    cd = part_grp.get("CellData")
    if cd is None:
        return False
    return "Von Mises_2D" in cd and "Plastic Strain_2D" in cd


def _step_slices(part_grp: h5py.Group, count_key: str, offset_key: str):
    """Yield (t, start, count) for a per-step concatenated dataset."""
    counts  = np.asarray(part_grp[count_key])
    offsets = np.asarray(part_grp[f"Steps/{offset_key}"])
    for t in range(len(counts)):
        yield t, int(offsets[t]), int(counts[t])


def _read_part(part_grp: h5py.Group):
    """Read one structural part.

    Returns
    -------
    pos     : [T, n, 3]  absolute node positions
    pd      : {key: [T, n]}  point-level fields (displacement, velocity)
    cd_pt   : {key: [T, n]}  cell fields averaged to points
    node_id : [n]  global node IDs (from t=0)
    part_id_arr : [n]  part ID broadcast to points (from t=0, first cell)
    """
    n_steps = len(part_grp["NumberOfPoints"])
    n_pts   = int(part_grp["NumberOfPoints"][0])
    n_cells = int(part_grp["NumberOfCells"][0])

    # --- positions (concatenated across steps) ---
    raw_pts = np.asarray(part_grp["Points"], dtype=np.float32)
    pos = np.empty((n_steps, n_pts, 3), dtype=np.float32)
    for t, start, count in _step_slices(part_grp, "NumberOfPoints", "PointOffsets"):
        pos[t] = raw_pts[start : start + count]

    # --- connectivity (constant, stored once) ---
    offsets_arr = np.asarray(part_grp["Offsets"][: n_cells + 1])
    conn        = np.asarray(part_grp["Connectivity"][
        : int(part_grp["NumberOfConnectivityIds"][0])
    ])
    cell_sizes  = np.diff(offsets_arr)

    # flat corner-to-node index for cell->point averaging
    corner_node = np.concatenate([
        conn[offsets_arr[i] : offsets_arr[i + 1]] for i in range(n_cells)
    ])
    cell_rep = np.repeat(np.arange(n_cells), cell_sizes)  # cell index per corner

    # --- point fields: Displacement, Velocity ---
    pd = {}
    for src_key, dst_key in _POINT_TARGET_MAP.items():
        raw = np.asarray(part_grp["PointData"][src_key], dtype=np.float32)
        arr = np.empty((n_steps, n_pts, raw.shape[-1]), dtype=np.float32)
        for t, start, count in _step_slices(part_grp, "NumberOfPoints",
                                             f"PointDataOffsets/{src_key}"):
            arr[t] = raw[start : start + count]
        pd[dst_key] = arr

    # --- cell fields -> averaged to points ---
    cd_pt = {}
    for src_key, dst_key in _CELL_TARGET_MAP.items():
        if src_key not in part_grp["CellData"]:
            continue
        raw = np.asarray(part_grp["CellData"][src_key], dtype=np.float32)
        arr = np.empty((n_steps, n_pts), dtype=np.float32)
        for t, start, count in _step_slices(part_grp, "NumberOfCells",
                                             f"CellDataOffsets/{src_key}"):
            cell_vals = raw[start : start + count]  # [n_cells]
            # scatter-add cell value to each corner node, then normalise
            node_sum   = np.zeros(n_pts, dtype=np.float64)
            node_count = np.zeros(n_pts, dtype=np.float64)
            np.add.at(node_sum,   corner_node, np.repeat(cell_vals, cell_sizes))
            np.add.at(node_count, corner_node, np.ones(len(corner_node), dtype=np.float64))
            node_count = np.maximum(node_count, 1)
            arr[t] = (node_sum / node_count).astype(np.float32)
        cd_pt[dst_key] = arr

    # --- node IDs and part IDs (from t=0) ---
    raw_nid = np.asarray(part_grp["PointData"]["NODE_ID"], dtype=np.int32)
    node_id = raw_nid[: n_pts]

    raw_pid = np.asarray(part_grp["CellData"]["Part_ID"], dtype=np.int32)
    part_id_val = int(raw_pid[0]) if len(raw_pid) > 0 else -1
    part_id_arr = np.full(n_pts, part_id_val, dtype=np.int32)

    return pos, pd, cd_pt, node_id, part_id_arr


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def load_simulation(
    vtkhdf_path: str,
    exclude_parts: tuple = (),
    verbose: bool = False,
) -> dict:
    """Load one full-body crash simulation into numpy arrays.

    Parameters
    ----------
    vtkhdf_path:
        Path to ``field_trajectory.vtkhdf``.
    exclude_parts:
        Part name substrings to skip (e.g. ``("ACCELEROMETER",)``).
    verbose:
        Print progress per part.

    Returns
    -------
    dict with keys:
        positions        [T, N, 3]   absolute node positions (mm)
        displacement     [T, N, 3]   nodal displacement (mm)
        velocity         [T, N, 3]   nodal velocity (mm/ms)
        stress_vm        [T, N]      von Mises stress (GPa)
        plastic_strain   [T, N]      effective plastic strain (-)
        specific_energy  [T, N]      specific internal energy
        part_id          [N]         part ID per node
        node_id          [N]         global node ID
        params           dict        from params.txt
        sim_name         str
        n_parts_merged   int
    """
    sim_dir  = os.path.dirname(vtkhdf_path)
    sim_name = os.path.basename(sim_dir)
    params   = _read_params(sim_dir)

    all_pos, all_disp, all_vel = [], [], []
    all_svm, all_eps, all_se   = [], [], []
    all_nid, all_pid           = [], []
    n_merged = 0

    with h5py.File(vtkhdf_path, "r") as f:
        root = f["VTKHDF"]
        part_names = sorted(root.keys())

        for pname in part_names:
            if pname in ("Assembly", "Steps"):
                continue
            if any(ex in pname for ex in exclude_parts):
                continue
            grp = root[pname]
            if not isinstance(grp, h5py.Group):
                continue
            if not _is_structural(grp):
                continue

            if verbose:
                print(f"  reading {pname} ...")

            pos, pd, cd_pt, nid, pid = _read_part(grp)
            all_pos.append(pos)
            all_disp.append(pd["displacement"])
            all_vel.append(pd["velocity"])
            all_svm.append(cd_pt.get("stress_vm"))
            all_eps.append(cd_pt.get("plastic_strain"))
            all_se.append(cd_pt.get("specific_energy"))
            all_nid.append(nid)
            all_pid.append(pid)
            n_merged += 1

    if n_merged == 0:
        raise ValueError(f"No structural parts found in {vtkhdf_path}")

    def _concat(lst):
        lst = [x for x in lst if x is not None]
        return np.concatenate(lst, axis=1) if lst else None

    return {
        "positions":       np.concatenate(all_pos,  axis=1),   # [T, N, 3]
        "displacement":    np.concatenate(all_disp, axis=1),   # [T, N, 3]
        "velocity":        np.concatenate(all_vel,  axis=1),   # [T, N, 3]
        "stress_vm":       _concat(all_svm),                   # [T, N]
        "plastic_strain":  _concat(all_eps),                   # [T, N]
        "specific_energy": _concat(all_se),                    # [T, N]
        "part_id":         np.concatenate(all_pid),            # [N]
        "node_id":         np.concatenate(all_nid),            # [N]
        "params":          params,
        "sim_name":        sim_name,
        "n_parts_merged":  n_merged,
    }


def list_parts(vtkhdf_path: str) -> list:
    """Return (part_name, n_points, n_cells, is_structural) for every part."""
    rows = []
    with h5py.File(vtkhdf_path, "r") as f:
        root = f["VTKHDF"]
        for pname in sorted(root.keys()):
            if pname in ("Assembly", "Steps"):
                continue
            grp = root[pname]
            if not isinstance(grp, h5py.Group):
                continue
            n_pts   = int(grp["NumberOfPoints"][0]) if "NumberOfPoints" in grp else 0
            n_cells = int(grp["NumberOfCells"][0])  if "NumberOfCells"  in grp else 0
            rows.append((pname, n_pts, n_cells, _is_structural(grp)))
    return rows


# ---------------------------------------------------------------------------
# CLI: quick summary of one simulation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Inspect a full-body crash simulation")
    parser.add_argument("vtkhdf", help="Path to field_trajectory.vtkhdf")
    parser.add_argument("--list-parts", action="store_true",
                        help="Print all parts instead of loading the full simulation")
    args = parser.parse_args()

    if args.list_parts:
        parts = list_parts(args.vtkhdf)
        structural = [(n, p, c) for n, p, c, s in parts if s]
        other      = [(n, p, c) for n, p, c, s in parts if not s]
        print(f"Structural parts ({len(structural)}):")
        for n, p, c in structural[:10]:
            print(f"  {n:50s}  pts={p}  cells={c}")
        if len(structural) > 10:
            print(f"  ... and {len(structural)-10} more")
        print(f"\nNon-structural parts ({len(other)}) — skipped by load_simulation()")
    else:
        print(f"Loading {args.vtkhdf} ...")
        d = load_simulation(args.vtkhdf, verbose=True)
        T, N = d["positions"].shape[:2]
        print(f"\n{'sim_name':<22}: {d['sim_name']}")
        print(f"{'parts merged':<22}: {d['n_parts_merged']}")
        print(f"{'timesteps (T)':<22}: {T}")
        print(f"{'nodes (N)':<22}: {N}")
        print(f"{'params':<22}: {d['params']}")
        print("\nArray shapes:")
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                print(f"  {k:<22}: {v.shape}  dtype={v.dtype}")
