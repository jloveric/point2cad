import argparse
import multiprocessing
import numpy as np
import os
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

from point2cad.fitting_one_surface import process_one_surface
from point2cad.io_utils import save_unclipped_meshes, save_clipped_meshes, save_topology
from point2cad.pointcloud_io import load_points_and_labels, resolve_input_paths
from point2cad.utils import seed_everything, normalize_points, make_colormap_optimal


def process_multiprocessing(cfg, uniq_labels, points, labels, device):
    out_meshes = {}
    with ProcessPoolExecutor(max_workers=cfg.max_parallel_surfaces) as executor:
        futures = {
            executor.submit(process_one_surface, idx, points, labels, cfg, device): idx
            for idx in uniq_labels
        }

        for future in tqdm(
            as_completed(futures), total=len(uniq_labels), desc="Fitting surfaces"
        ):
            idx = futures[future]
            out_meshes[idx] = future.result()
    out_meshes = [out_meshes[idx] for idx in uniq_labels if out_meshes[idx] is not None]
    return out_meshes


def process_singleprocessing(cfg, uniq_labels, points, labels, device):
    out_meshes = []
    for idx in tqdm(uniq_labels, total=len(uniq_labels), desc="Fitting surfaces"):
        surface = process_one_surface(idx, points, labels, cfg, device)
        if surface is not None:
            out_meshes.append(surface)
    return out_meshes


def run_pipeline(cfg, path_in, path_out, device, color_list, fn_process):
    os.makedirs(path_out, exist_ok=True)
    os.makedirs(f"{path_out}/unclipped", exist_ok=True)
    os.makedirs(f"{path_out}/clipped", exist_ok=True)
    os.makedirs(f"{path_out}/topo", exist_ok=True)

    points, labels = load_points_and_labels(path_in, max_points=cfg.max_points)
    points = normalize_points(points)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    uniq_labels = np.unique(labels)
    print(f"Processing {path_in}: {len(points)} points across {len(uniq_labels)} surfaces")

    out_meshes = fn_process(cfg, uniq_labels, points, labels, device)

    print("Saving unclipped meshes...")
    pm_meshes = save_unclipped_meshes(
        out_meshes, color_list, f"{path_out}/unclipped/mesh.ply"
    )

    print("Saving clipped meshes...")
    clipped_meshes = save_clipped_meshes(
        pm_meshes, out_meshes, color_list, f"{path_out}/clipped/mesh.ply"
    )

    print("Saving topology (edges and corners)...")
    save_topology(clipped_meshes, f"{path_out}/topo/topo.json")


def output_path_for_input(path_in, path_out, has_multiple_inputs):
    stem = Path(path_in).stem
    if path_out is None:
        return os.path.join("outputs", stem)
    if has_multiple_inputs:
        return os.path.join(path_out, stem)
    return path_out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    color_list = make_colormap_optimal()

    parser = argparse.ArgumentParser(description="Point2CAD pipeline")
    parser.add_argument("--path_in", type=str, default="./assets/abc_00949.xyzc")
    parser.add_argument(
        "--path_out",
        type=str,
        default=None,
        help="Output directory. Defaults to outputs/<input filename without extension>.",
    )
    parser.add_argument("--validate_checkpoint_path", type=str, default=None)
    parser.add_argument("--silent", default=True)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--max_parallel_surfaces", type=int, default=4)
    parser.add_argument(
        "--max_points",
        type=int,
        default=10000,
        help="Maximum input points after labeling; uses Open3D farthest point sampling. Set <=0 to disable.",
    )
    parser.add_argument("--num_inr_fit_attempts", type=int, default=1)
    parser.add_argument("--surfaces_multiprocessing", type=int, default=1)
    parser.add_argument(
        "--primitives_only",
        action="store_true",
        help="Skip neural INR fitting; use analytic primitives only (plane, sphere, cylinder, cone)",
    )
    parser.add_argument(
        "--planes_only",
        action="store_true",
        help="Fit only planes; skips INR and all non-planar primitives",
    )
    cfg = parser.parse_args()

    seed_everything(cfg.seed)

    fn_process = process_singleprocessing
    if cfg.surfaces_multiprocessing:
        multiprocessing.set_start_method("spawn", force=True)
        fn_process = process_multiprocessing

    assert os.path.exists(cfg.path_in), "Input points could not be accessed"

    input_paths = resolve_input_paths(cfg.path_in)
    for path_in in input_paths:
        path_out = output_path_for_input(path_in, cfg.path_out, len(input_paths) > 1)
        run_pipeline(cfg, path_in, path_out, device, color_list, fn_process)

    print("Done")
