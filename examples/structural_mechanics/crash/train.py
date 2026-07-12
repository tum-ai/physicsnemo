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

import os
import sys
import time
import logging

sys.path.insert(0, os.path.dirname(__file__))

import hydra
import omegaconf
from hydra.utils import instantiate
from omegaconf import DictConfig

import tempfile

import numpy as np
import torch
import wandb
from PIL import Image, ImageDraw
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from physicsnemo.core.version_check import OptionalImport
from physicsnemo.distributed.manager import DistributedManager
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.utils import load_checkpoint, save_checkpoint

# Optional: tabulate for metrics tables, torchinfo for model summary
_tabulate = OptionalImport("tabulate")
_torchinfo = OptionalImport("torchinfo")

# Import unified datapipe and utils
from datapipe import SimSample, simsample_collate
from omegaconf import open_dict
from utils import build_muon_optimizer


# ── GIF rendering (PIL only, matches inference/visualize_crash.py style) ──────
_GIF_BG      = (16, 16, 20)
_GIF_DATA_H  = 280
_GIF_TITLE_H = 20
_GIF_BAR_H   = 22
_GIF_GAP     = 8
_GIF_PAD     = 0.04
_GIF_MARK_R  = 1

_INFERNO = np.array([[0,0,4],[40,11,84],[101,21,110],[159,42,99],
                     [212,72,66],[245,125,21],[250,193,39],[252,255,164]], dtype=np.float64)
_VIRIDIS = np.array([[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]], dtype=np.float64)


def _build_lut(anchors, n=256):
    xp = np.linspace(0.0, 1.0, len(anchors))
    xs = np.linspace(0.0, 1.0, n)
    return np.stack([np.interp(xs, xp, anchors[:, c]) for c in range(3)], axis=1)


_GIF_LUTS = {"inferno": _build_lut(_INFERNO), "viridis": _build_lut(_VIRIDIS)}


def _gif_colorize(vals, vmin, vmax, lut):
    lut_arr = _GIF_LUTS[lut]
    t = np.clip((vals - vmin) / max(vmax - vmin, 1e-9), 0.0, 1.0)
    return lut_arr[(t * (len(lut_arr) - 1)).astype(np.int64)].astype(np.uint8)


def _gif_extents(coords_TN3, ax_h, ax_v):
    h, v = coords_TN3[..., ax_h], coords_TN3[..., ax_v]
    hmin, hmax = float(h.min()), float(h.max())
    vmin, vmax = float(v.min()), float(v.max())
    hp = (hmax - hmin) * _GIF_PAD or 1.0
    vp = (vmax - vmin) * _GIF_PAD or 1.0
    return (hmin - hp, hmax + hp), (vmin - vp, vmax + vp)


def _gif_panel(coords_N3, vals_N, ax_h, ax_v, hext, vext, lut, vmin, vmax, title):
    hr, vr = hext[1] - hext[0], vext[1] - vext[0]
    W = max(120, min(int(round(_GIF_DATA_H * hr / vr)) if vr > 0 else _GIF_DATA_H, 900))
    H = _GIF_DATA_H
    plot = np.full((H, W, 3), _GIF_BG, dtype=np.uint8)

    order = np.argsort(vals_N)
    h = coords_N3[order, ax_h]
    v = coords_N3[order, ax_v]
    colors = _gif_colorize(vals_N[order], vmin, vmax, lut)
    px = ((h - hext[0]) / hr * (W - 1)).astype(np.int64)
    py = (H - 1) - ((v - vext[0]) / vr * (H - 1)).astype(np.int64)
    r = _GIF_MARK_R
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            xx, yy = px + dx, py + dy
            m = (xx >= 0) & (xx < W) & (yy >= 0) & (yy < H)
            plot[yy[m], xx[m]] = colors[m]

    panel = Image.new("RGB", (W, _GIF_TITLE_H + H + _GIF_BAR_H), _GIF_BG)
    panel.paste(Image.fromarray(plot), (0, _GIF_TITLE_H))
    draw = ImageDraw.Draw(panel)
    draw.text((4, 4), title, fill=(230, 230, 230))
    bar = np.zeros((_GIF_BAR_H, W, 3), dtype=np.uint8)
    bar[:_GIF_BAR_H - 10] = _gif_colorize(np.linspace(vmin, vmax, W), vmin, vmax, lut)[None]
    panel.paste(Image.fromarray(bar), (0, _GIF_TITLE_H + H))
    draw.text((2, _GIF_TITLE_H + H + _GIF_BAR_H - 10), f"{vmin:.3g}", fill=(200, 200, 200))
    draw.text((W - 42, _GIF_TITLE_H + H + _GIF_BAR_H - 10), f"{vmax:.3g}", fill=(200, 200, 200))
    return panel


def _gif_compose(panels, header):
    total_w = sum(p.width for p in panels) + _GIF_GAP * (len(panels) - 1)
    head_h = 18
    frame = Image.new("RGB", (total_w, head_h + panels[0].height), _GIF_BG)
    x = 0
    for p in panels:
        frame.paste(p, (x, head_h))
        x += p.width + _GIF_GAP
    ImageDraw.Draw(frame).text((6, 4), header, fill=(255, 255, 255))
    return frame
# ─────────────────────────────────────────────────────────────────────────────


class Trainer:
    """Trainer for crash simulation models with unified SimSample input."""

    def __init__(self, cfg: DictConfig, logger0: RankZeroLoggingWrapper):
        assert DistributedManager.is_initialized()
        self.dist = DistributedManager()
        self.cfg = cfg
        self.rollout_steps = cfg.training.num_time_steps - 1
        self.amp = cfg.training.amp

        # --- Consistency check between model and datapipe ---
        model_name = cfg.model._target_
        datapipe_name = cfg.datapipe._target_

        if "MeshGraphNet" in model_name and "GraphDataset" not in datapipe_name:
            raise ValueError(
                f"Model {model_name} requires a graph datapipe, "
                f"but you selected {datapipe_name}."
            )
        if "Transolver" in model_name and "PointCloudDataset" not in datapipe_name:
            raise ValueError(
                f"Model {model_name} requires a point-cloud datapipe, "
                f"but you selected {datapipe_name}."
            )
        if "FIGConvUNet" in model_name and "PointCloudDataset" not in datapipe_name:
            raise ValueError(
                f"Model {model_name} requires a point-cloud datapipe, "
                f"but you selected {datapipe_name}."
            )

        # Dataset
        reader = instantiate(cfg.reader)
        logging.getLogger().setLevel(logging.INFO)
        dataset = instantiate(
            cfg.datapipe,
            name="crash_train",
            reader=reader,
            split="train",
            logger=logger0,
        )
        logging.getLogger().setLevel(logging.INFO)
        # Move stats to device
        self.data_stats = dict(
            node={k: v.to(self.dist.device) for k, v in dataset.node_stats.items()},
            edge={
                k: v.to(self.dist.device)
                for k, v in getattr(dataset, "edge_stats", {}).items()
            },
            feature={
                k: v.to(self.dist.device)
                for k, v in getattr(dataset, "feature_stats", {}).items()
            },
        )

        # Sampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.dist.world_size,
            rank=self.dist.rank,
            shuffle=True,
        )

        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,  # variable N per sample
            shuffle=(sampler is None),
            drop_last=True,
            pin_memory=True,
            num_workers=cfg.training.num_dataloader_workers,
            sampler=sampler,
            collate_fn=simsample_collate,
        )
        self.sampler = sampler

        if cfg.training.num_validation_samples > 0:
            self.num_validation_replicas = min(
                self.dist.world_size, cfg.training.num_validation_samples
            )
            self.num_validation_samples = (
                cfg.training.num_validation_samples
                // self.num_validation_replicas
                * self.num_validation_replicas
            )
            logger0.info(f"Number of validation samples: {self.num_validation_samples}")

            # Create a validation dataset
            val_cfg = self.cfg.datapipe
            with open_dict(val_cfg):  # or open_dict(cfg) to open the whole tree
                val_cfg.data_dir = self.cfg.training.raw_data_dir_validation
                val_cfg.num_samples = self.num_validation_samples
            val_dataset = instantiate(
                val_cfg,
                name="crash_validation",
                reader=reader,
                split="validation",
                logger=logger0,
                sample_type="all_time_steps",  # always all_time_steps for validation
            )

            if self.dist.rank < self.num_validation_replicas:
                # Sampler
                if self.dist.world_size > 1:
                    sampler = DistributedSampler(
                        val_dataset,
                        num_replicas=self.num_validation_replicas,
                        rank=self.dist.rank,
                        shuffle=False,
                        drop_last=True,
                    )
                else:
                    sampler = None

                self.val_dataloader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=1,  # variable N per sample
                    shuffle=False,
                    drop_last=True,
                    pin_memory=True,
                    num_workers=cfg.training.num_dataloader_workers,
                    sampler=sampler,
                    collate_fn=simsample_collate,
                )
            else:
                self.val_dataloader = torch.utils.data.DataLoader(
                    torch.utils.data.Subset(val_dataset, []), batch_size=1
                )

        # Model
        self.model = instantiate(cfg.model)
        logging.getLogger().setLevel(logging.INFO)
        self.model.to(self.dist.device)
        self.model.train()

        # Log model summary and parameter count (optional: torchinfo)
        if self.dist.rank == 0:
            num_params = sum(p.numel() for p in self.model.parameters())
            logger0.info(f"Model parameters: {num_params:,}")
            if _torchinfo.available:
                try:
                    logger0.info(f"\n{_torchinfo.summary(self.model, verbose=0)}")
                except Exception:
                    logger0.info(
                        "(torchinfo summary skipped: model requires sample input)"
                    )

        # distributed data parallel for multi-node training
        if self.dist.world_size > 1:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[self.dist.local_rank],
                output_device=self.dist.device,
                broadcast_buffers=self.dist.broadcast_buffers,
                find_unused_parameters=self.dist.find_unused_parameters,
            )

        # Loss
        self.criterion = torch.nn.MSELoss()

        # Optimizer (adam or muon; muon requires PyTorch >= 2.9)
        opt_name = cfg.training.get("optimizer", "adam")
        assert opt_name in ["adam", "muon"], f"Unsupported optimizer: {opt_name}"
        if opt_name == "muon":
            self.optimizer = build_muon_optimizer(self.model, cfg)
        else:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=cfg.training.start_lr, fused=True
            )
        logger0.info(f"Using {self.optimizer.__class__.__name__} optimizer")

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.training.epochs, eta_min=cfg.training.end_lr
        )
        self.scaler = GradScaler("cuda", enabled=self.amp)

        # Checkpoint
        if self.dist.world_size > 1:
            torch.distributed.barrier()
        self.epoch_init = load_checkpoint(
            cfg.training.ckpt_path,
            models=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            device=self.dist.device,
        )

        if self.dist.rank == 0:
            wandb.init(
                entity=cfg.training.get("wandb_entity", "julius-riel-tum-ai"),
                project=cfg.training.get("wandb_project", "MIT-InitialRuns"),
                name=cfg.get("experiment_name", None),
                config=omegaconf.OmegaConf.to_container(cfg, resolve=True),
            )

    def train(self, sample: SimSample):
        self.optimizer.zero_grad()
        loss = self.forward(sample)
        self.backward(loss)
        return loss

    def forward(self, sample: SimSample):
        with autocast(device_type="cuda", enabled=self.amp):
            # Model forward - returns [N, T, Fo]
            pred = self.model(sample=sample, data_stats=self.data_stats)

            # Target is [N, T, Fo]
            target = sample.node_target
            return self.criterion(pred, target)

    def backward(self, loss):
        if self.amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

    @torch.no_grad()
    def validate(self, epoch):
        """Run validation error computation"""
        self.model.eval()

        MSE = torch.zeros(1, device=self.dist.device)
        MSE_w_time = torch.zeros(self.rollout_steps, device=self.dist.device)
        for idx, sample in enumerate(self.val_dataloader):
            sample = sample[0].to(self.dist.device)  # SimSample .to()

            # Model forward - returns [N, T, Fo]
            pred = self.model(sample=sample, data_stats=self.data_stats)

            # Target is [N, T, Fo]
            target = sample.node_target

            # Compute and add error
            SqError = torch.square(pred - target)
            MSE_w_time += torch.mean(
                SqError, dim=(0, 2)
            )  # mean over N, Fo per timestep
            MSE += torch.mean(SqError)

        # Sum errors across all ranks
        if self.dist.world_size > 1:
            torch.distributed.all_reduce(MSE, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(MSE_w_time, op=torch.distributed.ReduceOp.SUM)

        val_stats = {
            "MSE_w_time": MSE_w_time / self.num_validation_samples,
            "MSE": MSE / self.num_validation_samples,
        }

        self.model.train()  # Switch back to training mode
        return val_stats

    @torch.no_grad()
    def log_simulation_gifs(self, epoch: int, stride: int = 2, fps: int = 12):
        """Render GT-vs-Pred-vs-Error GIFs for all val samples, logged under simulation/."""
        self.model.eval()

        for sample_idx, batch in enumerate(self.val_dataloader):
            sample = batch[0].to(self.dist.device)
            pred = self.model(sample=sample, data_stats=self.data_stats)
            target = sample.node_target  # [N, T, Fo]

            pred_np   = pred.cpu().float().numpy()    # [N, T, Fo]
            target_np = target.cpu().float().numpy()  # [N, T, Fo]

            T = pred_np.shape[1]
            # layout: [:, t, 0:3] = coords, then optional dynamic targets
            # ([:, t, 3] = eps, [:, t, 4] = stress_vm when configured)
            gt_coords   = target_np[:, :, :3].transpose(1, 0, 2)  # [T, N, 3]
            pred_coords = pred_np[:, :, :3].transpose(1, 0, 2)    # [T, N, 3]
            if target_np.shape[2] > 4:
                gt_scalar   = target_np[:, :, 4].T                # [T, N]
                pred_scalar = pred_np[:, :, 4].T                  # [T, N]
                err_scalar  = np.abs(gt_scalar - pred_scalar)     # [T, N]
                p2_coords   = gt_coords
                labels = ("GT stress_vm", "Pred stress_vm", "|Error|")
            else:
                # coords-only targets: color by displacement magnitude from the
                # first rollout frame; error = per-node coordinate error
                gt_scalar   = np.linalg.norm(gt_coords - gt_coords[0], axis=-1)
                pred_scalar = np.linalg.norm(pred_coords - gt_coords[0], axis=-1)
                err_scalar  = np.linalg.norm(pred_coords - gt_coords, axis=-1)
                p2_coords   = pred_coords
                labels = ("GT |disp|", "Pred |disp|", "|coord error|")

            frame_idx = list(range(0, T, max(1, stride)))
            ext_coords = np.concatenate(
                [gt_coords[frame_idx], p2_coords[frame_idx]], axis=1
            )
            hext, vext = _gif_extents(ext_coords, ax_h=1, ax_v=0)
            vmax = max(float(np.percentile(gt_scalar[frame_idx], 99)),
                       float(np.percentile(pred_scalar[frame_idx], 99))) or 1.0
            vmax_err = float(np.percentile(err_scalar[frame_idx], 99)) or 1.0

            frames = []
            for t in frame_idx:
                p1 = _gif_panel(gt_coords[t], gt_scalar[t],    1, 0, hext, vext,
                                "inferno", 0.0, vmax,    labels[0])
                p2 = _gif_panel(p2_coords[t], pred_scalar[t],  1, 0, hext, vext,
                                "inferno", 0.0, vmax,    labels[1])
                p3 = _gif_panel(gt_coords[t], err_scalar[t],   1, 0, hext, vext,
                                "inferno", 0.0, vmax_err, labels[2])
                frames.append(_gif_compose(
                    [p1, p2, p3],
                    f"Epoch {epoch + 1} | sample {sample_idx} | t={t}/{T - 1}",
                ))

            with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
                tmp = f.name
            frames[0].save(tmp, save_all=True, append_images=frames[1:],
                           duration=int(1000 / fps), loop=0, optimize=True)
            wandb.log(
                {f"simulation/sample_{sample_idx}": wandb.Video(tmp, fps=fps, format="gif")},
                step=epoch,
            )
            os.unlink(tmp)

        self.model.train()


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    DistributedManager.initialize()
    dist = DistributedManager()

    logger = PythonLogger("main")
    logger0 = RankZeroLoggingWrapper(logger, dist)
    logger0.file_logging()

    # Log full config and paths
    logger0.info(f"Config:\n{omegaconf.OmegaConf.to_yaml(cfg, resolve=True)}")
    logger0.info(f"W&B run: {cfg.training.get('wandb_entity', 'julius-riel-tum-ai')}/{cfg.training.get('wandb_project', 'MIT-InitialRuns')}")
    logger0.info(f"Checkpoint directory: {cfg.training.ckpt_path}")
    stats_dir = getattr(cfg.datapipe, "stats_dir")
    logger0.info(f"Stats directory: {stats_dir}")

    trainer = Trainer(cfg, logger0)
    logger0.info("Training started...")

    for epoch in range(trainer.epoch_init, cfg.training.epochs):
        if trainer.sampler is not None:
            trainer.sampler.set_epoch(epoch)

        total_loss = 0.0
        num_batches = 0
        start = time.time()
        batch_start = start
        epoch_len = len(trainer.dataloader)
        log_every = max(1, epoch_len // 10)  # Log ~10 times per epoch

        for batch_idx, sample in enumerate(trainer.dataloader):
            sample = sample[0].to(dist.device)  # SimSample .to()
            loss = trainer.train(sample)
            total_loss += loss.detach().item()
            num_batches += 1

            # Per-batch progress
            if (batch_idx + 1) % log_every == 0 or batch_idx == 0:
                batch_duration = time.time() - batch_start
                mem_gb = (
                    torch.cuda.memory_reserved() / 1024**3
                    if torch.cuda.is_available()
                    else 0.0
                )
                logger0.info(
                    f"Epoch {epoch + 1} [{batch_idx + 1}/{epoch_len}] "
                    f"Loss: {loss.detach().item():.6f} "
                    f"Duration: {batch_duration:.2f}s Mem: {mem_gb:.2f}GB"
                )
            batch_start = time.time()

        trainer.scheduler.step()

        avg_loss = total_loss / max(num_batches, 1)
        epoch_duration = time.time() - start
        logger0.info(
            f"Epoch {epoch + 1}/{cfg.training.epochs} "
            f"avg_loss: {avg_loss:.6f} "
            f"lr: {trainer.optimizer.param_groups[0]['lr']:.3e} "
            f"duration: {epoch_duration:.2f}s"
        )

        if dist.rank == 0:
            wandb.log(
                {
                    "train/loss": avg_loss,
                    "train/lr": trainer.optimizer.param_groups[0]["lr"],
                },
                step=epoch,
            )

        if dist.world_size > 1:
            torch.distributed.barrier()

        if dist.rank == 0 and (epoch + 1) % cfg.training.save_checkpoint_freq == 0:
            save_checkpoint(
                cfg.training.ckpt_path,
                models=trainer.model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                scaler=trainer.scaler,
                epoch=epoch + 1,
            )
            logger.info(f"Saved model on rank {dist.rank}")

        # Validation
        if (
            cfg.training.num_validation_samples > 0
            and (epoch + 1) % cfg.training.validation_freq == 0
        ):
            val_stats = trainer.validate(epoch)

            # Log validation metrics
            mse_val = val_stats["MSE"].item()
            mse_w_time = val_stats["MSE_w_time"]
            logger0.info(f"Validation epoch {epoch + 1}: MSE: {mse_val:.6f}")
            if _tabulate.available and dist.rank == 0:
                rows = [["MSE (overall)", f"{mse_val:.6f}"]]
                for i, m in enumerate(mse_w_time):
                    rows.append([f"timestep_{i}_MSE", f"{m.item():.6f}"])
                logger0.info(
                    f"\nValidation metrics:\n{_tabulate.tabulate(rows, headers=['Metric', 'Value'], tablefmt='pretty')}\n"
                )

            if dist.rank == 0:
                per_t = {
                    f"val/timestep_{i}_mse": val_stats["MSE_w_time"][i].item()
                    for i in range(len(val_stats["MSE_w_time"]))
                }
                wandb.log({"val/loss": val_stats["MSE"].item(), "val/mse": val_stats["MSE"].item(), **per_t}, step=epoch)
                trainer.log_simulation_gifs(epoch)

    logger0.info("Training completed!")
    if dist.rank == 0:
        wandb.finish()


if __name__ == "__main__":
    main()
