"""
Training loop — OPV2V real data only.
Full diagnostic CSV with ~35 metrics per epoch.
"""
from __future__ import annotations
import csv
import os
import random
import numpy as np
import math
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import Config
from .models.coop_detector import CoopDetector
from .loss.coop_loss import coop_loss
from .metrics import evaluate, _to_device
from .data.opv2v import OPV2VDataset, opv2v_collate


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _pick_device(cfg) -> torch.device:
    if cfg.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_loaders(cfg: Config):
    from dataclasses import replace as _replace
    # val/test never use depth labels (evaluate() has no depth term); "none" also
    # avoids the LiDAR frame filter, which would empty the test split (no .pcd)
    _cfg_noaug = _replace(cfg, augment=False, depth_source="none")
    tr = OPV2VDataset(cfg.train_dir,      cfg,
                      max_agents=cfg.max_agents, use_gt_pose=cfg.use_gt_pose)
    va = OPV2VDataset(cfg.val_dir,   _cfg_noaug,
                      max_agents=cfg.max_agents, use_gt_pose=cfg.use_gt_pose)
    te = OPV2VDataset(cfg.test_dir,  _cfg_noaug,
                      max_agents=cfg.max_agents, use_gt_pose=cfg.use_gt_pose)
    kw = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        collate_fn=opv2v_collate,
        pin_memory=(cfg.device == "cuda"),
    )
    return (DataLoader(tr, shuffle=True,  **kw),
            DataLoader(va, shuffle=False, **kw),
            DataLoader(te, shuffle=False, **kw))


def lr_at_epoch(epoch: int, cfg: Config, warmup_epochs: int = 5) -> float:
    """Linear LR warmup over the first `warmup_epochs`, then cosine decay
    to eta_min over the remaining epochs. Computed directly from `epoch`
    so it is exactly resume-safe -- no scheduler internal state needed."""
    eta_min = 1e-6
    if epoch < warmup_epochs:
        warmup_frac = (epoch + 1) / max(warmup_epochs, 1)
        return cfg.lr * warmup_frac
    progress = (epoch - warmup_epochs) / max(1, cfg.epochs - warmup_epochs)
    progress = min(max(progress, 0.0), 1.0)
    return eta_min + 0.5 * (cfg.lr - eta_min) * (1 + math.cos(math.pi * progress))


def lambda_schedule(epoch: int, cfg: Config) -> float:
    return 0.0 if epoch < cfg.warmup_epochs else cfg.lambda_bw


def gumbel_T(epoch: int, cfg: Config) -> float:
    """Linear anneal T0 -> T1 over `gumbel_epochs` (default: cfg.epochs).

    The anneal horizon is separate from cfg.epochs because the LR cosine may run over a
    longer horizon than training actually does -- the final round uses epochs=120 for the
    LR schedule but stops at max_epoch=59. Tying T to cfg.epochs there would end training
    at T=2.77, so selection would never reach its final temperature.
    """
    n = cfg.gumbel_epochs if getattr(cfg, "gumbel_epochs", -1) > 0 else cfg.epochs
    if n <= 1:
        return cfg.gumbel_T1
    f = min(max(epoch / (n - 1), 0.0), 1.0)
    return cfg.gumbel_T0 + f * (cfg.gumbel_T1 - cfg.gumbel_T0)


# Full CSV header — 35 metrics
_CSV_HEADER = [
    # training
    "epoch", "lr", "lam", "T",
    "loss", "loss_det", "loss_bw", "loss_depth",
    "loss_hm", "loss_off", "loss_sz", "loss_yaw", "loss_zh",
    # detection
    "ap30", "ap50", "ap70",
    "n_gt", "n_pred", "tp50", "fp50", "fn50",
    "precision50", "recall50",
    # OOF
    "oof_recall", "n_oof_gt",
    # localisation
    "iou_mean", "iou_median", "iou_std",
    "centre_rmse", "w_rmse", "l_rmse", "yaw_rmse",
    # heatmap
    "heat_max", "heat_mean", "heat_std",
    "heat_pos_mean", "heat_neg_mean", "val_loss_det",
    # decode
    "n_peaks_mean", "score_max", "score_mean",
    # communication
    "kb", "n_cells_mean",
    # histograms (string columns)
    "iou_hist", "conf_hist",
    # runtime
    "val_time_s",
]


def train(cfg: Config, train_loader=None, val_loader=None, test_loader=None):
    cfg.validate()
    cfg.save()
    _set_seed(cfg.seed)
    device = _pick_device(cfg)

    if train_loader is None:
        train_loader, val_loader, test_loader = build_loaders(cfg)

    model = CoopDetector(cfg).to(device)

    ck_dir  = f"{cfg.root}/checkpoints/{cfg.exp_name}"
    log_dir = f"{cfg.root}/logs/{cfg.exp_name}"
    os.makedirs(ck_dir,  exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    start_epoch = 0
    best_ap50   = -1.0
    last_ck     = f"{ck_dir}/last.pth"

    if cfg.resume and os.path.exists(last_ck):
        ck = torch.load(last_ck, map_location=device)
        state = ck["model"]
        if hasattr(model, "module"):
            model.module.load_state_dict(state)
        else:
            model.load_state_dict(state)
        start_epoch = ck["epoch"] + 1
        best_ap50   = ck.get("best_ap50", ck.get("best_ap70", -1.0))
        print(f"[{cfg.exp_name}] Resumed from epoch {start_epoch - 1}")

    # Differential learning rates: pretrained backbone at 0.1x LR,
    # randomly-initialised head/fusion/utility at full LR.
    backbone_ids = set(id(p) for p in model.backbone.parameters())
    params = [
        {"params": [p for p in model.parameters()
                    if p.requires_grad and id(p) not in backbone_ids],
         "lr": cfg.lr},
        {"params": [p for p in model.backbone.parameters() if p.requires_grad],
         "lr": cfg.lr * 0.1},
    ]

    lr_mults = [g["lr"] / cfg.lr for g in params]
    opt    = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.resume and os.path.exists(last_ck) and "optimizer" in ck:
        opt.load_state_dict(ck["optimizer"])
        print(f"[{cfg.exp_name}] Restored optimizer state from epoch {start_epoch - 1}")
    scaler = GradScaler(enabled=False)

    log_path     = f"{log_dir}/train_log.csv"
    write_header = not os.path.exists(log_path) or not cfg.resume
    with open(log_path, "a" if cfg.resume else "w", newline="") as f:
        if write_header:
            csv.writer(f).writerow(_CSV_HEADER)
    

    last_val = None
    
    for epoch in range(start_epoch, cfg.epochs):
        if getattr(cfg, "max_epoch", -1) >= 0 and epoch > cfg.max_epoch:
            print(f"[{cfg.exp_name}] max_epoch={cfg.max_epoch} reached; stopping (LR schedule still uses epochs={cfg.epochs})")
            break
        cur_lr = lr_at_epoch(epoch, cfg, getattr(cfg, "lr_warmup_epochs", 5))
        for _g, _mult in zip(opt.param_groups, lr_mults):
            _g["lr"] = cur_lr * _mult
        model.train()
        lam = lambda_schedule(epoch, cfg)
        T   = gumbel_T(epoch, cfg)

        losses, parts_accum = [], {}
        
        
        
        
        accum_steps = getattr(cfg, "grad_accum_steps", 1)
        opt.zero_grad(set_to_none=True)
        for step_i, scene in enumerate(tqdm(train_loader,
                          desc=f"[{cfg.exp_name}] epoch {epoch}",
                          leave=True)):
            scene = _to_device(scene, device)
            with autocast(enabled=False):
                out  = model(scene, hard=False, gumbel_T=T)
                loss, parts = coop_loss(out, scene, cfg, lam=lam)
            if not torch.isfinite(loss):
                continue
            scaler.scale(loss / accum_steps).backward()

            if (step_i + 1) % accum_steps == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    [p for g in params for p in g["params"]], cfg.grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            losses.append(parts["total"])
            for k, v in parts.items():
                parts_accum.setdefault(k, []).append(v)

        # flush any remaining accumulated gradients at epoch end
        if (step_i + 1) % accum_steps != 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                [p for g in params for p in g["params"]], cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

        # Validate every val_every epochs (default 2) to save time
        do_val = (
            (epoch % getattr(cfg, "val_every", 2) == 0)
            or (epoch == cfg.epochs - 1)
        )

        if do_val:
            val = evaluate(
                model,
                val_loader,
                cfg,
                hard=True,
                tau=cfg.tau,
                budget_frac=cfg.budget_frac,
            )
            last_val = val

        else:
            if last_val is None:
                val = {
                    "ap30": float("nan"),
                    "ap50": float("nan"),
                    "ap70": float("nan"),
                    "n_gt": float("nan"),
                    "n_pred": float("nan"),
                    "tp50": float("nan"),
                    "fp50": float("nan"),
                    "fn50": float("nan"),
                    "precision50": float("nan"),
                    "recall50": float("nan"),
                    "oof_recall": float("nan"),
                    "n_oof_gt": float("nan"),
                    "iou_mean": float("nan"),
                    "iou_median": float("nan"),
                    "iou_std": float("nan"),
                    "centre_rmse": float("nan"),
                    "w_rmse": float("nan"),
                    "l_rmse": float("nan"),
                    "yaw_rmse": float("nan"),
                    "heat_max": float("nan"),
                    "heat_mean": float("nan"),
                    "heat_std": float("nan"),
                    "heat_pos_mean": float("nan"),
                    "heat_neg_mean": float("nan"),
                    "val_loss_det": float("nan"),
                    "n_peaks_mean": float("nan"),
                    "score_max": float("nan"),
                    "score_mean": float("nan"),
                    "kb": float("nan"),
                    "n_cells_mean": float("nan"),
                    "iou_hist": "",
                    "conf_hist": "",
                    "val_time_s": float("nan"),
                }
            else:
                val = last_val



        mean_loss  = float(np.mean(losses)) if losses else 0.0
        mean_det   = float(np.mean(parts_accum.get("det",      [0.0])))
        mean_bw    = float(np.mean(parts_accum.get("bw_bytes", [0.0])))
        mean_depth = float(np.mean(parts_accum.get("depth",    [0.0])))
        mean_hm  = float(np.mean(parts_accum.get("det_hm",  [0.0])))
        mean_off = float(np.mean(parts_accum.get("det_off", [0.0])))
        mean_sz  = float(np.mean(parts_accum.get("det_sz",  [0.0])))
        mean_yaw = float(np.mean(parts_accum.get("det_yaw", [0.0])))
        mean_zh  = float(np.mean(parts_accum.get("det_zh",  [0.0])))

        row = [
            epoch, cur_lr, lam, T,
            mean_loss, mean_det, mean_bw, mean_depth,
            mean_hm, mean_off, mean_sz, mean_yaw, mean_zh,
            val["ap30"], val["ap50"], val["ap70"],
            val["n_gt"], val["n_pred"],
            val["tp50"], val["fp50"], val["fn50"],
            val["precision50"], val["recall50"],
            val["oof_recall"], val["n_oof_gt"],
            val["iou_mean"], val["iou_median"], val["iou_std"],
            val["centre_rmse"], val["w_rmse"], val["l_rmse"], val["yaw_rmse"],
            val["heat_max"], val["heat_mean"], val["heat_std"],
            val["heat_pos_mean"], val["heat_neg_mean"], val["val_loss_det"],
            val["n_peaks_mean"], val["score_max"], val["score_mean"],
            val["kb"], val["n_cells_mean"],
            val["iou_hist"], val["conf_hist"],
            val["val_time_s"],
        ]
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

        # Update best metric first
        if do_val and val["ap50"] > best_ap50:
            best_ap50 = val["ap50"]

        ck_payload = {
            "model": model.module.state_dict() if hasattr(model, "module")
                    else model.state_dict(),
            "optimizer": opt.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_ap50": best_ap50,
        }

 


        # Only keep last.pth + best.pth (per-epoch checkpoints removed to save disk)
        torch.save(ck_payload, last_ck)
        # Periodic checkpoint every cfg.save_every epochs (default 20; for post-hoc evaluation),
        # plus EVERY epoch from cfg.save_all_from on. The final round reports the mean +/- sd over
        # the last five epochs rather than one final checkpoint, so each of those must be kept.
        _sa = getattr(cfg, "save_all_from", -1)
        if (epoch > 0 and epoch % getattr(cfg, "save_every", 20) == 0) or (_sa >= 0 and epoch >= _sa):
            torch.save(ck_payload, f"{ck_dir}/epoch_{epoch}.pth")

        # Save best checkpoint
        if do_val and val["ap50"] == best_ap50:
            torch.save(ck_payload, f"{ck_dir}/best.pth")

        # Rich print
        print(f"\n[{cfg.exp_name}] epoch {epoch:3d}  lam={lam:.1e}  T={T:.2f}")
        print(f"  loss={mean_loss:.4f}  (det={mean_det:.4f}  bw={mean_bw:.2e}  depth={mean_depth:.4f})  val_loss={val['val_loss_det']:.4f}")
        print(f"  AP30/50/70={val['ap30']:.3f}/{val['ap50']:.3f}/{val['ap70']:.3f}  OOF={val['oof_recall']:.3f}")
        print(f"  GT={val['n_gt']}  Pred={val['n_pred']}  TP={val['tp50']}  FP={val['fp50']}  FN={val['fn50']}")
        print(f"  Precision={val['precision50']:.3f}  Recall={val['recall50']:.3f}")
        print(f"  IoU mean={val['iou_mean']:.3f}  median={val['iou_median']:.3f}")
        print(f"  CentreRMSE={val['centre_rmse']:.2f}m  W_RMSE={val['w_rmse']:.2f}m  L_RMSE={val['l_rmse']:.2f}m  Yaw_RMSE={val['yaw_rmse']:.1f}°")
        print(f"  Heat max={val['heat_max']:.4f}  pos={val['heat_pos_mean']:.4f}  neg={val['heat_neg_mean']:.4f}")
        print(f"  Peaks={val['n_peaks_mean']:.1f}  ScoreMax={val['score_max']:.3f}  ScoreMean={val['score_mean']:.3f}")
        print(f"  KB={val['kb']:.1f}  val_time={val['val_time_s']:.0f}s")
        print("-" * 70)

    # Return best checkpoint
    best_path = f"{ck_dir}/best.pth"
    if os.path.exists(best_path):
        ck = torch.load(best_path, map_location=device)
        state = ck["model"]
        if hasattr(model, "module"):
            model.module.load_state_dict(state)
        else:
            model.load_state_dict(state)
        print(f"[{cfg.exp_name}] Loaded best checkpoint (AP@0.5={best_ap50:.3f})")

    return model
