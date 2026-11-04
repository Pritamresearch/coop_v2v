"""
OPV2V Dataset Loader  —  production implementation for Kelvin2
==============================================================

Reads the real OPV2V data from::

    /mnt/scratch2/users/2641309/datasets/OPV2V/
        DATA_train/train_chunks/train/
        DATA_validate/validate_chunks/validate/
        DATA_test/test_chunks/test/

YAML format (one file per CAV per timestamp)::

    camera0 / camera1 / camera2 / camera3:
        intrinsic:  [[fx,0,cx],[0,fy,cy],[0,0,1]]   # 3x3 matrix
        extrinsic:  [[...]]                          # 4x4  cam -> LiDAR
        cords:      [x,y,z,roll,yaw,pitch]           # CARLA world coords

    lidar_pose:       [x,y,z,roll,yaw,pitch]         # CARLA world coords
    true_ego_pos:     [x,y,z,roll,yaw,pitch]         # ground-truth pose
    predicted_ego_pos:[x,y,z,roll,yaw,pitch]         # GPS-noisy pose

    vehicles:
        <id>:
            location: [x,y,z]          # CARLA world, box front-axis center
            angle:    [roll,yaw,pitch]  # degrees
            extent:   [half_l,half_w,half_h]
            center:   [dx,dy,dz]       # offset from front-axis center to bbox center

Coordinate frames used throughout
----------------------------------
- CARLA world frame   : x=east, y=south, z=up  (left-handed; yaw CCW when viewed from above)
- LiDAR frame per CAV : standard OpenCOOD (forward=x, left=y, up=z)
- Ego BEV frame       : same as LiDAR, 200×200 grid at 0.5 m/cell centred on ego LiDAR

Pipeline outputs (per __getitem__)
-----------------------------------
See the scene-dict contract at the bottom of this file.
All shapes quoted for (N agents, 4 cams, ...):

    images        (N, 4, 3, H, W)     float32  [0,1]
    intrins       (N, 4, 3, 3)        float32  camera intrinsic K
    rots          (N, 4, 3, 3)        float32  cam -> ego (LiDAR) rotation
    trans         (N, 4, 3)           float32  cam -> ego translation (m)
    post_rots     None                (no augmentation applied here)
    post_trans    None
    rel_pose      (N, 3)              float32  [dx, dy, dyaw] each agent in ego frame
    depth_gt_img  (N, 4, fH, fW)     int64    per-pixel depth-bin index; -1 = ignore
    gt:
        heat      (H, W)              float32  Gaussian heatmap in [0,1]
        offset    (2, H, W)           float32  sub-cell (dx,dy)
        size      (2, H, W)           float32  log(w), log(l)
        yaw       (2, H, W)           float32  (sin, cos) of heading
        zh        (2, H, W)           float32  (z_centre, height)
        mask      (H, W)              float32  1 at object centres
        boxes     (M, 7)              float32  (cx,cy,cz,w,l,h,yaw) in ego metres
        oof       (M,)                bool     True if outside ego physical camera FoV
    n_agents      int
"""
from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

# ------------------------------------------------------------------ #
# Coordinate helpers                                                   #
# ------------------------------------------------------------------ #

def _deg2rad(d: float) -> float:
    return d * math.pi / 180.0


def _carla_rpy_to_R(roll_deg: float, yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """
    Build a 3×3 rotation matrix from CARLA's (roll, yaw, pitch) in degrees.

    CARLA convention (left-handed world, OpenCOOD docs):
        yaw   : rotation about Z  (CCW positive when viewed from above in CARLA world)
        pitch : rotation about Y
        roll  : rotation about X
    We use the standard ZYX Euler convention used in OpenCOOD.
    """
    r = _deg2rad(roll_deg)
    y = _deg2rad(yaw_deg)
    p = _deg2rad(pitch_deg)

    cy, sy = math.cos(y), math.sin(y)
    cp, sp = math.cos(p), math.sin(p)
    cr, sr = math.cos(r), math.sin(r)

    # ZYX: R = Rz(y) @ Ry(p) @ Rx(r)
    R = np.array([
        [cy * cp,  cy * sp * sr - sy * cr,  cy * sp * cr + sy * sr],
        [sy * cp,  sy * sp * sr + cy * cr,  sy * sp * cr - cy * sr],
        [-sp,      cp * sr,                  cp * cr              ],
    ], dtype=np.float32)
    return R


def _lidar_pose_to_T(pose6: list) -> np.ndarray:
    """
    Convert a 6-DOF CARLA pose [x,y,z,roll,yaw,pitch] to a 4×4
    world->sensor transform (i.e. T_world_lidar).
    """
    x, y, z, roll, yaw, pitch = pose6
    R = _carla_rpy_to_R(roll, yaw, pitch)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3,  3] = [x, y, z]
    return T


def _relative_pose(ego_T: np.ndarray, other_T: np.ndarray) -> np.ndarray:
    """
    Compute the pose of 'other' in the ego LiDAR frame.
    Returns (dx, dy, dyaw) in metres / radians.
    ego_T, other_T : 4×4 world->LiDAR transforms.
    """
    # other in ego frame: T_ego_other = inv(T_world_ego) @ T_world_other
    T_rel = np.linalg.inv(ego_T) @ other_T
    dx   = float(T_rel[0, 3])
    dy   = float(T_rel[1, 3])
    dyaw = float(math.atan2(T_rel[1, 0], T_rel[0, 0]))
    return np.array([dx, dy, dyaw], dtype=np.float32)


def _cam_extr_to_rot_trans(
    extrinsic_4x4: np.ndarray,
    lidar_T_world: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    OPV2V extrinsic is T_cam_lidar  (4×4, camera FROM lidar).
    LSS expects  rots  = R_cam2ego  (i.e. cam -> ego/LiDAR rotation, 3×3)
                 trans = t_cam2ego  (i.e. cam origin in ego frame, 3)

    T_cam_lidar maps points  p_cam = T_cam_lidar @ p_lidar.
    So lidar-to-cam: T_lc = extrinsic_4x4
        cam-to-lidar: T_cl = inv(extrinsic_4x4)
    rots  = T_cl[:3, :3]
    trans = T_cl[:3,  3]
    """
    T_cl = np.linalg.inv(extrinsic_4x4.astype(np.float64)).astype(np.float32)
    rots  = T_cl[:3, :3]
    trans = T_cl[:3,  3]
    return rots, trans


# ------------------------------------------------------------------ #
# BEV Ground-truth builder                                            #
# ------------------------------------------------------------------ #

def _gaussian_2d(h: int, w: int, cy: float, cx: float, sigma: float) -> np.ndarray:
    """Render a Gaussian centred at (cy, cx) into an (h, w) array."""
    x  = np.arange(w, dtype=np.float32)
    y  = np.arange(h, dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    return np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))



def _build_bev_gt_from_boxes(boxes_np: np.ndarray, cfg) -> dict:
    """
    Rebuild BEV GT tensors from a (M,7) array of [cx,cy,cz,w,l,h,yaw]
    in ego-frame metres. Used after BEV rotation augmentation.
    """
    from src.data.bev_augment import draw_gaussian_sized
    H, W = cfg.grid_H, cfg.grid_W
    cell = cfg.cell_m
    heat   = np.zeros((H, W), dtype=np.float32)
    offset = np.zeros((2, H, W), dtype=np.float32)
    size_m = np.zeros((2, H, W), dtype=np.float32)
    yaw_t  = np.zeros((2, H, W), dtype=np.float32)
    zh     = np.zeros((2, H, W), dtype=np.float32)
    mask   = np.zeros((H, W), dtype=np.float32)

    if boxes_np is None or len(boxes_np) == 0:
        return {"heat": heat, "offset": offset, "size": size_m,
                "yaw": yaw_t, "zh": zh, "mask": mask,
                "boxes": np.zeros((0, 7), dtype=np.float32)}

    for b in boxes_np:
        cx, cy, cz, w, l, h, yaw = b
        col_f = (cx - cfg.bev_x_min) / cell
        row_f = (cy - cfg.bev_y_min) / cell
        col, row = int(round(col_f)), int(round(row_f))
        if not (0 <= col < W and 0 <= row < H):
            continue
        if abs(cx) < 3.0 and abs(cy) < 2.0:
            continue
        draw_gaussian_sized(heat, row_f, col_f, w, l, cell,
                            centerpoint=getattr(cfg, "gauss_mode", "legacy") == "centerpoint")
        mask[row, col]      = 1.0
        offset[0, row, col] = col_f - col
        offset[1, row, col] = row_f - row
        size_m[0, row, col] = math.log(max(w, 0.1))
        size_m[1, row, col] = math.log(max(l, 0.1))
        yaw_t[0, row, col]  = math.sin(yaw)
        yaw_t[1, row, col]  = math.cos(yaw)
        zh[0, row, col]     = cz
        zh[1, row, col]     = h

    return {"heat": heat, "offset": offset, "size": size_m,
            "yaw": yaw_t, "zh": zh, "mask": mask, "boxes": boxes_np}

def _build_bev_gt(
    vehicles: dict,
    ego_T_world: np.ndarray,          # 4×4  world->ego-LiDAR
    cfg,
    ego_T_world_fwd: np.ndarray = None,   # 4x4 T_world_ego (forward, for correct yaw)
) -> dict:
    """
    Project CARLA-world vehicle annotations into the ego BEV frame
    and build the CenterPoint-style heatmap / regression targets.

    Returns a dict with keys: heat, offset, size, yaw, zh, mask, boxes, oof
    ego_T_world : inv of T_world_ego  (maps world points into ego-LiDAR frame)
    ego_T_world_fwd : 4x4 T_world_ego (forward, for correct yaw)
    """
    H, W = cfg.grid_H, cfg.grid_W
    cell = cfg.cell_m
    x_min, y_min = cfg.bev_x_min, cfg.bev_y_min

    heat   = np.zeros((H, W),  dtype=np.float32)
    offset = np.zeros((2, H, W), dtype=np.float32)
    size_m = np.zeros((2, H, W), dtype=np.float32)
    yaw_t  = np.zeros((2, H, W), dtype=np.float32)
    zh     = np.zeros((2, H, W), dtype=np.float32)
    mask   = np.zeros((H, W),  dtype=np.float32)
    boxes_list: List[List[float]] = []

    for vid, vdata in vehicles.items():
        loc   = np.array(vdata["location"], dtype=np.float64)   # [x,y,z] world
        angle = vdata["angle"]                                    # [roll, yaw, pitch] deg
        ext   = vdata["extent"]                                   # [half_l, half_w, half_h]
        ctr   = np.array(vdata.get("center", [0., 0., 0.]),
                         dtype=np.float64)                        # offset in vehicle frame

        # full 3-D box centre in world frame (location is front-axis; add center offset)
        R_obj = _carla_rpy_to_R(angle[0], angle[1], angle[2]).astype(np.float64)
        world_pt = loc + R_obj @ ctr                              # (3,)

        # transform to ego LiDAR frame
        world_h = np.array([world_pt[0], world_pt[1], world_pt[2], 1.0])
        ego_pt  = (ego_T_world @ world_h)[:3]                    # (3,)

        cx_m, cy_m, cz_m = float(ego_pt[0]), float(ego_pt[1]), float(ego_pt[2])

        # pixel coords in BEV
        col_f = (cx_m - x_min) / cell
        row_f = (cy_m - y_min) / cell
        col   = int(round(col_f))
        row   = int(round(row_f))
        if not (0 <= col < W and 0 <= row < H):
            continue
        # Skip ego vehicle body -- boxes within 3m x 2m of ego origin
        # are never valid detection targets but create ghost detections.
        if abs(cx_m) < 3.0 and abs(cy_m) < 2.0:
            continue

        half_l, half_w, half_h = [float(e) for e in ext]
        l_m = 2.0 * half_l
        w_m = 2.0 * half_w
        h_m = 2.0 * half_h

        # yaw in ego frame: heading of this vehicle
        # The vehicle yaw in CARLA world; apply ego inverse
        # Exact yaw in ego frame: rotate vehicle forward vector through ego R
        # Replaces the approximate angle-subtraction which breaks under BEV rotation augmentation
        v_fwd_world = R_obj @ np.array([1.0, 0.0, 0.0])   # vehicle forward in world
        R_ego_from_world = ego_T_world[:3, :3]              # world->ego rotation
        v_fwd_ego = R_ego_from_world @ v_fwd_world          # vehicle forward in ego
        yaw_ego = math.atan2(float(v_fwd_ego[1]), float(v_fwd_ego[0]))  # exact heading

        if getattr(cfg, "gauss_mode", "legacy") == "centerpoint":
            # same size-adaptive target as the (augmented) training path
            from src.data.bev_augment import draw_gaussian_sized
            draw_gaussian_sized(heat, row_f, col_f, w_m, l_m, cell, centerpoint=True)
        else:
            # draw heatmap Gaussian (sigma = 2 cells)
            g = _gaussian_2d(H, W, row_f, col_f, sigma=2.0)
            np.maximum(heat, g, out=heat)
            heat[row, col] = 1.0  # ensure exact 1.0 at centre for focal loss

        mask[row, col]      = 1.0
        offset[0, row, col] = col_f - col
        offset[1, row, col] = row_f - row
        size_m[0, row, col] = math.log(max(w_m, 0.1))
        size_m[1, row, col] = math.log(max(l_m, 0.1))
        yaw_t[0, row, col]  = math.sin(yaw_ego)
        yaw_t[1, row, col]  = math.cos(yaw_ego)
        zh[0, row, col]     = cz_m
        zh[1, row, col]     = h_m

        boxes_list.append([cx_m, cy_m, cz_m, w_m, l_m, h_m, yaw_ego])

    boxes = np.array(boxes_list, dtype=np.float32) if boxes_list else np.zeros((0, 7), dtype=np.float32)
    return {
        "heat":   heat,
        "offset": offset,
        "size":   size_m,
        "yaw":    yaw_t,
        "zh":     zh,
        "mask":   mask,
        "boxes":  boxes,
    }


# ------------------------------------------------------------------ #
# Out-of-FoV flag                                                     #
# ------------------------------------------------------------------ #

def _check_oof(
    boxes: np.ndarray,          # (M, 7)  cx,cy,cz,w,l,h,yaw  in ego LiDAR frame
    rots_ego: np.ndarray,       # (4, 3, 3) cam->ego rotation
    trans_ego: np.ndarray,      # (4, 3)   cam->ego translation
    intrins: np.ndarray,        # (4, 3, 3) camera intrinsics
    H_img: int,
    W_img: int,
) -> np.ndarray:
    """
    For each GT box check if it projects into at least one of the 4 ego cameras.
    Returns bool array (M,): True = object is OUTSIDE all ego cameras (out-of-FoV).
    """
    M = len(boxes)
    if M == 0:
        return np.zeros(0, dtype=bool)

    oof = np.ones(M, dtype=bool)                # assume OOF until proven visible

    for j in range(M):
        cx, cy, cz = boxes[j, 0], boxes[j, 1], boxes[j, 2]
        pt_ego = np.array([cx, cy, cz], dtype=np.float64)

        for cam in range(4):
            R = rots_ego[cam].astype(np.float64)    # cam->ego
            t = trans_ego[cam].astype(np.float64)   # cam origin in ego
            K = intrins[cam].astype(np.float64)

            # pt in cam frame: R^T @ (pt_ego - t)   [since R is cam->ego]
            pt_cam = R.T @ (pt_ego - t)
            depth  = pt_cam[0]                      # x-forward convention
            if depth < 0.5:
                continue                             # behind camera

            # project: standard pinhole, cam frame x=forward
            u = K[0, 0] * pt_cam[1] / depth + K[0, 2]
            v = K[1, 1] * (-pt_cam[2]) / depth + K[1, 2]
            if 0 <= u < W_img and 0 <= v < H_img:
                oof[j] = False
                break
    return oof


# ------------------------------------------------------------------ #
# Depth GT from LiDAR PCD                                             #
# ------------------------------------------------------------------ #

def _build_depth_gt(
    pcd_path: str,
    rots_ego: np.ndarray,    # (4,3,3) cam->ego
    trans_ego: np.ndarray,   # (4,3)
    intrins: np.ndarray,     # (4,3,3)
    H_img: int,
    W_img: int,
    downsample: int,
    d_min: float,
    d_max: float,
    depth_K: int,
) -> np.ndarray:
    """
    Project the agent's own LiDAR point cloud into each camera to get per-pixel
    depth-bin indices for L_depth supervision.
    Returns (4, fH, fW) int64; -1 = no LiDAR return / out of range.
    """
    fH, fW = H_img // downsample, W_img // downsample
    result = -np.ones((4, fH, fW), dtype=np.int64)

    if not os.path.exists(pcd_path):
        return result

    try:
        pts = _load_pcd(pcd_path)                    # (N,3)
    except Exception:
        return result

    # pts are in LiDAR (ego) frame
    d_edges = np.linspace(d_min, d_max, depth_K + 1)

    for cam in range(4):
        R = rots_ego[cam].astype(np.float64)
        t = trans_ego[cam].astype(np.float64)
        K = intrins[cam].astype(np.float64)

        # transform: pt_cam = R^T @ (pt_lidar - t)
        pts_cam = (pts - t[None, :]) @ R               # (N,3)  [R is cam->ego, so R.T is ego->cam]
        # Note: pts_cam = R.T @ (pt - t) = R^T (pt - t)
        # Using matmul: (pt - t) @ R is (pt-t) @ R which is R^T applied row-wise = correct

        depth = pts_cam[:, 0]                           # x-forward
        valid = depth > 0.5
        pts_cam, depth = pts_cam[valid], depth[valid]

        u_f = K[0, 0] * pts_cam[:, 1] / depth + K[0, 2]
        v_f = K[1, 1] * (-pts_cam[:, 2]) / depth + K[1, 2]

        # downsample to feature grid
        fu = np.floor(u_f / downsample).astype(int)
        fv = np.floor(v_f / downsample).astype(int)
        in_fov = (fu >= 0) & (fu < fW) & (fv >= 0) & (fv < fH)
        in_rng = (depth >= d_min) & (depth <= d_max)
        keep   = in_fov & in_rng
        if not keep.any():
            continue

        # several returns can fall in one feature cell: keep the NEAREST
        # (the visible surface), vectorised instead of a per-point loop
        pix  = fv[keep] * fW + fu[keep]
        dmin = np.full(fH * fW, np.inf)
        np.minimum.at(dmin, pix, depth[keep])
        hit  = np.isfinite(dmin)
        bins = np.clip(np.searchsorted(d_edges, dmin[hit]) - 1, 0, depth_K - 1)
        result[cam].reshape(-1)[hit] = bins

    return result


def _load_pcd(path: str) -> np.ndarray:
    """
    Pure-numpy PCD reader (ASCII or binary). Returns the (N, 3) float32 xyz points.

    Deliberately does NOT use open3d: importing open3d inside a forked
    DataLoader worker deadlocks (all workers hang in `import open3d`), which
    stalled the first LiDAR-depth training run. Verified point-for-point
    against open3d on real OPV2V files.
    """
    with open(path, "rb") as f:
        raw = f.read()
    hdr_end = raw.find(b"DATA")
    hdr_end = raw.find(b"\n", hdr_end) + 1
    header = {}
    for line in raw[:hdr_end].decode("ascii", errors="ignore").splitlines():
        parts = line.split()
        if parts and not parts[0].startswith("#"):
            header[parts[0]] = parts[1:]
    n_pts = int(header["POINTS"][0])
    fields = header["FIELDS"]
    if n_pts == 0:
        return np.zeros((0, 3), dtype=np.float32)
    kind = header["DATA"][0]
    body = raw[hdr_end:]
    if kind == "ascii":
        arr = np.fromstring(body.decode("ascii", errors="ignore"), dtype=np.float64, sep=" ")
        arr = arr[: n_pts * len(fields)].reshape(n_pts, len(fields))
        cols = [fields.index(c) for c in ("x", "y", "z")]
        return arr[:, cols].astype(np.float32)
    if kind == "binary":
        tmap = {("F", 4): "f4", ("F", 8): "f8", ("U", 1): "u1", ("U", 2): "u2", ("U", 4): "u4",
                ("I", 1): "i1", ("I", 2): "i2", ("I", 4): "i4"}
        dt = np.dtype([(n, tmap[(t, int(sz))]) for n, sz, t in zip(fields, header["SIZE"], header["TYPE"])])
        rec = np.frombuffer(body, dtype=dt, count=n_pts)
        return np.stack([rec["x"], rec["y"], rec["z"]], axis=1).astype(np.float32)
    raise ValueError(f"unsupported PCD DATA type {kind!r} in {path}")


# ------------------------------------------------------------------ #
# Frame index builder                                                  #
# ------------------------------------------------------------------ #

def _build_frame_index(split_dir: str) -> List[Tuple[str, str, List[str]]]:
    """
    Enumerate all (scenario, timestamp, [cav_dirs]) tuples in the split.
    Returns list of (scenario_path, timestamp_stem, [cav_dir_paths]).
    Each element is one cooperative frame (one step in the simulation).
    """
    frames = []
    split_path = Path(split_dir)
    if not split_path.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    for scenario_dir in sorted(split_path.iterdir()):
        if not scenario_dir.is_dir():
            continue
        if scenario_dir.name == "data_protocol.yaml":
            continue

        # gather all CAV subdirs (named by integer CAV id)
        cav_dirs = sorted(
            [d for d in scenario_dir.iterdir()
             if d.is_dir() and d.name.lstrip("-").isdigit()],
            key=lambda d: int(d.name)
        )
        if not cav_dirs:
            continue

        # timestamps are defined by the YAML files in the first (ego) CAV dir
        ego_dir = cav_dirs[0]
        stamps = sorted(
            {p.stem for p in ego_dir.glob("*.yaml")
             if not p.name.startswith("data_protocol")}
        )
        for ts in stamps:
            # keep only frames where ALL CAVs have the same timestamp file
            valid_cavs = [d for d in cav_dirs if (d / f"{ts}.yaml").exists()]
            if len(valid_cavs) >= 2:             # need at least ego + 1 collaborator
                frames.append((str(scenario_dir), ts, [str(d) for d in valid_cavs]))

    return frames


# ------------------------------------------------------------------ #
# Dataset class                                                        #
# ------------------------------------------------------------------ #

class OPV2VDataset(Dataset):
    """
    Real OPV2V camera dataset for Kelvin2.

    Parameters
    ----------
    split_dir : str
        Path to the split directory, e.g.
        /mnt/scratch2/users/2641309/datasets/OPV2V/DATA_train/train_chunks/train
    cfg       : Config
        The project Config dataclass (src/config.py).
    max_agents: int
        Cap on number of agents per frame (ego + collaborators).
        OPV2V has 2–7; cap at 5 for memory.
    use_gt_pose: bool
        If True use true_ego_pos (ideal); if False use predicted_ego_pos (GPS noisy).
    """

    # OPV2V image resolution (fixed in the dataset)
    IMG_H = 384
    IMG_W = 512

    def __init__(
        self,
        split_dir: str,
        cfg,
        max_agents: int = 5,
        use_gt_pose: bool = True,
    ):
        self.split_dir  = split_dir
        self.cfg        = cfg
        self.max_agents = max_agents
        self.use_gt_pose = use_gt_pose

        print(f"[OPV2VDataset] scanning {split_dir} ...")
        self.frames = _build_frame_index(split_dir)
        if getattr(cfg, "depth_source", "zoedepth") == "lidar":
            _before = len(self.frames)
            self.frames = [
                f for f in self.frames
                if os.path.exists(os.path.join(f[2][0], f"{f[1]}.pcd"))
            ]
            print(f"[OPV2VDataset] depth_source=lidar: filtered "
                  f"{_before} -> {len(self.frames)} frames with .pcd present")
        print(f"[OPV2VDataset] {len(self.frames)} cooperative frames found.")

        # Image normalisation (ImageNet stats, matching CoBEVT / RayFusion)
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.augment = getattr(cfg, "augment", True)
        self._rng   = np.random.default_rng(getattr(cfg, "seed", 0))

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> dict:
        scenario_dir, ts, cav_dirs = self.frames[idx]

        # Cap agents
        cav_dirs = cav_dirs[: self.max_agents]
        N = len(cav_dirs)

        # ----------------------------------------------------------
        # 1) Load all CAV YAML files
        # ----------------------------------------------------------
        yamls = []
        for cav_dir in cav_dirs:
            yaml_path = os.path.join(cav_dir, f"{ts}.yaml")
            with open(yaml_path, "r") as f:
                yamls.append(yaml.load(f, Loader=yaml.Loader))

        # ----------------------------------------------------------
        # 2) Build ego world transform from agent 0 (LiDAR pose)
        # ----------------------------------------------------------
        ego_yaml = yamls[0]
        pose_key = "true_ego_pos" if self.use_gt_pose else "predicted_ego_pos"
        # LiDAR and ego pose may differ slightly; use lidar_pose for geometry
        ego_lidar_pose = ego_yaml.get("lidar_pose", ego_yaml[pose_key])
        ego_T  = _lidar_pose_to_T(ego_lidar_pose)      # 4×4 world->ego-LiDAR
        ego_T_world = ego_T                             # T_world_ego (world frame of ego lidar)
        ego_T_world_inv = np.linalg.inv(ego_T_world)   # ego-LiDAR -> world

        # ----------------------------------------------------------
        # 3) Per-agent: cameras, images, relative poses
        # ----------------------------------------------------------
        cam_names = ["camera0", "camera1", "camera2", "camera3"]
        n_cams    = len(cam_names)

        all_images  = np.zeros((N, n_cams, 3, self.IMG_H, self.IMG_W), dtype=np.float32)
        all_intrins = np.zeros((N, n_cams, 3, 3), dtype=np.float32)
        all_rots    = np.zeros((N, n_cams, 3, 3), dtype=np.float32)
        all_trans   = np.zeros((N, n_cams, 3),    dtype=np.float32)
        all_rel     = np.zeros((N, 3),             dtype=np.float32)   # [dx,dy,dyaw]

        depth_gt_imgs = -np.ones((N, n_cams,
                                   self.IMG_H // self.cfg.lss_downsample,
                                   self.IMG_W // self.cfg.lss_downsample),
                                  dtype=np.int64)

        ego_rots  = None   # store for OOF check
        ego_trans = None
        ego_intrins = None

        for i, (cav_dir, y) in enumerate(zip(cav_dirs, yamls)):
            # relative pose in ego LiDAR frame
            cav_pose = y.get("lidar_pose", y.get(pose_key, ego_lidar_pose))
            cav_T  = _lidar_pose_to_T(cav_pose)
            all_rel[i] = _relative_pose(ego_T_world, cav_T)

            for cam_idx, cam_name in enumerate(cam_names):
                cam_data = y.get(cam_name, {})
                if not cam_data:
                    # missing camera: fill identity
                    all_intrins[i, cam_idx] = np.eye(3, dtype=np.float32)
                    all_rots[i, cam_idx]    = np.eye(3, dtype=np.float32)
                    continue

                # intrinsic (3×3) -- rescale from native 600x800 to actual (IMG_H, IMG_W)
                K = np.array(cam_data["intrinsic"], dtype=np.float32)
                scale_x = self.IMG_W / 800.0
                scale_y = self.IMG_H / 600.0
                K[0, 0] *= scale_x   # fx
                K[0, 2] *= scale_x   # cx
                K[1, 1] *= scale_y   # fy
                K[1, 2] *= scale_y   # cy
                all_intrins[i, cam_idx] = K

                # extrinsic is cam->LiDAR  (4×4)
                ext4 = np.array(cam_data["extrinsic"], dtype=np.float32)
                rots_ij, trans_ij = _cam_extr_to_rot_trans(ext4, None)
                all_rots[i, cam_idx]  = rots_ij    # cam->LiDAR (= cam->ego)
                all_trans[i, cam_idx] = trans_ij

                # load RGB image
                img_path = os.path.join(cav_dir, f"{ts}_{cam_name}.png")
                img = self._load_image(img_path)
                all_images[i, cam_idx] = img        # (3, H, W) float32 normalised

            if i == 0:
                ego_rots   = all_rots[0]
                ego_trans  = all_trans[0]
                ego_intrins = all_intrins[0]

            # depth GT: either precomputed ZoeDepth monocular priors
            # (camera-only) or real LiDAR (oracle experiment), per cfg.depth_source
            if self.cfg.depth_source == "lidar":
                _pcd_path = os.path.join(cav_dir, f"{ts}.pcd")
                if os.path.exists(_pcd_path):
                    depth_gt_imgs[i] = _build_depth_gt(
                        _pcd_path,
                        all_rots[i], all_trans[i], all_intrins[i],
                        self.IMG_H, self.IMG_W,
                        self.cfg.lss_downsample,
                        self.cfg.lss_d_min, self.cfg.lss_d_max,
                        self.cfg.depth_K,
                    )
            elif self.cfg.depth_source == "zoedepth":
                _depth_planes = []
                for _ci, _cn in enumerate(cam_names):
                    _npy = os.path.join(cav_dir,
                        f"{ts}_{_cn}_depth_prior.npy")
                    if os.path.exists(_npy):
                        _d = np.load(_npy).astype(np.int64)
                        _fH = self.IMG_H // self.cfg.lss_downsample
                        _fW = self.IMG_W // self.cfg.lss_downsample
                        _ri = np.round(np.linspace(0, _d.shape[0]-1, _fH)).astype(int)
                        _ci = np.round(np.linspace(0, _d.shape[1]-1, _fW)).astype(int)
                        _depth_planes.append(_d[np.ix_(_ri, _ci)])
                    else:
                        fH = self.IMG_H // self.cfg.lss_downsample
                        fW = self.IMG_W // self.cfg.lss_downsample
                        _depth_planes.append(
                            -np.ones((fH, fW), dtype=np.int64))
                if any(p.max() >= 0 for p in _depth_planes):
                    depth_gt_imgs[i] = np.stack(_depth_planes, axis=0)

        # ----------------------------------------------------------
        # 4) BEV GT from ego-frame vehicle annotations
        # ----------------------------------------------------------
        vehicles = ego_yaml.get("vehicles", {})
        gt_dict  = _build_bev_gt(vehicles, np.linalg.inv(ego_T_world), self.cfg, ego_T_world)

        # BEV rotation augmentation (train only)
        # Breaks the static ghost-detection shortcut by randomising ego-relative
        # object positions so the model cannot memorise fixed BEV locations.
        if self.augment and getattr(self.cfg, "augment", True):
            from src.data.bev_augment import bev_rotate_scene
            all_rots, all_trans, all_rel, rot_boxes = bev_rotate_scene(
                all_rots, all_trans, all_rel,
                gt_dict["boxes"].copy(),
                max_angle_deg=180.0,
                rng=self._rng,
            )
            gt_dict = _build_bev_gt_from_boxes(rot_boxes, self.cfg)


        # OOF flags (objects outside all ego cameras)
        boxes = gt_dict["boxes"]
        if len(boxes) > 0 and ego_rots is not None:
            oof_flags = _check_oof(
                boxes, ego_rots, ego_trans, ego_intrins,
                self.IMG_H, self.IMG_W,
            )
        else:
            oof_flags = np.zeros(len(boxes), dtype=bool)
        gt_dict["oof"] = oof_flags

        # ----------------------------------------------------------
        # 5) Convert to tensors and return
        # ----------------------------------------------------------
        def t(x): return torch.from_numpy(x)

        scene = {
            "images":       t(all_images),
            "intrins":      t(all_intrins),
            "rots":         t(all_rots),
            "trans":        t(all_trans),
            "post_rots":    None,
            "post_trans":   None,
            "rel_pose":     t(all_rel),
            "depth_gt_img": t(depth_gt_imgs),
            "gt": {
                "heat":   t(gt_dict["heat"]),
                "offset": t(gt_dict["offset"]),
                "size":   t(gt_dict["size"]),
                "yaw":    t(gt_dict["yaw"]),
                "zh":     t(gt_dict["zh"]),
                "mask":   t(gt_dict["mask"]),
                "boxes":  t(gt_dict["boxes"]) if len(gt_dict["boxes"]) > 0
                          else torch.zeros((0, 7), dtype=torch.float32),
                "oof":    torch.from_numpy(oof_flags),
            },
            "n_agents": N,
        }
        return scene

    # ----------------------------------------------------------
    # helpers
    # ----------------------------------------------------------
    def _augment_image(self, img: np.ndarray) -> np.ndarray:
        """Apply colour jitter to a uint8 HxWx3 RGB image (training only)."""
        # brightness
        factor = random.uniform(0.6, 1.4)
        img = np.clip(img.astype(np.float32) * factor, 0, 255)
        # contrast
        factor = random.uniform(0.6, 1.4)
        mean = img.mean()
        img = np.clip((img - mean) * factor + mean, 0, 255)
        # saturation
        factor = random.uniform(0.6, 1.4)
        gray = img.mean(axis=2, keepdims=True)
        img = np.clip(gray + (img - gray) * factor, 0, 255)
        # hue shift (small)
        img = img.astype(np.uint8)
        import cv2 as _cv2
        hsv = _cv2.cvtColor(img, _cv2.COLOR_RGB2HSV).astype(np.int32)
        hsv[:, :, 0] = (hsv[:, :, 0] + random.randint(-10, 10)) % 180
        img = _cv2.cvtColor(hsv.astype(np.uint8), _cv2.COLOR_HSV2RGB)
        return img

    def _load_image(self, path: str) -> np.ndarray:
        """
        Load an RGB image as a (3, H, W) float32 numpy array,
        ImageNet-normalised, resized to (IMG_H, IMG_W) if needed.
        """
        if not os.path.exists(path):
            return np.zeros((3, self.IMG_H, self.IMG_W), dtype=np.float32)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            return np.zeros((3, self.IMG_H, self.IMG_W), dtype=np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[:2] != (self.IMG_H, self.IMG_W):
            img = cv2.resize(img, (self.IMG_W, self.IMG_H), interpolation=cv2.INTER_LINEAR)
        if self.augment:
            img = self._augment_image(img)
        img = img.astype(np.float32) / 255.0
        img = (img - self._mean[None, None, :]) / self._std[None, None, :]
        return img.transpose(2, 0, 1)          # (3, H, W)


# ------------------------------------------------------------------ #
# DataLoader collation                                                 #
# ------------------------------------------------------------------ #

def opv2v_collate(batch):
    """
    Agent count varies per scene; we do not stack across the batch.
    For batch_size=1 (standard for this project) just unwrap.
    For batch_size>1 return a list.
    """
    if len(batch) == 1:
        return batch[0]
    return batch


# ------------------------------------------------------------------ #
# Quick sanity check (run directly)                                   #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import sys
    from dataclasses import dataclass

    @dataclass
    class _Cfg:
        grid_H: int = 200; grid_W: int = 200; cell_m: float = 0.5
        bev_x_min: float = -50.0; bev_y_min: float = -50.0
        bev_x_max: float =  50.0; bev_y_max: float =  50.0
        lss_d_min: float = 4.0;   lss_d_max: float = 45.0
        depth_K:   int   = 64;    lss_downsample: int = 16

    split = (sys.argv[1] if len(sys.argv) > 1
             else "/mnt/scratch2/users/2641309/datasets/OPV2V/"
                  "DATA_validate/validate_chunks/validate")
    cfg = _Cfg()
    ds  = OPV2VDataset(split, cfg, max_agents=5)
    print(f"Dataset length: {len(ds)}")
    if len(ds) > 0:
        scene = ds[0]
        print("images  :", scene["images"].shape)
        print("rel_pose:", scene["rel_pose"])
        print("n_agents:", scene["n_agents"])
        print("GT boxes:", scene["gt"]["boxes"].shape)
        print("OOF     :", scene["gt"]["oof"].sum().item(), "out-of-FoV objects")
        print("depth_gt: range", scene["depth_gt_img"].min().item(),
              "to", scene["depth_gt_img"].max().item())
        print("PASS: OPV2V loader is working on real data.")
