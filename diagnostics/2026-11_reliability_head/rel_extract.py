"""Cache per-pixel data for the reliability-head test, from a FROZEN backbone.

For each LiDAR-labelled image pixel, stores the encoder's image feature vector plus the quantities
the baselines need. Only labelled pixels are kept, so the cache stays small.

Reliability must be predicted from IMAGE features, not BEV geometry: depth is biased near, so far
content is splatted into near BEV cells and a BEV-side predictor would see the wrong neighbourhood.

    rel_extract.py <ckpt> <split> <shard> <n_shards> <n_frames_total> <out.npz>
"""
import sys, os, time
sys.path.insert(0, "/mnt/scratch2/users/2641309/projects/coop_v2v")
from dataclasses import replace
import numpy as np, torch, torch.nn.functional as F
from src.config import Config
from src.data.opv2v import OPV2VDataset, opv2v_collate
from src.models.lss_backbone import LSSBackbone
from src.metrics import _to_device
from src.depth_bins import depth_bin_edges

ckpt, split, shard, n_shards, n_total, out = (
    sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6])
torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))

cfg = replace(Config(), device="cpu", augment=False, depth_source="lidar")
backbone = LSSBackbone(cfg)
sd = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
backbone.load_state_dict({k[9:]: v for k, v in sd.items() if k.startswith("backbone.")})
backbone.eval()
for p in backbone.parameters():
    p.requires_grad_(False)

edges = depth_bin_edges(cfg.lss_d_min, cfg.lss_d_max, cfg.depth_K, getattr(cfg, "depth_bins", "linear"))
ctr = torch.tensor((edges[:-1] + edges[1:]) / 2, dtype=torch.float32)
pts = torch.linspace(cfg.lss_d_min, cfg.lss_d_max, cfg.depth_K)

ds = OPV2VDataset(cfg.train_dir if split == "train" else cfg.val_dir, cfg,
                  max_agents=1, use_gt_pose=cfg.use_gt_pose)
idxs = np.linspace(0, len(ds) - 1, n_total).astype(int)[shard::n_shards]
print(f"{split} shard {shard}/{n_shards}: {len(idxs)} frames of {len(ds)}", flush=True)

FEAT, ENT, U, V, CAM, PRED, TRUE, FRAME = [], [], [], [], [], [], [], []
t0 = time.time()
with torch.no_grad():
    for n, i in enumerate(idxs):
        s = _to_device(opv2v_collate([ds[int(i)]]), torch.device("cpu"))
        lab = s["depth_gt_img"][0]                                  # (n_cams, fH, fW)
        m = lab >= 0
        if not m.any():
            continue
        # frozen encoder: depth logits + context features, exactly as the lift uses them
        depth_logits_img, context = backbone.enc(s["images"][0])    # (n_cams,K,fH,fW), (n_cams,C,fH,fW)
        p = F.softmax(depth_logits_img.float(), dim=1)
        ent = -(p * (p + 1e-9).log()).sum(1) / np.log(cfg.depth_K)  # (n_cams,fH,fW)
        pred = pts[p.argmax(1)]                                     # argmax point estimate
        true = ctr[lab.clamp(min=0)]
        nC, C, fH, fW = context.shape
        gu, gv = torch.meshgrid(torch.arange(fH), torch.arange(fW), indexing="ij")
        cam_id = torch.arange(nC).view(nC, 1, 1).expand(nC, fH, fW)
        feat = context.permute(0, 2, 3, 1)[m]                       # (n_px, C)
        FEAT.append(feat.numpy().astype(np.float16))                # fp16: cache size, not precision-critical
        ENT.append(ent[m].numpy()); PRED.append(pred[m].numpy()); TRUE.append(true[m].numpy())
        U.append(gv.expand(nC, fH, fW)[m].numpy().astype(np.int16))
        V.append(gu.expand(nC, fH, fW)[m].numpy().astype(np.int16))
        CAM.append(cam_id[m].numpy().astype(np.int8))
        FRAME.append(np.full(int(m.sum()), int(i), dtype=np.int32))
        if (n + 1) % 10 == 0:
            print(f"  {n+1}/{len(idxs)}  {(time.time()-t0)/(n+1):.1f}s/frame", flush=True)

np.savez_compressed(out,
                    feat=np.concatenate(FEAT), ent=np.concatenate(ENT), u=np.concatenate(U),
                    v=np.concatenate(V), cam=np.concatenate(CAM), pred=np.concatenate(PRED),
                    true=np.concatenate(TRUE), frame=np.concatenate(FRAME))
print(f"saved {out}: {sum(len(x) for x in ENT)} pixels, {(time.time()-t0)/60:.1f} min")
