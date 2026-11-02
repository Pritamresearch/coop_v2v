"""Run last.pth forward on real frames (val + a few train w/o aug), cache outputs to .pt."""
import sys, time
from dataclasses import replace
sys.path.insert(0, "/mnt/scratch2/users/2641309/projects/coop_v2v")
import torch
from src.config import Config
from src.baselines import make
from src.data.opv2v import OPV2VDataset, opv2v_collate
from src.models.coop_detector import CoopDetector
from src.metrics import _to_device

ckpt, preset, outp, n_val, n_tr = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
torch.set_num_threads(8)
base = Config(); base.validate()
cfg = make(preset, base)
ck = torch.load(ckpt, map_location="cpu")
model = CoopDetector(cfg); model.load_state_dict(ck["model"]); model.eval()
print("epoch", ck.get("epoch"), flush=True)
cfg_na = replace(cfg, augment=False)
sets = {"val": OPV2VDataset(cfg.val_dir, cfg_na, max_agents=cfg.max_agents, use_gt_pose=cfg.use_gt_pose),
        "train": OPV2VDataset(cfg.train_dir, cfg_na, max_agents=cfg.max_agents, use_gt_pose=cfg.use_gt_pose)}
res = []
with torch.no_grad():
    for split, n in [("val", n_val), ("train", n_tr)]:
        ds = sets[split]; ds.augment = False
        idxs = torch.linspace(0, len(ds) - 1, n).long().tolist()   # spread across scenarios
        for i in idxs:
            t = time.time()
            scene = _to_device(opv2v_collate([ds[i]]), torch.device("cpu"))
            out = model(scene, hard=True, gumbel_T=cfg.gumbel_T1, tau=cfg.tau, budget_frac=cfg.budget_frac)
            keep = {k: out[k].float().cpu() for k in ("heat", "offset", "size", "yaw", "zh") if k in out}
            keep["n_cells"] = int(out.get("n_cells", 0))
            gt = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in scene["gt"].items()}
            res.append({"split": split, "idx": i, "out": keep, "gt": gt})
            print(split, i, f"{time.time()-t:.1f}s", "ngt", int(gt["mask"].sum()), flush=True)
torch.save({"epoch": ck.get("epoch"), "frames": res}, outp)
print("saved", outp)
