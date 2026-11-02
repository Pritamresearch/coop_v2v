import sys; sys.path.insert(0, "/mnt/scratch2/users/2641309/projects/coop_v2v")
import numpy as np, torch, torch.nn.functional as F
import src.metrics as M
from src.config import Config; from src.baselines import make
cfg = make(sys.argv[2], Config()); C = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
fr = [f for f in C["frames"] if f["split"] == "val"]
orig_peaks = M._peaks
def blur(h, s):
    if s == 0: return h
    r = int(3*s); x = torch.arange(-r, r+1).float(); k = torch.exp(-x**2/(2*s*s)); k /= k.sum()
    h4 = h[None, None]; h4 = F.conv2d(F.pad(h4, (r, r, 0, 0), mode="replicate"), k.view(1,1,1,-1))
    return F.conv2d(F.pad(h4, (0, 0, r, r), mode="replicate"), k.view(1,1,-1,1))[0, 0]
print(f"val frames={len(fr)}   (baseline = blur 0, kernel 3, nms 2.0)")
for s in [0, 1, 2, 3]:
    for kern in [3, 7]:
        for nms in [2.0, 4.0]:
            M._peaks = lambda h, kernel=3, _k=kern: orig_peaks(h, _k)
            cfg.nms_radius_m = nms; ev = M.Evaluator(cfg); npk = 0
            for f in fr:
                o = dict(f["out"]); o["heat"] = blur(o["heat"], s)
                b, sc = M.decode(o, cfg, score_thresh=0.2, topk=200); npk += len(sc)
                ev.update(b, sc, f["gt"]["boxes"].numpy(), f["gt"]["oof"].numpy(), kb=0.0)
            r = ev.compute()
            print(f"  blur_sigma={s} peak_kernel={kern} nms={nms}: peaks/fr={npk/len(fr):5.1f} AP30={r['ap30']:.4f} AP50={r['ap50']:.4f} rec50={r['recall50']:.3f}")
