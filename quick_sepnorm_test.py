"""
quick_sepnorm_test.py -- arm 3: separate pos/neg normalization.

Does NOT edit src/train.py or src/loss/coop_loss.py on disk (both are
shared with arms 1 and 2). Instead, monkey-patches the `coop_loss` name
inside the already-imported src.train module's namespace, since train()
looks up `coop_loss` as a module-level global at call time.
"""
from dataclasses import replace
from src.config import Config
from src.baselines import make
import src.train as train_module
from src.train import build_loaders
from src.loss.coop_loss_sepnorm import coop_loss as coop_loss_sepnorm

# monkey-patch: src.train.train() calls coop_loss(...) by looking it up
# in its own module's globals at call time -- this redirects it.
train_module.coop_loss = coop_loss_sepnorm

def main():
    base = Config()
    base.validate()
    print("\n========== Building OPV2V data loaders ==========")
    tr, va, te = build_loaders(base)
    print(f"  Train: {len(tr.dataset)}  Val: {len(va.dataset)}  Test: {len(te.dataset)}")

    cfg = make("ours_full", base)
    cfg = replace(cfg, exp_name="ours_full_sepnorm", epochs=10)

    print(f"\n{'='*60}")
    print(f"  Training: {cfg.exp_name}  (separate pos/neg normalization, epochs={cfg.epochs})")
    print(f"{'='*60}")
    train_module.train(cfg, tr, va, te)

if __name__ == "__main__":
    main()
