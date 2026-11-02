from dataclasses import replace
from src.config import Config
from src.baselines import make
from src.train import train, build_loaders

def main():
    base = Config()
    base.validate()
    tr, va, te = build_loaders(base)
    cfg = make("ours_full", base)
    cfg = replace(cfg, exp_name="ours_full_augmented", epochs=15)
    print(f"Training: {cfg.exp_name}  augment={cfg.augment}  epochs={cfg.epochs}")
    train(cfg, tr, va, te)

if __name__ == "__main__":
    main()
