
from itertools import chain
import hydra
import torch
from omegaconf import OmegaConf
import numpy as np

from utils import seed_everything


@hydra.main(
    config_path="../configs/",
    config_name="run.yaml",
    version_base=None,
)
def main(cfg):
    # print out the full config
    print(OmegaConf.to_yaml(cfg))

    if cfg.device in ["unset", "auto"]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)

    seed_everything(cfg.seed, cfg.force_deterministic)

    logger = hydra.utils.instantiate(cfg.logger)
    hparams = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    logger.init_run(hparams)

    dm = hydra.utils.instantiate(cfg.dataset.init)

    model = hydra.utils.instantiate(cfg.model.init).to(device)

    if cfg.compile_model:
        model = torch.compile(model)
    models = [model]
    trainer = hydra.utils.instantiate(cfg.trainer.init, models=models, logger=logger, datamodule=dm, device=device)

    results = trainer.train(**cfg.trainer.train)
    # trainer.train may return None (it currently does). Only convert to a
    # tensor when results is present and convertible to avoid TypeError.
    if results is None:
        print("trainer.train() returned None — no results to convert")
    else:
        # Minimal conversion: if trainer returns a dict/list/scalar of numeric
        # values (numpy or Python), convert them into a 1-D tensor.
        try:
            if isinstance(results, dict):
                vals = [v.item() if hasattr(v, "item") else v for v in results.values()]
                results = torch.tensor(vals, dtype=torch.float32)
            elif isinstance(results, (list, tuple)):
                vals = [v.item() if hasattr(v, "item") else v for v in results]
                results = torch.tensor(vals, dtype=torch.float32)
            elif hasattr(results, "item"):
                results = torch.tensor(results.item())
            else:
                print("trainer.train() returned non-convertible value:", results)
        except Exception as e:
            print("Failed converting results to tensor:", e)

if __name__ == "__main__":
    main()
