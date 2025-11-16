from functools import partial

import numpy as np
import torch
from tqdm import tqdm

class SemiSupervisedEnsemble:
    def __init__(
        self,
        supervised_criterion,
        optimizer,
        scheduler,
        device,
        models,
        logger,
        datamodule,
    ):
        self.device = device
        self.models = models

        # Optim related things
        self.supervised_criterion = supervised_criterion
        all_params = [p for m in self.models for p in m.parameters()]
        self.optimizer = optimizer(params=all_params)
        self.scheduler = scheduler(optimizer=self.optimizer)

        # Dataloader setup
        self.train_dataloader = datamodule.train_dataloader()
        self.val_dataloader = datamodule.val_dataloader()
        self.test_dataloader = datamodule.test_dataloader()

        # Logging
        self.logger = logger

    def validate(self):
        for model in self.models:
            model.eval()

        val_losses = []
        
        with torch.no_grad():
            for x, targets in self.val_dataloader:
                x, targets = x.to(self.device), targets.to(self.device)
                
                # Ensemble prediction
                preds = [model(x) for model in self.models]
                avg_preds = torch.stack(preds).mean(0)
                
                val_loss = torch.nn.functional.mse_loss(avg_preds, targets)
                val_losses.append(val_loss.item())
        val_loss = np.mean(val_losses)
        return {"val_MSE": val_loss}

    def train(self, total_epochs, validation_interval, early_stopping_patience=None, early_stopping_min_delta=0.0):
        """Train loop with optional early stopping.

        early_stopping_patience: number of validation checks with no improvement before stopping.
        early_stopping_min_delta: minimum decrease in validation metric to count as improvement.
        """
        best_val = float("inf")
        no_improve_count = 0
        last_summary = None

        for epoch in (pbar := tqdm(range(1, total_epochs + 1))):
            for model in self.models:
                model.train()
            supervised_losses_logged = []
            for x, targets in self.train_dataloader:
                x, targets = x.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                # Supervised loss
                supervised_losses = [self.supervised_criterion(model(x), targets) for model in self.models]
                supervised_loss = sum(supervised_losses)
                supervised_losses_logged.append(supervised_loss.detach().item() / len(self.models))  # type: ignore
                loss = supervised_loss
                loss.backward()  # type: ignore
                self.optimizer.step()
            self.scheduler.step()
            supervised_losses_logged = np.mean(supervised_losses_logged)

            summary_dict = {
                "supervised_loss": supervised_losses_logged,
            }

            do_validate = (epoch % validation_interval == 0) or (epoch == total_epochs)
            if do_validate:
                val_metrics = self.validate()
                summary_dict.update(val_metrics)
                pbar.set_postfix(summary_dict)

                # Early stopping logic uses validation metric named 'val_MSE'
                try:
                    val_metric = val_metrics.get("val_MSE")
                except Exception:
                    val_metric = None

                if val_metric is not None:
                    # improvement if val_metric decreased more than min_delta
                    if val_metric + early_stopping_min_delta < best_val:
                        best_val = val_metric
                        no_improve_count = 0
                    else:
                        no_improve_count += 1

                    if early_stopping_patience is not None and early_stopping_patience > 0:
                        if no_improve_count >= early_stopping_patience:
                            summary_dict["early_stopped"] = True
                            summary_dict["best_val"] = best_val
                            summary_dict["stopped_epoch"] = epoch
                            self.logger.log_dict(summary_dict, step=epoch)
                            last_summary = summary_dict
                            pbar.write(f"Early stopping triggered at epoch {epoch} (no_improve_count={no_improve_count}).")
                            break

            self.logger.log_dict(summary_dict, step=epoch)
            last_summary = summary_dict

        return last_summary