from functools import partial

import numpy as np
import torch
from tqdm import tqdm
from copy import deepcopy

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
        lambda_mt,
        ema_decay,
        mt_augment_scale,
        grad_clip_norm: float = 0.0,
        use_target_normalization: bool = True,
    ):
        # Semi-supervised hyperparameters (configurable)
        self.device = device
        self.models = models
        for m in self.models:
            m.to(device)
            for p in m.parameters():
                p.requires_grad = True

        self.teacher_models = deepcopy(models)
        for teacher in self.teacher_models:
            teacher.to(device)
            for p in teacher.parameters():
                p.requires_grad = False
            # keep teacher in eval mode so it provides a stable target (no dropout/bn updates)
            teacher.eval()

        for m in self.models:
            m.to(device)
            for p in m.parameters():
                p.requires_grad = True

                
        # set from init args so they can be configured via Hydra
        self.lambda_mt = float(lambda_mt)
        self.lambda_cps = 1.0
        self.ema_decay = float(ema_decay)
        self.mt_augment_scale = float(mt_augment_scale)
        self.grad_clip_norm = float(grad_clip_norm)
        self.use_target_normalization = bool(use_target_normalization)

        # Optim related things
        self.supervised_criterion = supervised_criterion
        all_params = [p for m in self.models for p in m.parameters()]
        self.optimizer = optimizer(params=all_params)
        self.scheduler = scheduler(optimizer=self.optimizer)

        # Dataloader setup
        self.train_dataloader = datamodule.train_dataloader()
        self.val_dataloader = datamodule.val_dataloader()
        self.test_dataloader = datamodule.test_dataloader()
        self.unlabeled_train_dataloader = datamodule.unsupervised_train_dataloader()

        # Logging
        self.logger = logger
        # place to store best model weights found during training
        self._best_state = None
        self._best_epoch = None


    # ---------------------------
    # Mean Teacher EMA update
    # ---------------------------
    def update_teacher(self):
        for teacher, student in zip(self.teacher_models, self.models):
            for tp, sp in zip(teacher.parameters(), student.parameters()):
                tp.data = self.ema_decay * tp.data + (1.0 - self.ema_decay) * sp.data

    # ---------------------------
    # N-CPS consistency
    # ---------------------------
    def noisy_augment(self, data):
        # simple example: gaussian noise using configurable scale
        noisy_x = data.x + float(self.mt_augment_scale) * torch.randn_like(data.x)
        data_aug = deepcopy(data)
        data_aug.x = noisy_x
        return data_aug

    def validate(self):
        for model in self.models:
            model.eval()

        val_losses = []
        
        with torch.no_grad():
            for x, targets in self.val_dataloader:
                x, targets = x.to(self.device), targets.to(self.device)
                
                # Ensemble prediction
                preds = [model(x) for model in self.models]
                # If using target normalization, model outputs are in normalized space;
                # un-normalize before computing validation MSE so it's in original scale.
                if getattr(self, 'use_target_normalization', False):
                    preds = [(p * self.target_std + self.target_mean) for p in preds]
                avg_preds = torch.stack(preds).mean(0)

                val_loss = torch.nn.functional.mse_loss(avg_preds, targets)
                val_losses.append(val_loss.item())
        val_loss = np.mean(val_losses)
        return {"val_MSE": val_loss}

    def test(self):
        """Evaluate models on the test set and return test metrics.

        Returns a dict like {"test_MSE": float}.
        """
        for model in self.models:
            model.eval()

        test_losses = []
        with torch.no_grad():
            for x, targets in self.test_dataloader:
                x, targets = x.to(self.device), targets.to(self.device)

                preds = [model(x) for model in self.models]
                # If using target normalization, un-normalize predictions
                if getattr(self, 'use_target_normalization', False):
                    preds = [(p * self.target_std + self.target_mean) for p in preds]

                avg_preds = torch.stack(preds).mean(0)
                test_loss = torch.nn.functional.mse_loss(avg_preds, targets)
                test_losses.append(test_loss.item())

        test_loss = float(np.mean(test_losses)) if len(test_losses) > 0 else float('nan')
        # log and return
        try:
            self.logger.log_dict({"test_MSE": test_loss})
        except Exception:
            pass
        return {"test_MSE": test_loss}

    def train(self, total_epochs, validation_interval, early_stopping_patience=None, **kwargs):
        final_results = {}
        patience_counter = 0
        # allow overriding patience from config; default to 10 if not provided
        patience = int(early_stopping_patience) if early_stopping_patience is not None else 10
        best_val_loss = float('inf')

        # If target normalization is enabled, compute train target mean/std once
        self.target_mean = 0.0
        self.target_std = 1.0
        if self.use_target_normalization:
            all_targets = []
            for _, t in self.train_dataloader:
                # t may be (batch,1) or (batch,)
                if isinstance(t, (list, tuple)):
                    t = t[0]
                all_targets.append(t.detach().cpu())
            if len(all_targets) > 0:
                all_targets = torch.cat(all_targets, dim=0)
                self.target_mean = float(all_targets.mean())
                self.target_std = float(all_targets.std())
                if self.target_std == 0:
                    self.target_std = 1.0

        unlabeled_iter = iter(self.unlabeled_train_dataloader)

        for epoch in (pbar := tqdm(range(1, total_epochs + 1))):
            for m in self.models:
                m.train()

            supervised_log = []
            mt_log = []

            for x_labeled, targets in self.train_dataloader:
                # Get unlabeled batch
                try:
                    x_unl = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(self.unlabeled_train_dataloader)
                    x_unl = next(unlabeled_iter)

                x_labeled, targets = x_labeled.to(self.device), targets.to(self.device)
                x_unl = x_unl[0].to(self.device)

                # create an augmented view for the student so MT loss is meaningful
                x_unl_student = self.noisy_augment(x_unl)

                self.optimizer.zero_grad()

                # -------------------------
                # 1. Supervised loss (optionally using target normalization)
                # -------------------------
                preds = [m(x_labeled) for m in self.models]

                if self.use_target_normalization:
                    targets_norm = (targets - self.target_mean) / self.target_std
                else:
                    targets_norm = targets

                # loss used for backward (on normalized targets when enabled)
                sup_losses = [self.supervised_criterion(p, targets_norm) for p in preds]
                sup_loss = sum(sup_losses) / len(self.models)

                # For logging, compute un-normalized MSE between ensemble preds and raw targets
                try:
                    with torch.no_grad():
                        preds_un = [(p * self.target_std + self.target_mean) if self.use_target_normalization else p for p in preds]
                        ensemble_un = torch.stack(preds_un).mean(0)
                        supervised_log.append(torch.nn.functional.mse_loss(ensemble_un, targets).item())
                except Exception:
                    supervised_log.append(sup_loss.item())


                # -------------------------
                # 2. Mean Teacher loss
                # -------------------------
                # student sees augmented view, teacher sees original (stable) view
                student_out = [m(x_unl_student) for m in self.models]
                teacher_out = [tm(x_unl).detach() for tm in self.teacher_models]


                mt_loss = 0
                for s, t in zip(student_out, teacher_out):
                    mt_loss += torch.nn.functional.mse_loss(s, t)
                mt_loss = mt_loss / len(self.models)
                mt_log.append(mt_loss.item())


                # -------------------------
                # Total loss
                # -------------------------
                loss = sup_loss + self.lambda_mt * mt_loss
                loss.backward()

                # gradient clipping (if enabled)
                if self.grad_clip_norm and self.grad_clip_norm > 0.0:
                    params = [p for m in self.models for p in m.parameters() if p.grad is not None]
                    torch.nn.utils.clip_grad_norm_(params, self.grad_clip_norm)

                self.optimizer.step()

                # Update EMA teacher
                self.update_teacher()

            self.scheduler.step()

            summary_dict = {
                "supervised_loss": np.mean(supervised_log),
                "mean_teacher_loss": np.mean(mt_log),
            }

            if epoch % validation_interval == 0 or epoch == total_epochs:
                val_metrics = self.validate()
                summary_dict.update(val_metrics)
                pbar.set_postfix(summary_dict)

                # Early stopping
                cur_val = val_metrics["val_MSE"]
                if cur_val < best_val_loss:
                    best_val_loss = cur_val
                    patience_counter = 0
                    # save best model weights (deepcopy state_dicts)
                    try:
                        self._best_state = [ {k: v.cpu().clone() for k, v in m.state_dict().items()} for m in self.models ]
                        self._best_epoch = epoch
                    except Exception:
                        self._best_state = None
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch}")
                        break

            self.logger.log_dict(summary_dict, step=epoch)
            final_results = summary_dict

        # If we saved a best checkpoint during training, restore it so subsequent
        # testing uses the best-validation weights rather than the final weights.
        if self._best_state is not None:
            try:
                for m, state in zip(self.models, self._best_state):
                    m.load_state_dict(state)
                print(f"Restored best model from epoch {self._best_epoch} for testing/evaluation.")
            except Exception:
                print("Failed to restore best model state; using final weights.")

        return final_results
