import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import wandb
import pytorch_lightning as pl
from pytorch_lightning.cli import instantiate_class

from typing import List, Union


plt.switch_backend("agg")  # otherwise will get thread not running in main loop error
# plt.switch_backend("TkAgg")  # otherwise will get thread not running in main loop error

import losses as losses
from metrics import compute_metrics
from features import get_transform
from utils import (
    unit_to_hz,
    hz_to_unit,
)
from plot_log_utils import plot_and_log


class Trainer(pl.LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        loss_fn: nn.Module,
        regularizations: List[nn.Module] = [],
        pretrain_loss_fn: List[nn.Module] = [],
        pretrain_steps: int = 0,
        feature_extractor: Union[str, dict] = "cqt",  # Encoder input
        transform: Union[str, dict] = "identity",  # Where to compute loss
        optimizer_init: dict = {},
        scheduler: dict = None,
        evaluation_metrics: dict = {},
        freq_hz_min: Union[float, str] = 32.7,
        freq_hz_max: Union[float, str] = 8000,
        temperature: float = 1.0,
        sample_rate: int = 16000,
        log_kwargs: dict = {},
        detach_weights: bool = False,
        baseline_gt_pitch: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore="encoder, decoder, loss_fn")

        self.encoder = encoder
        self.decoder = decoder

        self.optimizer_init = optimizer_init
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.evaluation_metrics = evaluation_metrics
        self.sample_rate = sample_rate
        self.regularizations = regularizations

        self.pretrain_loss_fn = pretrain_loss_fn
        self.pretrain_steps = pretrain_steps

        self.baseline_gt_pitch = baseline_gt_pitch
        self.detach_weights = detach_weights
        self.temperature = temperature
        self.feature_extractor = get_transform(feature_extractor, sample_rate)
        self.transform = get_transform(transform, sample_rate)

        self.freq_hz_min = (
            self.feature_extractor.get_frequencies()[0] if freq_hz_min == "auto" else freq_hz_min
        )
        self.freq_hz_max = (
            self.feature_extractor.get_frequencies()[-1] if freq_hz_max == "auto" else freq_hz_max
        )

        # Only for plotting
        spec_transform = "stft"  # if transform == 'identity' else transform
        self.spec_transform = get_transform(spec_transform, sample_rate)  # Only for plotting
        self.log_kwargs = log_kwargs

    def encode(self, x, **kwargs):
        features = self.feature_extractor(x[:, :-1])
        batch, time, freq = features.shape
        n_frames = time

        # Encoder takes (batch, channels, freq), and here we treat each time individually
        features = features.reshape(batch * time, freq).unsqueeze(1)

        z = self.encoder(features)

        pitch_outputs = self.encoder.predict_pitch(z["frequency"], temperature=self.temperature)
        pitch_unit = pitch_outputs.pop("pitch_unit")
        pitch_hz = unit_to_hz(pitch_unit, self.freq_hz_min, self.freq_hz_max)

        # This mode serves as baseline for reconstruction error and weight/gain estimation
        if self.baseline_gt_pitch:
            pitch_hz = kwargs["gt_pitch_hz"]
            if pitch_hz.ndim == 1:
                pitch_hz = pitch_hz.unsqueeze(-1).repeat(1, n_frames).reshape(batch * n_frames, 1)
            pitch_unit = hz_to_unit(pitch_hz, self.freq_hz_min, self.freq_hz_max)

        weights = z.get("weights", None)
        if weights is None:
            weights = torch.ones_like(pitch_unit)
        if weights.ndim == 1:
            weights = weights.unsqueeze(-1)

        # TODO - Improve this later ------------------
        for key, value in z.items():
            # Unpack batch and time from first dimension and keep the rest
            z[key] = value.reshape(batch, time, *value.shape[1:])
        pitch_unit = pitch_unit.reshape(batch, time, *pitch_unit.shape[1:])
        pitch_hz = pitch_hz.reshape(batch, time, *pitch_hz.shape[1:])
        weights = weights.reshape(batch, time, *weights.shape[1:])

        z.update({"pitch": pitch_unit})
        z["weights"] = weights
        z["pitch_hz"] = pitch_hz
        z.update(pitch_outputs)

        if self.decoder.__class__.__name__ == "Sinusoidal":
            synth_params = (weights, pitch_hz)
        else:
            raise NotImplementedError

        return z, synth_params

    def encode_pitch(self, x):
        features = self.feature_extractor(x)
        batch, time, freq = features.shape
        features = features.reshape(batch * time, freq).unsqueeze(1)
        z = self.encoder(features)
        pitch_outputs = self.encoder.predict_pitch(z["frequency"], temperature=self.temperature)
        pitch_unit = pitch_outputs.pop("pitch_unit")
        pitch_unit = pitch_unit.reshape(batch, time, *pitch_unit.shape[1:])
        return pitch_unit

    def decode(self, *synth_params):
        x_hat = self.decoder(*synth_params)  # Synthesize audio
        return x_hat

    def forward(self, x, **kwargs):
        z, synth_params = self.encode(x, **kwargs)
        x_hat = self.decode(*synth_params)

        if self.detach_weights:  # Not used in the paper
            synth_params = list(synth_params)
            synth_params[0] = synth_params[0].detach()
            x_hat_weights_detached = self.decoder(*synth_params)
            z.update({"x_hat_weights_detached": x_hat_weights_detached})

        z.update({"x_hat": x_hat})
        return z

    def configure_optimizers(self):
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        optimizer = instantiate_class(params, self.optimizer_init)
        if self.scheduler is None:
            return optimizer
        scheduler = instantiate_class(args=optimizer, init=self.scheduler)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def shared_step(self, batch, batch_idx, step_name):
        x = batch["x"]
        frd_kwargs = {}
        frd_kwargs = {} if not self.baseline_gt_pitch else {"gt_pitch_hz": batch["frequency"]}

        # Forward pass
        output = self(x, **frd_kwargs)

        output.update({"x": batch["x"]})
        x_hat = output["x_hat"]
        true_pitch = batch.get("frequency", None)
        true_weights = batch.get("weights", torch.tensor([0]).to(x.device))

        pitch_hz = output["pitch_hz"]

        dim_diff = pitch_hz.ndim - true_pitch.ndim
        if dim_diff == 1:
            true_pitch_hz = true_pitch.unsqueeze(1).expand_as(pitch_hz)
        elif dim_diff == 2:
            true_pitch_hz = true_pitch.unsqueeze(1).unsqueeze(1).expand_as(pitch_hz)

        output.update(
            {
                "true_pitch_hz": true_pitch_hz,
                "true_pitch": true_pitch,
                "true_weights": true_weights,
                "true_frequency_unit": hz_to_unit(true_pitch, self.freq_hz_min, self.freq_hz_max),
                "frequency_unit": output["pitch"],
            }
        )

        # Map spectra to unit scale ([0,1])
        x_pos = None
        y_pos = None
        
        if hasattr(self.loss_fn, "log_scaled_x") and self.loss_fn.log_scaled_x:
            x_pos = torch.tensor(self.transform.get_frequencies()).to(x.device)
            # Hz to unit scales frequencies logarithmically to [0,1]
            x_pos = hz_to_unit(x_pos, self.freq_hz_min, self.freq_hz_max) 
            y_pos = x_pos.clone()
        else:
            if not isinstance(self.transform, nn.Identity):
                x_pos = torch.tensor(self.transform.get_frequencies()).to(x.device)
                # Scales frequencies linearly to [0,1]
                x_pos = x_pos / x_pos.max()
                y_pos = x_pos.clone()

        spec_x = self.transform(x)
        spec_x_hat = self.transform(x_hat)

        loss = 0
        # Pretrain losses
        if self.global_step < self.pretrain_steps:
            for loss_fn in self.pretrain_loss_fn:
                name = loss_fn.__class__.__name__
                if name == "ParameterLoss":
                    name = f"{name}_{loss_fn.parameter}"
                elif name == "MSSLoss":
                    _loss = loss_fn(x, x_hat)
                elif name == "Wasserstein1D":
                    _loss = loss_fn(spec_x, spec_x_hat, x_pos=x_pos, y_pos=y_pos)
                elif name == "Wasserstein1DWithTransform":
                    _loss = loss_fn(x, x_hat)
                elif name == "MixOfLosses":
                    _loss = loss_fn(spec_x, spec_x_hat, x_pos=x_pos, y_pos=y_pos)
                else:
                    _loss = loss_fn(output)
                loss += _loss

                self.log(f"loss/{step_name}/{name}_pretrain", _loss)

        compute_main_loss = True
        compute_main_loss = (self.global_step >= self.pretrain_steps) and (
            not isinstance(self.loss_fn, nn.Identity)
        )

        # Training losses
        if compute_main_loss:
            if isinstance(self.loss_fn, losses.MixOfLosses):
                distance = {}
                for loss_fn, weight in zip(self.loss_fn.losses, self.loss_fn.weights):
                    input1 = spec_x
                    input2 = spec_x_hat
                    name = loss_fn.__class__.__name__
                    if name == "MSSLoss":
                        input1 = x
                        input2 = x_hat
                    elif name == "Wasserstein1D":
                        if self.detach_weights:
                            input2 = self.transform(output["x_hat_weights_detached"])

                    loss_ = loss_fn(input1, input2, x_pos=x_pos, y_pos=y_pos) * weight
                    distance[name] = loss_
            else:
                input1 = spec_x
                input2 = spec_x_hat
                if self.loss_fn.__class__.__name__ == "MSSLoss":
                    input1 = x
                    input2 = x_hat
                distance = self.loss_fn(input1, input2, x_pos=x_pos, y_pos=y_pos)

            # Unpack losses if dict
            if isinstance(distance, dict):
                # Sum all losses, log all losses
                for key, value in distance.items():
                    _loss = value.mean()
                    self.log(f"loss/{step_name}/{key}", _loss, prog_bar=False)
                    loss += _loss
            else:
                loss = distance.mean()
                self.log(
                    f"loss/{step_name}/{self.loss_fn.__class__.__name__}", loss, prog_bar=False
                )

        # Regularization losses (not used in the paper)
        for reg in self.regularizations:
            name = reg.__class__.__name__
            if name == "ParameterLoss":
                name = f"{name}_{reg.parameter}"
            if name == "F0ConsistencyReg":
                _loss = reg(output["frequency_unit"], self.encode_pitch(output["x_hat"]))
            else:
                _loss = reg(output)
            loss += _loss
            self.log(f"loss/{step_name}/{name}", _loss)

        # log total loss
        self.log(f"loss/{step_name}", loss, prog_bar=True)

        # This is only for plotting on wandb
        x_transform_vals = self.spec_transform.get_frequencies()
        spec_x_reduced = self.spec_transform(x, reduce=True, restore_shape=False)
        spec_x_hat_reduced = self.spec_transform(x_hat, reduce=True, restore_shape=False)
        spec_x = self.spec_transform(x)
        spec_x_hat = self.spec_transform(x_hat)
        output.update({"x_transform_vals": x_transform_vals})
        output.update({"spec_x": spec_x, "spec_x_hat": spec_x_hat})
        output.update({"spec_x_reduced": spec_x_reduced, "spec_x_hat_reduced": spec_x_hat_reduced})

        return output, loss

    def training_step(self, batch, batch_idx):
        outputs, loss = self.shared_step(batch, batch_idx, "train")
        if self.current_epoch % self.trainer.check_val_every_n_epoch == 0:
            computed_metrics = compute_metrics(self, "train", **outputs)
            for key, value in computed_metrics.items():
                self.log(f"train_metrics/{key}", value)
            plot_and_log(self, batch_idx, "train", **outputs, transform=None)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs, loss = self.shared_step(batch, batch_idx, "val")
        computed_metrics = compute_metrics(self, "val", **outputs)
        for key, value in computed_metrics.items():
            self.log(f"val_metrics/{key}", value)
        plot_and_log(
            self, batch_idx, "val", idx=self.log_kwargs.get("idx", 0), **outputs, transform=None
        )
        return loss

    def test_step(self, batch, batch_idx, *args):
        outputs, loss = self.shared_step(batch, batch_idx, "test")
        computed_metrics = compute_metrics(self, "test", **outputs)
        return computed_metrics

    def test_epoch_end(self, outputs):
        # Average metrics across all batches
        create_tab = False
        fam = "all"
        if type(outputs[0]) == list:
            create_tab = True
            table = wandb.Table(columns=["Class"] + list(outputs[0][0].keys()))
        else:
            outputs = [outputs]
        for i, dataloader_outputs in enumerate(outputs):
            mean_metrics = {}
            if create_tab:
                fam = self.trainer.datamodule.test_dataloader()[i].dataset.instrument_family
            for key in dataloader_outputs[0].keys():
                mean_metrics[key] = torch.stack([x[key] for x in dataloader_outputs]).mean()
                if fam == "all":
                    self.log(f"test_metrics/{key}", mean_metrics[key])

            if create_tab:
                row = [fam] + list(mean_metrics.values())
                table.add_data(*row)

        if create_tab:
            wandb.log({"test_metrics": table})
