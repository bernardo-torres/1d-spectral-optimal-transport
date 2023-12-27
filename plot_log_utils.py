# Just some utils to plot and log signals, spectograms, etc during training

import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import wandb


def plot_spectogram(tensor, n_fft=1024, sample_rate=None, with_pitch=None):
    """Plots spectogram of a tensor.
    Args:
        tensor: torch.Tensor of shape [n_samples]
        n_fft: int, number of FFT bins
        sample_rate: int, sample rate of tensor
        with_pitch: tensor, whether to plot pitch on top of spectogram
    Returns:
        shape of spectogram"""

    spec = torchaudio.transforms.Spectrogram(n_fft=n_fft)(tensor)
    if sample_rate is None:
        sample_rate = 1
    lims = [0, spec.shape[1], 0, sample_rate / 2]
    plt.imshow((spec + 1e-7).log2(), origin="lower", aspect="auto", extent=lims)
    if with_pitch is not None:
        # Create a time axis based on the length of the pitch curve
        time_axis = torch.arange(with_pitch.shape[0], dtype=torch.float32)
        # Scale the time axis to match the x-axis of the spectrogram
        x_scale = spec.shape[1] / with_pitch.shape[0]
        time_axis *= x_scale

        # Plot the pitch curve on the same axis
        plt.plot(time_axis, with_pitch, color="white", linewidth=1)
    return spec.shape


def plot_signal_decorator(plot_func):
    @torch.inference_mode()
    def wrapper(
        batch_idx, signal, step_name, idx=0, name="", apply_decorator=True, *args, **kwargs
    ):
        if not apply_decorator:
            plot_func(signal, idx=idx, *args, **kwargs)
            return None

        _idx = idx
        if isinstance(idx, int):
            _idx = [idx]
        if batch_idx in _idx:
            plt.ioff()
            # plt.figure(figsize=(10, 10))
            plot_func(signal, *args, **kwargs)
            plt.title(name)
            extra = f"_{batch_idx}" if not isinstance(idx, int) else ""
            wandb.log({f"Signal{extra}_{step_name}/{name}": wandb.Image(plt)})
            plt.close()

    return wrapper


@torch.inference_mode()
def plot_and_log(trainer, batch_idx, step_name, idx=0, transform=None, **outputs):
    # idx = 1
    log_signal(batch_idx, outputs["x"], step_name, idx=idx, name="Original Signal")
    log_signal(batch_idx, outputs["x_hat"], step_name, idx=idx, name="Reconstructed Signal")
    log_signal(batch_idx, outputs["spec_x"], step_name, idx=idx, name="Original Spectrum")
    log_signal(batch_idx, outputs["spec_x_hat"], step_name, idx=idx, name="Reconstructed Spectrum")
    log_signal(
        batch_idx,
        outputs["spec_x"] + outputs["spec_x_hat"],
        step_name,
        idx=idx,
        name="Original and reconstructed Spectrum",
    )
    log_signal(
        batch_idx,
        outputs["spec_x_reduced"],
        step_name,
        idx=idx,
        name="Original Spectrum (reduced)",
        x_values=outputs.get("x_transform_vals", None),
    )
    log_signal(
        batch_idx,
        outputs["spec_x_hat_reduced"],
        step_name,
        idx=idx,
        name="Reconstructed Spectrum (reduced)",
        x_values=outputs.get("x_transform_vals", None),
    )
    log_signals(
        batch_idx,
        {"Original": outputs["spec_x_reduced"], "Reconstructed": outputs["spec_x_hat_reduced"]},
        step_name,
        idx=idx,
        name="Original vs Reconstructed",
        x_values=outputs.get("x_transform_vals", None).numpy(),
    )
    # log_signals(
    #     batch_idx,
    #     {"Original": outputs["spec_x_reduced"], "Reconstructed": outputs["spec_x_hat_reduced"]},
    #     step_name,
    #     name="Original vs Reconstructed (zoomed))",
    #     zoom=0.4,
    # )

    if outputs.get("gain", None) is not None:
        log_signal(batch_idx, outputs["gain"], step_name, idx=idx, name="Gain")

    if outputs.get("loudness", None) is not None:
        log_signal(batch_idx, outputs["loudness"], step_name, idx=idx, name="Loudness")

    if outputs.get("probabilities", None) is not None:
        if outputs["probabilities"].shape[1] > 1:
            vert_lines = [
                outputs.get("true_frequency_unit", None)[:, i].cpu().numpy()
                if outputs.get("true_frequency_unit", None) is not None
                else None
                for i in range(outputs["probabilities"].shape[1])
            ]
            for i in range(outputs["probabilities"].shape[1]):
                # vert_lines = [outputs.get("true_frequency_unit", None)[:, i].cpu().numpy() if 
                # outputs.get("true_frequency_unit", None) is not None else None
                log_histogram(
                    batch_idx,
                    outputs["probabilities"][:, i, :].unsqueeze(1),
                    step_name,
                    idx=idx,
                    name=f"Probabilities {i}",
                    x_values=trainer.feature_extractor.get_frequencies(),
                    vertical_line=vert_lines,
                )
        else:
            log_histogram(
                batch_idx,
                outputs["probabilities"],
                step_name,
                idx=idx,
                name="Probabilities",
                x_values=trainer.feature_extractor.get_frequencies(),
                vertical_line=outputs.get("true_frequency_unit", None).cpu().numpy(),
            )

    if outputs.get("kernel", None) is not None:
        log_signal(batch_idx, outputs["kernel"], step_name, idx=idx, name="Kernel")


def plot_signal(signal, idx=0, label="", spec=False):
    if signal.ndim == 2:
        signal = signal[idx]
    elif signal.ndim == 3:
        signal = signal[idx]  # remove batch dimension
        if signal.shape[0] == 1:
            signal = signal[0]
        else:
            # raise NotImplementedError
            plt.imshow(signal.permute(1, 0).detach().cpu().numpy(), origin="lower", aspect="auto")
            return
    if spec:
        plot_spectogram(signal.detach().cpu().numpy(), n_fft=512)
    else:
        plt.plot(signal.detach().cpu().numpy(), label=label)


@plot_signal_decorator
def log_signal(signal, idx=0, spec=False, img=False, label="", x_values=None):
    plot_signal(signal, idx, spec=spec, label=label)
    if x_values is not None:
        change_ticks(x_values, signal.shape[-1])


@plot_signal_decorator
def log_signals(signals_dict, idx=0, zoom=1, spec=False, x_values=None):
    for name, signal in signals_dict.items():
        # if signal.ndim == 2:
        #     signal = signal[0]
        # elif signal.ndim == 3:
        #     signal = signal[0]
        #     if signal.shape[0] == 1:
        #         signal = signal[0]
        #     else:
        #         raise NotImplementedError
        if zoom != 1:
            signal = signal[: int(signal.shape[-1] * zoom)]
        # log_signal(0, signal, "any_step", idx=0, spec=spec, label=name, apply_decorator=False)
        plot_signal(signal, idx=idx, spec=spec, label=name)
        # plt.plot(signal.detach().cpu().numpy(), label=name)
    # plt.ylim(0, 0.8)
    if x_values is not None:
        change_ticks(x_values, signal.shape[-1])
    plt.legend()


@plot_signal_decorator
def log_histogram(signal, idx=0, x_values=None, vertical_line=None):
    if signal.ndim == 2:
        signal = signal.unsqueeze(1)
    x = torch.arange(signal.shape[-1])
    for i, mode in enumerate(signal[idx].detach().cpu().numpy()):
        plt.bar(
            x,
            signal[idx].detach().cpu().numpy()[i],
            alpha=0.4,
            width=1,
            # label=f"Mode {i}",
        )
        if vertical_line is not None:
            if isinstance(vertical_line, list):
                for v_line in vertical_line:
                    plt.axvline(
                        v_line[idx] * signal.shape[-1], color="black", linestyle="--", alpha=0.5
                    )
            else:
                plt.axvline(
                    vertical_line[idx] * signal.shape[-1], color="black", linestyle="--", alpha=0.5
                )
    if x_values is not None:
        change_ticks(x_values, signal.shape[-1])
    # plt.legend()


def change_ticks(x_values, signal_size):
    if x_values is not None:
        x_tick_values = np.round(x_values, 1)
        x_tick_values = x_tick_values[:: int(len(x_tick_values) / 10)]

        plt.xticks(
            np.arange(0, signal_size, int(signal_size / 10)),
            x_tick_values,
            rotation=45,
        )
