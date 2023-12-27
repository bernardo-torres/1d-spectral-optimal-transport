import torch
import numpy as np
import mir_eval.melody as mir_melody
from utils import create_harmonics, create_log_spaced_harmonics, safe_log, safe_log10

from features import compute_mag
from losses import MeanDifference, mean_difference, Wasserstein1D
import functools


@torch.inference_mode()
def mse(x, x_hat, sort=False):
    return MeanDifference("L2")(x, x_hat, sort=sort)


@torch.inference_mode()
def pitch_accuracy_fn(pred_pitch, true_pitch, type="raw"):
    """Computes pitch accuracy using mir_eval library
    Args:
        true_pitch: true pitch values in Hz. Shape (batch_size, n_partials) or (batch_size, time,
                n_partials)
        pred_pitch: predicted pitch values in Hz. Shape (batch_size, n_partials or (batch_size,
                time, n_partials)
        type: 'raw' or 'chroma'
    Returns:
        raw_pitch_accuracy: raw pitch accuracy"""

    if true_pitch.ndim == 3:
        true_pitch = true_pitch.reshape(-1, true_pitch.shape[-1])
        pred_pitch = pred_pitch.reshape(-1, pred_pitch.shape[-1])

    # Convert to cents
    true_pitch_cent = mir_melody.hz2cents(true_pitch.detach().cpu().flatten().numpy())
    pred_pitch_cent = mir_melody.hz2cents(pred_pitch.detach().cpu().flatten().numpy())
    voicing = np.ones_like(true_pitch_cent)
    if type == "raw":
        pitch_accuracy = mir_melody.raw_pitch_accuracy(
            voicing, true_pitch_cent, voicing, pred_pitch_cent
        )
    elif type == "chroma":
        pitch_accuracy = mir_melody.raw_chroma_accuracy(
            voicing, true_pitch_cent, voicing, pred_pitch_cent
        )
    elif type == "octave_difference":
        pitch_accuracy = mean_octave_difference(voicing, true_pitch_cent, voicing, pred_pitch_cent)
    else:
        raise ValueError("type must be raw or chroma")
    return torch.tensor(pitch_accuracy)


@torch.inference_mode()
def ms_spectral_distance(
    target_audio,
    audio,
    fft_sizes,
    mag_weight=1.0,
    logmag_weight=1.0,
    log_spectral_distance_weight=0,
    loss_type="L1",
):
    spectrogram_ops = []
    for size in fft_sizes:
        spectrogram_op = functools.partial(compute_mag, size=size, add_in_sqrt=1e-10)
        spectrogram_ops.append(spectrogram_op)

    loss = 0.0
    # Compute loss for each fft size.
    for i, loss_op in enumerate(spectrogram_ops):
        target_mag = loss_op(target_audio)
        value_mag = loss_op(audio)

        # Add magnitude loss.
        if mag_weight > 0:
            loss += mag_weight * mean_difference(target_mag, value_mag, loss_type)

        # Add logmagnitude loss, reusing spectrogram.
        if logmag_weight > 0:
            target = safe_log(target_mag)
            value = safe_log(value_mag)
            loss += logmag_weight * mean_difference(target, value, loss_type)

        if log_spectral_distance_weight > 0:
            target = 10 * safe_log10(target_mag**2)
            value = 10 * safe_log10(value_mag**2)
            loss += log_spectral_distance_weight * mean_difference(target, value, loss_type)

    return loss


def mean_octave_difference(ref_voicing, ref_cent, est_voicing, est_cent):
    """Computes the mean number of octave differences between a reference and
    estimated pitch sequence in cents. All 4 sequences must be of the same
    length.

    Examples
    --------
    >>> ref_time, ref_freq = mir_eval.io.load_time_series('ref.txt')
    >>> est_time, est_freq = mir_eval.io.load_time_series('est.txt')
    >>> (ref_v, ref_c,
    ...  est_v, est_c) = mir_eval.melody.to_cent_voicing(ref_time,
    ...                                                  ref_freq,
    ...                                                  est_time,
    ...                                                  est_freq)
    >>> mean_oct_diff = mean_octave_difference(ref_v, ref_c, est_v, est_c)

    Parameters
    ----------
    ref_voicing : np.ndarray
        Reference voicing array
    ref_cent : np.ndarray
        Reference pitch sequence in cents
    est_voicing : np.ndarray
        Estimated voicing array
    est_cent : np.ndarray
        Estimate pitch sequence in cents

    Returns
    -------
    mean_oct_diff : float
        Mean octave difference between corresponding reference and estimate
        pitch values

    """
    # Your code to validate voicing can be included here
    # When input arrays are empty, return 0
    if ref_voicing.size == 0 or est_cent.size == 0 or ref_cent.size == 0:
        return 0.0

    nonzero_freqs = np.logical_and(est_cent != 0, ref_cent != 0)

    if sum(nonzero_freqs) == 0:
        return 0.0

    freq_diff_cents = (ref_cent - est_cent)[nonzero_freqs]
    sign = np.sign(freq_diff_cents)
    freq_diff_cents = freq_diff_cents + 50 * sign  # 50 cents = 1/2 semitone, 3% error
    oct_diff = np.floor(np.abs(freq_diff_cents) / 1200)

    mean_oct_diff = np.sum(ref_voicing[nonzero_freqs] * oct_diff * sign) / np.sum(ref_voicing)

    return mean_oct_diff


@torch.inference_mode()
def wasserstein_distance(x, x_hat, p=1, n_fft=512):
    mag_x = compute_mag(x, size=n_fft).permute(0, 2, 1)
    mag_x_hat = compute_mag(x_hat, size=n_fft).permute(0, 2, 1)
    distance = Wasserstein1D(p=p, fixed_x=mag_x.shape[-1]).to(x.device)(mag_x, mag_x_hat)
    return distance


@torch.inference_mode()
def compute_metrics(trainer, step_name, **outputs):
    x = outputs["x"]
    x_hat = outputs["x_hat"]
    pitch = outputs["pitch"]
    pitch_hz = outputs["pitch_hz"]
    true_pitch = outputs["true_pitch"]
    true_pitch_hz = outputs["true_pitch_hz"]
    true_weights = outputs["true_weights"]
    weights = outputs.get("weights", torch.ones_like(true_weights))

    metrics_dict = {}

    if weights.ndim == 3 and true_weights.ndim == 2:
        true_weights = true_weights.unsqueeze(1).repeat(1, weights.shape[1], 1)

    if trainer.evaluation_metrics.get("mse", False):
        _mse = mse(x, x_hat)
        metrics_dict["mse"] = _mse
    if trainer.evaluation_metrics.get("log_spectral_distance"):
        log_spectral_distance = ms_spectral_distance(
            x,
            x_hat,
            fft_sizes=[1024],
            mag_weight=0,
            logmag_weight=0,
            log_spectral_distance_weight=1.0,
            loss_type="L2",
        )

        metrics_dict["log_spectral_distance"] = log_spectral_distance
    if trainer.evaluation_metrics.get("mss"):
        mss = ms_spectral_distance(
            x,
            x_hat,
            fft_sizes=[2048, 1024, 512, 256, 128, 64],
            mag_weight=1,
            logmag_weight=1,
            loss_type="L1",
        )

        metrics_dict["mss"] = mss
    if trainer.evaluation_metrics.get("pitch_mse", False):
        pitch_mse = mse(outputs["frequency_unit"], outputs["true_frequency_unit"], sort=True)
        pitch_mse_db = 10 * safe_log10(pitch_mse)
        metrics_dict["pitch_mse"] = pitch_mse
        metrics_dict["pitch_mse_db"] = pitch_mse_db

    if trainer.evaluation_metrics.get("raw_pitch_accuracy", False):
        raw_pitch_accuracy = pitch_accuracy_fn(pitch_hz, true_pitch_hz, type="raw")
        metrics_dict["raw_pitch_accuracy"] = raw_pitch_accuracy

    if trainer.evaluation_metrics.get("raw_chroma_accuracy", False):
        raw_chroma_accuracy = pitch_accuracy_fn(pitch_hz, true_pitch_hz, type="chroma")
        metrics_dict["raw_chroma_accuracy"] = raw_chroma_accuracy
    if trainer.evaluation_metrics.get("octave_difference", False):
        octave_difference = pitch_accuracy_fn(pitch_hz, true_pitch_hz, type="octave_difference")

        metrics_dict["octave_difference"] = octave_difference
    if trainer.evaluation_metrics.get("1-wasserstein", False):
        wasserstein = wasserstein_distance(x, x_hat)
        metrics_dict["1-wasserstein"] = wasserstein

    if trainer.evaluation_metrics.get("2-wasserstein", False):
        wasserstein = wasserstein_distance(x, x_hat, p=2)
        metrics_dict["2-wasserstein"] = wasserstein

    return metrics_dict
