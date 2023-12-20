import torch

import numpy as np
import functools
from typing import Callable, TypeVar

Number = TypeVar("Number", int, float, np.ndarray, torch.Tensor)


def get_fn_by_name(name: str, **kwargs) -> Callable:
    """Get scaling function by name."""
    # return if it is already a function
    if callable(name):
        return name
    if name == "exp_sigmoid":
        return functools.partial(exp_sigmoid, **kwargs)
    elif name == "frequencies_softmax":
        return functools.partial(frequencies_softmax, **kwargs)
    elif name == "identity":
        return lambda x: x
    elif name is None:
        return None
    else:
        raise ValueError("Unknown scaling function: {}".format(name))


def exp_sigmoid(x, exponent=10.0, max_value=2.0, threshold=1e-7):
    """Exponentiated Sigmoid pointwise nonlinearity.

    Bounds input to [threshold, max_value] with slope given by exponent.

          Args:
            x: Input tensor.
            exponent: In nonlinear regime (away from x=0), the output varies by this
              factor for every change of x by 1.0.
            max_value: Limiting value at x=inf.
            threshold: Limiting value at x=-inf. Stablizes training when outputs are
              pushed to 0.

          Returns:
            A tensor with pointwise nonlinearity applied.
    """

    x = x.type(torch.float32)
    exponent = torch.tensor(exponent, dtype=torch.float32, device=x.device)
    return max_value * torch.sigmoid(x) ** torch.log(exponent) + threshold


def get_cqt_n_bins(sr, fmin, bins_per_semitone=3):
    max_semitones_per_sr = int(np.floor(12 * np.log2(sr / 2) - 12 * np.log2(fmin)))
    return max_semitones_per_sr * bins_per_semitone


def unit_to_mu(unit: Number, unit_min: Number, unit_max: Number, clip: bool = False) -> Number:
    """Map a unit interval [0, 1] to the [0,1] range relatiev to [mu_min, mu_max],
    where mu_min and mu_max are the min and max values of the parameter mu.
    """
    mu = unit * (unit_max - unit_min) + unit_min
    return torch.clamp(mu, 0.0, 1.0) if clip else mu


def mu_to_unit(mu: Number, unit_min: Number, unit_max: Number, clip: bool = False) -> Number:
    """Map a frequency value in the range [0, 1] to the interval [mu_min, mu_min]."""
    unit = (mu - unit_min) / (unit_max - unit_min)
    return torch.clamp(unit, 0.0, 1.0) if clip else unit


def unit_to_hz(unit: Number, hz_min: Number, hz_max: Number, clip: bool = False) -> Number:
    """Map unit interval [0, 1] to [hz_min, hz_max], scaling logarithmically."""
    midi = unit_to_midi(unit, midi_min=hz_to_midi(hz_min), midi_max=hz_to_midi(hz_max), clip=clip)
    return midi_to_hz(midi)


def unit_to_midi(
    unit: Number, midi_min: Number = 20.0, midi_max: Number = 90.0, clip: bool = False
) -> Number:
    """Map the unit interval [0, 1] to MIDI notes."""
    unit = torch.clamp(unit, 0.0, 1.0) if clip else unit
    return midi_min + (midi_max - midi_min) * unit


def midi_to_hz(notes: Number) -> Number:
    """TF-compatible midi_to_hz function."""
    notes = torch_float32(notes)
    return 440.0 * (2.0 ** ((notes - 69.0) / 12.0))


def hz_to_midi(frequencies: Number) -> Number:
    """TF-compatible hz_to_midi function."""
    frequencies = torch_float32(frequencies)
    notes = 12.0 * (logb(frequencies, 2.0) - logb(440.0, 2.0)) + 69.0
    # Map 0 Hz to MIDI 0 (Replace -inf MIDI with 0.)
    notes = torch.where(
        frequencies <= 0.0,
        torch.tensor(0.0, dtype=torch.float32, device=frequencies.device),
        notes,
    )
    return notes


def hz_to_unit(
    hz: Number, hz_min: Number = 20.0, hz_max: Number = 8000.0, clip: bool = False
) -> Number:
    """Map [hz_min, hz_max] to unit interval [0, 1], scaling logarithmically."""
    midi = hz_to_midi(hz)
    return midi_to_unit(midi, midi_min=hz_to_midi(hz_min), midi_max=hz_to_midi(hz_max), clip=clip)


def midi_to_unit(
    midi: Number, midi_min: Number = 20.0, midi_max: Number = 90.0, clip: bool = False
) -> Number:
    """Map MIDI notes to the unit interval [0, 1]."""
    unit = (midi - midi_min) / (midi_max - midi_min)
    return torch.clamp(unit, 0.0, 1.0) if clip else unit


def torch_float32(x):
    """Ensure array/tensor is a float32 torch.Tensor."""
    if isinstance(x, torch.Tensor):
        return x.type(torch.float32)  # This is a no-op if x is float32.
    else:
        return torch.tensor(x, dtype=torch.float32)


def logb(x, base=2.0, safe=False):
    """Logarithm with base as an argument."""
    x = torch_float32(x)
    base = torch_float32(base)
    if safe:
        return safe_divide(safe_log(x), safe_log(base))
    else:
        return torch.log(x) / torch.log(base)


def safe_divide(numerator, denominator, eps=1e-7):
    """Avoid dividing by zero by adding a small epsilon."""
    safe_denominator = torch.where(
        denominator <= eps,
        torch.tensor(eps, dtype=torch.float32, device=denominator.device),
        denominator,
    )
    return numerator / safe_denominator


def safe_log(x, eps=1e-5):
    """Avoid taking the log of a non-positive number."""
    """ changed 0.0 to eps after issue https://github.com/magenta/ddsp/issues/191"""
    eps = torch.tensor(eps, device=x.device)
    # safe_x = torch.where(x <= 0.0, eps, x)
    safe_x = torch.where(x <= eps, eps, x)
    return torch.log(safe_x)


def safe_log10(x, eps=1e-5):
    eps = torch.tensor(eps, device=x.device)
    safe_x = torch.where(x <= eps, eps, x)
    return torch.log10(safe_x)


def normalize_stft_unit_energy(audio, NUM_FFT, HOPSIZE):
    """
    Normalize the STFT of an audio signal such that each time frame has unit energy.

    Parameters:
        audio (torch.Tensor): The input audio signal (batched or unbatched).
        NUM_FFT (int): The FFT size for the STFT.
        HOPSIZE (int): The hop size for the STFT.

    Returns:
        torch.Tensor: The recovered audio signal after STFT normalization.
    """

    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    # Calculate padding size
    pad_size = NUM_FFT // 2

    # Pad the audio along the time dimension
    audio_padded = torch.nn.functional.pad(audio, (pad_size, pad_size), "constant", 0)

    # Compute STFT
    S = torch.stft(
        audio_padded,
        n_fft=NUM_FFT,
        hop_length=HOPSIZE,
        win_length=NUM_FFT,
        center=True,
        return_complex=True,
    )

    # Compute magnitude
    amplitude = S.abs()

    # Normalize each time frame to have unit energy
    amplitude = amplitude / (torch.norm(amplitude, dim=-2, keepdim=True) + 1e-7)

    # Recover audio from normalized STFT (use correct phase)
    audio_recovered = torch.istft(
        amplitude * S / (S.abs() + 1e-7),
        n_fft=NUM_FFT,
        hop_length=HOPSIZE,
        win_length=NUM_FFT,
        center=True,
        return_complex=False,
    )

    # Remove padding along the time dimension
    audio_recovered = audio_recovered[:, pad_size:-pad_size]

    # Pad time dimension to match original audio length
    audio_recovered = torch.nn.functional.pad(
        audio_recovered, (0, audio.shape[-1] - audio_recovered.shape[-1]), "constant", 0
    )

    return audio_recovered.squeeze(0)


def create_harmonics(frequency_vector, n_harmonics):
    """Create harmonics of a given frequency vector.
    Args:
        frequency_vector: A tensor of shape [batch, 1, ...] of fundamental frequencies in Hz.
        n_harmonics: The number of harmonics to create.
    Returns:
        A tensor of harmonic frequencies with shape [batch, n_harmonics, ...].
    """

    # Create harmonics
    n_harmonics = torch.arange(1, n_harmonics + 1, device=frequency_vector.device)
    harmonics = frequency_vector * n_harmonics.unsqueeze(0)
    return harmonics


def create_log_spaced_harmonics(frequency_vector_unit, n_harmonics, hz_min, hz_max):
    """Create harmonics of a given normalized frequency vector.
    Args:
        frequency_vector: A tensor of shape [batch, 1, ...] of fundamental frequencies in [0,1].
        n_harmonics: The number of harmonics to create.
        hz_min: The minimum frequency in Hz (corresponding to 0).
        hz_max: The maximum frequency in Hz (corresponding to 1).
    Returns:
        A tensor of harmonic frequencies with shape [batch, n_harmonics, ...].
    """

    # Create harmonics
    n_harmonics = torch.arange(1, n_harmonics + 1, device=frequency_vector_unit.device)
    freq_vector = unit_to_hz(frequency_vector_unit, hz_min, hz_max)
    harmonics = freq_vector * n_harmonics.unsqueeze(0)
    harmonics = hz_to_unit(harmonics, hz_min, hz_max)
    return harmonics


def pad_for_stft(signal, frame_size, hop_length):
    """pads the given signal so that all samples are taken into account by torch.stft
     mimics tf.stft(pad_end=True) where the window is slid until it is completely beyond
     the signal.
    input has shape [batch_size, nb_timesteps]
    output has shape [batch_size, nb_timesteps + padding]"""

    signal_len = signal.shape[1]
    # incomplete_frame_len = signal_len % hop_length

    # ----- mimics tf.stft(pad_end=True)-----------------------------------
    # Calculate number of frames, using double negatives to round up.
    num_frames = -(-signal_len // hop_length)
    # Pad the signal by up to frame_length samples based on how many samples
    # are remaining starting from last_frame_position.
    pad_samples = max(0, frame_size + hop_length * (num_frames - 1) - signal_len)
    # ---------------------------------------------------------------------

    if pad_samples == 0:
        # no padding needed
        return signal
    else:
        signal = torch.nn.functional.pad(signal, pad=(0, pad_samples))
        return signal


def frequencies_softmax(
    freqs: torch.Tensor, depth: int = 64, hz_min: float = 20.0, hz_max: float = 8000.0
) -> torch.Tensor:
    """Softmax to logarithmically scale network outputs to frequencies.
    Args:
      freqs: Neural network outputs, [batch, time, n_sinusoids * depth] or
        [batch, time, n_sinusoids, depth].
      depth: If freqs is 3-D, the number of softmax components per a sinusoid to
        unroll from the last dimension.
      hz_min: Lowest frequency to consider.
      hz_max: Highest frequency to consider.
    Returns:
      A tensor of frequencies in hertz [batch, time, n_sinusoids].
    """
    if len(freqs.shape) == 3:
        # Add depth: [B, T, N*D] -> [B, T, N, D]
        freqs = _add_depth_axis(freqs, depth)
    else:
        depth = int(freqs.shape[-1])

    # Probs: [B, T, N, D].
    f_probs = torch.nn.functional.softmax(freqs, dim=-1)

    # [1, 1, 1, D]
    unit_bins = torch.linspace(0.0, 1.0, depth, device=f_probs.device)
    unit_bins = unit_bins[None, None, None, :]

    # unit_bins represents a number of frequencies in unit scale that are combined by
    # the softmax output.
    # This way, arbitrary frequencies can be chosen. The number of frequencies is given
    # by n_sinusoids.
    # [B, T, N]
    f_unit = torch.sum(unit_bins * f_probs, axis=-1, keepdim=False)
    return unit_to_hz(f_unit, hz_min=hz_min, hz_max=hz_max)


def _add_depth_axis(freqs: torch.Tensor, depth: int = 1) -> torch.Tensor:
    """Turns [batch, time, sinusoids*depth] to [batch, time, sinusoids, depth]."""
    freqs = freqs[..., None]
    # Unpack sinusoids dimension.
    n_batch, n_time, n_combined, _ = freqs.shape
    n_sinusoids = int(n_combined) // depth
    return torch.reshape(freqs, (n_batch, n_time, n_sinusoids, depth))


def log10(x, eps=1e-5):
    """Logarithm with base 10."""
    return logb(x, base=10, safe=True)


def power_to_db(power, ref_db=0.0, range_db=80):
    """Converts power from linear scale to decibels."""

    # Convert to decibels.
    range_db = torch.tensor(range_db).to(power)
    pmin = 10 ** -(range_db / 10.0)
    power = torch.maximum(pmin, power)
    db = 10.0 * log10(power)

    # Set dynamic range.
    db -= torch.tensor(ref_db).to(power)
    db = torch.maximum(db, -range_db)
    return db
