# PyTorch implementation of DDSP following closely the original code
# https://github.com/magenta/ddsp/blob/master/ddsp/synths.py
# Code is adapted from https://github.com/schufo/umss


import ddsp as core
from utils import get_fn_by_name


import torch
import torch.nn as nn


# Processor Base Class ---------------------------------------------------------
class Processor(nn.Module):
    """Abstract base class for signal processors.
    Since most effects / synths require specificly formatted control signals
    (such as amplitudes and frequenices), each processor implements a
    get_controls(inputs) method, where inputs are a variable number of tensor
    arguments that are typically neural network outputs. Check each child class
    for the class-specific arguments it expects. This gives a dictionary of
    controls that can then be passed to get_signal(controls). The
    get_outputs(inputs) method calls both in succession and returns a nested
    output dictionary with all controls and signals.
    """

    def __init__(self):
        super().__init__()

    def forward(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
        """Convert input tensors arguments into a signal tensor."""

        controls = self.get_controls(*args, **kwargs)
        signal = self.get_signal(**controls)
        return signal

    def get_controls(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> dict:
        """Convert input tensor arguments into a dict of processor controls."""
        raise NotImplementedError

    def get_signal(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
        """Convert control tensors into a signal tensor."""
        raise


class Sinusoidal(Processor):
    """Synthesize audio with a bank of arbitrary sinusoidal oscillators."""

    def __init__(
        self,
        n_samples=64000,
        sample_rate=16000,
        amp_scale_fn="exp_sigmoid",
        amp_resample_method="window",
        freq_scale_fn="frequencies_softmax",
        harmonic=0,
        apply_roll_off=False,
    ):

        super().__init__()
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.amp_scale_fn = get_fn_by_name(amp_scale_fn)
        self.amp_resample_method = amp_resample_method
        self.freq_scale_fn = get_fn_by_name(freq_scale_fn)
        self.harmonic = harmonic
        self.apply_roll_off = apply_roll_off

    def get_controls(self, amplitudes, frequencies):
        """Convert network output tensors into a dictionary of synthesizer controls.
        Args:
          amplitudes: 3-D Tensor of synthesizer controls, of shape
            [batch, time, n_sinusoids].
          frequencies: 3-D Tensor of synthesizer controls, of shape
            [batch, time, n_sinusoids]. Expects strictly positive in Hertz.
        Returns:
          controls: Dictionary of tensors of synthesizer controls.
        """
        # Scale the inputs.
        if self.amp_scale_fn is not None:
            amplitudes = self.amp_scale_fn(amplitudes)

        if self.freq_scale_fn is not None:
            frequencies = self.freq_scale_fn(frequencies)

        if self.harmonic:
            frequencies = core.get_harmonic_frequencies(frequencies, amplitudes.shape[-1])

        amplitudes = core.remove_above_nyquist(frequencies, amplitudes, self.sample_rate)

        return {"amplitudes": amplitudes, "frequencies": frequencies}

    def get_signal(self, amplitudes, frequencies):
        """Synthesize audio with sinusoidal synthesizer from controls.
        Args:
          amplitudes: Amplitude tensor of shape [batch, n_frames, n_sinusoids].
            Expects float32 that is strictly positive.
          frequencies: Tensor of shape [batch, n_frames, n_sinusoids].
            Expects float32 in Hertz that is strictly positive.
        Returns:
          signal: A tensor of harmonic waves of shape [batch, n_samples].
        """
        # Create sample-wise envelopes.
        amplitude_envelopes = core.resample(
            amplitudes,
            self.n_samples,
            method=self.amp_resample_method,
            add_endpoint=True,
        )
        frequency_envelopes = core.resample(frequencies, self.n_samples)

        signal = core.oscillator_bank(
            frequency_envelopes=frequency_envelopes,
            amplitude_envelopes=amplitude_envelopes,
            sample_rate=self.sample_rate,
        )

        if self.apply_roll_off:
            harmonics_roll_off = torch.tensor(6).to(signal.device)
            filter_mag = core.slope_frequency_response(harmonics_roll_off, n_freqs=65, f_ref=500)[0]
            # Expand to match batch size
            filter_mag = filter_mag.expand(signal.shape[0], -1)
            signal = core.frequency_filter(signal, filter_mag)

        return signal
