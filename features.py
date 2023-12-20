# Some of this code was adapted from https://github.com/SonyCSLParis/pesto 
# Some other bits come from https://github.com/schufo/umss

import torch
import torch.nn as nn

from nnAudio import features
from utils import get_cqt_n_bins, torch_float32, pad_for_stft, power_to_db
import ddsp as core
import librosa as li
import functools
from scipy.signal import get_window


def get_default_cqt_args(sr):
    return dict(
        bins_per_semitone=3,
        fmin=32.7,
        n_bins=get_cqt_n_bins(sr, 32.7, 3),
        output_format="Complex",
        verbose=False,
        center=True,
        pad_mode="constant",
    )


def get_default_stft_args(sr):
    return dict(
        log=False, n_fft=1024, hop_length=256, center=True, output_format="Magnitude"
    )


def get_transform(transform, sample_rate):
    if isinstance(transform, dict):
        transform = transform.copy()
        name = transform.pop("type")
        kwargs = transform
        get_default = False
    else:
        name = transform
        get_default = True

    if name == "stft":
        if get_default:
            kwargs = get_default_stft_args(sample_rate)
        kwargs.update({"sr": sample_rate})
        # return STFT(**kwargs)
        return TorchSTFT(**kwargs)
    elif name == "cqt":
        if get_default:
            kwargs = get_default_cqt_args(sample_rate)
        if kwargs.get("n_bins", None) == "auto":
            kwargs["n_bins"] = get_cqt_n_bins(
                sample_rate, kwargs["fmin"], kwargs["bins_per_semitone"]
            )
        kwargs.update({"sr": sample_rate})
        return CQT(**kwargs)
    elif name == "identity":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown transform {name}")


class STFT(nn.Module):
    def __init__(self, **kwargs):
        super(STFT, self).__init__()
        self.log = kwargs.pop("log", False)
        self.transform = features.STFT(**kwargs)  # .to("cuda")
        # self.transform = torch.fft.rfft

    def forward(self, x, **kwargs):
        x = self.transform(x).permute(0, 2, 1)
        if kwargs.get("reduce", False):
            x = x.mean(dim=1)
        # if kwargs.get("restore_shape", False):
        #     x = x.unsqueeze(1)
        if kwargs.get("log", False) or self.log:
            x = core.safe_log(x)
        return x  # (batch, time, freq)

    def get_frequencies(self):
        return self.transform.bins2freq


class TorchSTFT(nn.Module):
    def __init__(self, **kwargs):
        super(TorchSTFT, self).__init__()

        self.n_fft = kwargs.pop("n_fft", 1024)
        hop_length = kwargs.pop("hop_length", 256)
        self.sr = kwargs.pop("sr", 16000)
        self.log = kwargs.pop("log", False)
        window = kwargs.pop("window", None)
        if window is not None:
            window = get_window(window, self.n_fft)

        self.transform = functools.partial(
            compute_mag,
            size=self.n_fft,
            overlap=1 - hop_length / self.n_fft,
            window=window,
        )

    def forward(self, x, **kwargs):
        x = self.transform(x).permute(0, 2, 1)
        if kwargs.get("reduce", False):
            x = x.mean(dim=1)
        if kwargs.get("log", False) or self.log:
            x = core.safe_log(x)
        return x  # (batch, time, freq)

    def get_frequencies(self):
        return torch.fft.rfftfreq(self.n_fft, d=1 / self.sr)


class CQT(nn.Module):
    def __init__(self, bins_per_semitone: int = 3, **cqt_kwargs):
        super(CQT, self).__init__()
        self.bins_per_semitone = bins_per_semitone

        self.cqt_kwargs = cqt_kwargs
        self.cqt_kwargs["bins_per_octave"] = 12 * bins_per_semitone
        self.cqt = None

        self.sr = None
        self.step_size = None

        # log-magnitude
        self.eps = torch.finfo(torch.float32).eps

        # cropping (not needed for this project)
        # self.lowest_bin = int(11 * self.bins_per_semitone / 2 + 0.5)
        # self.highest_bin = self.lowest_bin + 88 * self.bins_per_semitone
        self.lowest_bin = 0
        # default to entire spectrum
        self.highest_bin = self.cqt_kwargs["n_bins"]
        self.log = self.cqt_kwargs.pop("log", False)

        self.init_cqt_layer(device="cuda")

    def forward(self, x: torch.Tensor, log=False, reduce=False, restore_shape=True):
        r"""

        Args:
            x: audio waveform, any sampling rate, shape (num_samples)

        Returns:
            log-magnitude CQT, shape (
        """
        # compute CQT from input waveform
        complex_cqt = torch.view_as_complex(self.cqt(x)).permute(
            0, 2, 1
        )  # batch, time, freq
        batch_size, time, freq = complex_cqt.shape

        # Merge two first dimensions 
        complex_cqt = complex_cqt.reshape(-1, complex_cqt.shape[2]).unsqueeze(
            1
        )  # batch*time, 1, freq

        # reshape and crop borders to fit training input shape
        complex_cqt = complex_cqt[..., self.lowest_bin: self.highest_bin]

        log_cqt = complex_cqt.abs()  # .clamp_(min=self.eps)

        if log or self.log:
            log_cqt = core.safe_log(log_cqt, eps=self.eps).mul_(20)

        if reduce:
            log_cqt = log_cqt.reshape(batch_size, time, freq)
            log_cqt = log_cqt.mean(
                dim=1, keepdim=True
            )  # Mean over time 
            
        if restore_shape:
            log_cqt = log_cqt.reshape(batch_size, -1, freq)  # batch, time, freq
        return log_cqt

    def init_cqt_layer(self, sr: int = None, hop_length: int = None, device="cuda"):
        # self.step_size = hop_length / sr
        if hop_length is not None:
            self.cqt_kwargs["hop_length"] = hop_length
        if sr is not None:
            self.cqt_kwargs["sr"] = sr
        self.cqt = features.cqt.CQT(**self.cqt_kwargs).to(device)

    def get_frequencies(self):
        return self.cqt.frequencies[self.lowest_bin: self.highest_bin]


def stft(audio, frame_size=2048, overlap=0.75, center=False, pad_end=True, window=None):
    """Differentiable stft in PyTorch, computed in batch."""
    audio = torch_float32(audio)
    hop_length = int(frame_size * (1.0 - overlap))
    if pad_end:
        # pad signal so that STFT window is slid until it is
        # completely beyond the signal
        audio = pad_for_stft(audio, frame_size, hop_length)
    assert frame_size * overlap % 2.0 == 0.0
    if window is None:
        window = torch.hann_window(int(frame_size), device=audio.device)
    else:
        window = torch_float32(window).to(audio.device)
    s = torch.stft(
        input=audio,
        n_fft=int(frame_size),
        hop_length=hop_length,
        win_length=int(frame_size),
        window=window,
        center=center,
        normalized=True,
        return_complex=True,
    )
    return s


def compute_mag(
    audio,
    size=2048,
    overlap=0.75,
    pad_end=True,
    center=False,
    add_in_sqrt=0.0,
    window=None,
):
    # change because of change in torch stft
    stft_cmplx = stft(
        audio,
        frame_size=size,
        overlap=overlap,
        center=center,
        pad_end=pad_end,
        window=window,
    )
    # add_in_sqrt is added before sqrt is taken because the gradient of torch.sqrt(0) is NaN
    mag = stft_cmplx.abs()
    return torch_float32(mag)


def a_weighting_from_audio(
    audio,
    num_fft,
    hopsize,
    sample_rate=16000,
    weighting=None,
):

    is_1d = True if audio.ndim == 1 else False
    audio = audio.unsqueeze(0) if is_1d else audio

    S = torch.stft(
        audio,
        n_fft=num_fft,
        hop_length=hopsize,
        win_length=num_fft,
        center=True,
        return_complex=True,
    )  # S is complex

    # S = np.log(abs(S) + 1e-7)
    amplitude = S.abs()  # recover magnitude from complex STFT
    power = amplitude**2

    if weighting is None:
        f = li.fft_frequencies(sr=sample_rate, n_fft=num_fft)
        a_weighting = torch.tensor(li.A_weighting(f)).to(amplitude)
        a_weighting = a_weighting.repeat(audio.shape[0], 1)

        # Perform weighting in linear scale, a_weighting given in decibels.
        weighting = 10 ** (a_weighting / 10)

    power = power * weighting.unsqueeze(-1)

    avg_power = torch.mean(power, dim=-2)  # [..., :-1]

    loudness = power_to_db(avg_power, ref_db=0, range_db=80)

    loudness = loudness[0] if is_1d else loudness
    return loudness


def get_loudness(audio, hopsize, num_fft=1024, sample_rate=16000, weighting=None):
    return (
        a_weighting_from_audio(
            audio, num_fft, hopsize, sample_rate, weighting=weighting
        )
        + 50
    ) / 80
