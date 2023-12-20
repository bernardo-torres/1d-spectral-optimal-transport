import torch
import numpy as np
from scipy import fftpack


def get_harmonic_frequencies(frequencies: torch.Tensor, n_harmonics: int) -> torch.Tensor:
    """Create integer multiples of the fundamental frequency.

    Args:
      frequencies: Fundamental frequencies (Hz). Shape [batch_size, :, 1].
      n_harmonics: Number of harmonics.

    Returns:
      harmonic_frequencies: Oscillator frequencies (Hz).
        Shape [batch_size, :, n_harmonics].
    """

    frequencies = frequencies.type(torch.float32)

    f_ratios = torch.linspace(1.0, float(n_harmonics), int(n_harmonics), device=frequencies.device)
    harmonic_frequencies = frequencies * f_ratios
    return harmonic_frequencies


def remove_above_nyquist(
    frequency_envelopes: torch.Tensor, amplitude_envelopes: torch.Tensor, sample_rate: int = 16000
) -> torch.Tensor:
    """Set amplitudes for oscillators above nyquist to 0.

    Args:
      frequency_envelopes: Sample-wise oscillator frequencies (Hz). Shape
        [batch_size, n_samples, n_sinusoids].
      amplitude_envelopes: Sample-wise oscillator amplitude. Shape [batch_size,
        n_samples, n_sinusoids].
      sample_rate: Sample rate in samples per a second.

    Returns:
      amplitude_envelopes: Sample-wise filtered oscillator amplitude.
        Shape [batch_size, n_samples, n_sinusoids].
    """
    frequency_envelopes = frequency_envelopes.type(torch.float32)
    amplitude_envelopes = amplitude_envelopes.type(torch.float32)

    amplitude_envelopes = torch.where(
        (frequency_envelopes >= sample_rate / 2.0),
        torch.zeros_like(amplitude_envelopes),
        amplitude_envelopes,
    )

    return amplitude_envelopes


def resample(
    inputs: torch.Tensor, n_timesteps: int, method: str = "bilinear", add_endpoint: bool = True
) -> torch.Tensor:
    """Interpolates a tensor from n_frames to n_timesteps.
    Args:
      inputs: Framewise 1-D, 2-D, 3-D, or 4-D Tensor. Shape [n_frames],
        [batch_size, n_frames], [batch_size, n_frames, channels], or
        [batch_size, n_frames, n_freq, channels].
      n_timesteps: Time resolution of the output signal.
      method: Type of resampling, must be in ['nearest', 'bilinear', 'bicubic',
        'window']. Linear and cubic ar typical bilinear, bicubic interpolation.
        'window' uses overlapping windows (only for upsampling) which is smoother
        for amplitude envelopes with large frame sizes.
      add_endpoint: Hold the last timestep for an additional step as the endpoint.
        Then, n_timesteps is divided evenly into n_frames segments. If false, use
        the last timestep as the endpoint, producing (n_frames - 1) segments with
        each having a length of n_timesteps / (n_frames - 1).
    Returns:
      Interpolated 1-D, 2-D, 3-D, or 4-D Tensor. Shape [n_timesteps],
        [batch_size, n_timesteps], [batch_size, n_timesteps, channels], or
        [batch_size, n_timesteps, n_freqs, channels].
    Raises:
      ValueError: If method is 'window' and input is 4-D.
      ValueError: If method is not one of 'nearest', 'bilinear', 'bicubic', or
        'window'.
    """
    inputs = inputs.type(torch.float32)
    is_1d = len(inputs.shape) == 1
    is_2d = len(inputs.shape) == 2
    is_4d = len(inputs.shape) == 4

    # Ensure inputs are at least 3d.
    if is_1d:
        inputs = inputs[None, :, None]
    elif is_2d:
        inputs = inputs[:, :, None]

    # resample
    if method == "window":
        outputs = upsample_with_windows(inputs, n_timesteps, add_endpoint)
    elif method in ["nearest", "bilinear", "bicubic"]:
        outputs = inputs[:, :, :, None] if not is_4d else inputs

        outputs = outputs.permute(0, 2, 1, 3)  # [batch_size, n_channels, n_frames, optional]
        outputs = torch.nn.functional.interpolate(
            outputs,
            size=[n_timesteps, outputs.shape[3]],
            mode=method,
            align_corners=not add_endpoint,
        )
        outputs = outputs.permute(0, 2, 1, 3)  # [batch_size, n_frames, n_channels, optional]
        outputs = outputs[:, :, :, 0] if not is_4d else outputs
    else:
        raise ValueError(
            "Method ({}) is invalid. Must be one of {}.".format(
                method, "['nearest', 'bilinear', 'bicubic', 'window']"
            )
        )

    # Return outputs to the same dimensionality of the inputs.
    if is_1d:
        outputs = outputs[0, :, 0]
    elif is_2d:
        outputs = outputs[:, :, 0]

    return outputs


def upsample_with_windows(
    inputs: torch.Tensor, n_timesteps: int, add_endpoint: bool = True
) -> torch.Tensor:
    """Upsample a series of frames using using overlapping hann windows.
    Good for amplitude envelopes.

    Args:
      inputs: Framewise 3-D tensor. Shape [batch_size, n_frames, n_channels].
      n_timesteps: The time resolution of the output signal.
      add_endpoint: Hold the last timestep for an additional step as the endpoint.
        Then, n_timesteps is divided evenly into n_frames segments. If false, use
        the last timestep as the endpoint, producing (n_frames - 1) segments with
        each having a length of n_timesteps / (n_frames - 1).

    Returns:
      Upsampled 3-D tensor. Shape [batch_size, n_timesteps, n_channels].

    Raises:
      ValueError: If input does not have 3 dimensions.
      ValueError: If attempting to use function for downsampling.
      ValueError: If n_timesteps is not divisible by n_frames (if add_endpoint is
        true) or n_frames - 1 (if add_endpoint is false).
    """
    inputs = inputs.type(torch.float32)

    if len(inputs.shape) != 3:
        raise ValueError(
            "Upsample_with_windows() only supports 3 dimensions, " "not {}.".format(inputs.shape)
        )

    # Mimic behavior of tf.image.resize.
    # For forward (not endpointed), hold value for last interval.
    if add_endpoint:
        inputs = torch.cat([inputs, inputs[:, -1:, :]], dim=1)

    n_frames = int(inputs.shape[1])
    n_intervals = n_frames - 1

    if n_frames >= n_timesteps:
        raise ValueError(
            "Upsample with windows cannot be used for downsampling"
            "More input frames ({}) than output timesteps ({})".format(n_frames, n_timesteps)
        )

    if n_timesteps % n_intervals != 0.0:
        minus_one = "" if add_endpoint else " - 1"
        raise ValueError(
            "For upsampling, the target number of timesteps must be divisible "
            "by the number of input frames{}. (timesteps:{}, frames:{}, "
            "add_endpoint={}).".format(minus_one, n_timesteps, n_frames, add_endpoint)
        )

    # Constant overlap-add, half overlapping windows.
    hop_size = n_timesteps // n_intervals
    window_length = 2 * hop_size
    window = torch.hann_window(window_length, device=inputs.device)  # [window]

    # Transpose for overlap_and_add.
    x = inputs.permute(0, 2, 1)  # [batch_size, n_channels, n_frames]

    # Broadcast multiply.
    # Add dimension for windows [batch_size, n_channels, window, n_frames].
    x = x[:, :, None, :]
    window = window[None, None, :, None]
    x_windowed = x * window

    n_channels = x.shape[1]
    x_windowed = x_windowed.reshape((-1, n_channels * window_length, n_frames))

    # overlap and add
    x = torch.nn.functional.fold(
        x_windowed,
        output_size=(1, n_timesteps + window_length),
        kernel_size=(1, window_length),
        stride=(1, hop_size),
    )
    # x is [batch_size, n_channels, 1, n_timesteps]

    x = x.squeeze(2)  # [batch_size, n_channels n_timesteps]

    # Transpose back.
    x = x.permute(0, 2, 1)  # [batch_size, n_timesteps, n_channels]

    # Trim the rise and fall of the first and last window.
    return x[:, hop_size:-hop_size, :]


def oscillator_bank(
    frequency_envelopes: torch.Tensor,
    amplitude_envelopes: torch.Tensor,
    sample_rate: int = 16000,
    sum_sinusoids: bool = True,
    use_angular_cumsum: bool = False,
) -> torch.Tensor:
    """Generates audio from sample-wise frequencies for a bank of oscillators.

    Args:
      frequency_envelopes: Sample-wise oscillator frequencies (Hz). Shape
        [batch_size, n_samples, n_sinusoids].
      amplitude_envelopes: Sample-wise oscillator amplitude. Shape [batch_size,
        n_samples, n_sinusoids].
      sample_rate: Sample rate in samples per a second.
      sum_sinusoids: Add up audio from all the sinusoids.
      use_angular_cumsum: If synthesized examples are longer than ~100k audio
        samples, consider use_angular_cumsum to avoid accumulating noticible phase
        errors due to the limited precision of tf.cumsum. Unlike the rest of the
        library, this property can be set with global dependency injection with
        gin. Set the gin parameter `oscillator_bank.use_angular_cumsum=True`
        to activate. Avoids accumulation of errors for generation, but don't use
        usually for training because it is slower on accelerators.

    Returns:
      wav: Sample-wise audio. Shape [batch_size, n_samples, n_sinusoids] if
        sum_sinusoids=False, else shape is [batch_size, n_samples].
    """
    frequency_envelopes = frequency_envelopes.type(torch.float32)
    amplitude_envelopes = amplitude_envelopes.type(torch.float32)

    # Don't exceed Nyquist.
    amplitude_envelopes = remove_above_nyquist(
        frequency_envelopes, amplitude_envelopes, sample_rate
    )

    # Angular frequency, Hz -> radians per sample.
    omegas = frequency_envelopes * (2.0 * np.pi)  # rad / sec
    omegas = omegas / float(sample_rate)  # rad / sample

    # Accumulate phase and synthesize.
    if use_angular_cumsum:
        # Avoids accumulation errors.
        phases = angular_cumsum(omegas)
    else:
        # Pytorch cumsum is not deterministic...
        # torch.use_deterministic_algorithms(False)        
        phases = torch.cumsum(omegas, dim=1)
        # torch.use_deterministic_algorithms(True)

    # Convert to waveforms.
    wavs = torch.sin(phases)
    audio = amplitude_envelopes * wavs  # [mb, n_samples, n_sinusoids]
    if sum_sinusoids:
        audio = torch.sum(audio, dim=-1)  # [mb, n_samples]
    return audio


def angular_cumsum(angular_frequency, chunk_size=1000):
    """Get phase by cumulative sumation of angular frequency.

    Custom cumsum splits first axis into chunks to avoid accumulation error.
    Just taking tf.sin(tf.cumsum(angular_frequency)) leads to accumulation of
    phase errors that are audible for long segments or at high sample rates. Also,
    in reduced precision settings, cumsum can overflow the threshold.

    During generation, if syntheiszed examples are longer than ~100k samples,
    consider using angular_sum to avoid noticible phase errors. This version is
    currently activated by global gin injection. Set the gin parameter
    `oscillator_bank.use_angular_cumsum=True` to activate.

    Given that we are going to take the sin of the accumulated phase anyways, we
    don't care about the phase modulo 2 pi. This code chops the incoming frequency
    into chunks, applies cumsum to each chunk, takes mod 2pi, and then stitches
    them back together by adding the cumulative values of the final step of each
    chunk to the next chunk.

    Seems to be ~30% faster on CPU, but at least 40% slower on TPU.

    Args:
      angular_frequency: Radians per a sample. Shape [batch, time, ...].
        If there is no batch dimension, one will be temporarily added.
      chunk_size: Number of samples per a chunk. to avoid overflow at low
         precision [chunk_size <= (accumulation_threshold / pi)].

    Returns:
      The accumulated phase in range [0, 2*pi], shape [batch, time, ...].
    """
    # Get tensor shapes.
    n_batch = angular_frequency.shape[0]
    n_time = angular_frequency.shape[1]
    n_dims = len(angular_frequency.shape)
    n_ch_dims = n_dims - 2

    # Pad if needed.
    remainder = n_time % chunk_size
    if remainder:
        pad = chunk_size - remainder
        angular_frequency = pad_axis(angular_frequency, [0, pad], axis=1)

    # Split input into chunks.
    length = angular_frequency.shape[1]
    n_chunks = int(length / chunk_size)
    chunks = torch.reshape(angular_frequency, [n_batch, n_chunks, chunk_size] + [-1] * n_ch_dims)
    phase = torch.cumsum(chunks, dim=2)

    # Add offsets.
    # Offset of the next row is the last entry of the previous row.
    offsets = phase[:, :, -1:, ...] % (2.0 * np.pi)
    offsets = pad_axis(offsets, [1, 0], axis=1)
    offsets = offsets[:, :-1, ...]

    # Offset is cumulative among the rows.
    offsets = torch.cumsum(offsets, dim=1) % (2.0 * np.pi)
    phase = phase + offsets

    # Put back in original shape.
    phase = phase % (2.0 * np.pi)
    phase = torch.reshape(phase, [n_batch, length] + [-1] * n_ch_dims)

    # Remove padding if added it.
    if remainder:
        phase = phase[:, :n_time]
    return phase


def pad_axis(x, padding=(0, 0), axis=0, **pad_kwargs):
    """Pads only one axis of a tensor.
    Args:
      x: Input tensor.
      padding: Tuple of number of samples to pad (before, after).
      axis: Which axis to pad.
      **pad_kwargs: Other kwargs to pass to tf.pad.
    Returns:
      A tensor padded with padding along axis.
    """
    n_end_dims = len(x.shape) - axis - 1
    n_end_dims *= n_end_dims > 0
    paddings = [0, 0] * n_end_dims + list(padding) + [0, 0] * axis
    return torch.nn.functional.pad(x, paddings, **pad_kwargs)


def frequency_filter(
    audio: torch.Tensor,
    magnitudes: torch.Tensor,
    window_size: int = 0,
    padding: str = "same",
    cross_fade: bool = False,
) -> torch.Tensor:
    """Filter audio with a finite impulse response filter.

    Args:
      audio: Input audio. Tensor of shape [batch, audio_timesteps].
      magnitudes: Frequency transfer curve. Float32 Tensor of shape [batch,
        n_frames, n_frequencies] or [batch, n_frequencies]. The frequencies of the
        last dimension are ordered as [0, f_nyqist / (n_frequencies -1), ...,
        f_nyquist], where f_nyquist is (sample_rate / 2). Automatically splits the
        audio into equally sized frames to match frames in magnitudes.
      window_size: Size of the window to apply in the time domain. If window_size
        is less than 1, it is set as the default (n_frequencies).
      padding: Either 'valid' or 'same'. For 'same' the final output to be the
        same size as the input audio (audio_timesteps). For 'valid' the audio is
        extended to include the tail of the impulse response (audio_timesteps +
        window_size - 1).

    Returns:
      Filtered audio. Tensor of shape
          [batch, audio_timesteps + window_size - 1] ('valid' padding) or shape
          [batch, audio_timesteps] ('same' padding).
    """
    impulse_response = frequency_impulse_response(magnitudes, window_size=window_size)
    return fft_convolve(audio, impulse_response, padding=padding, cross_fade=cross_fade)


def frequency_impulse_response(magnitudes: torch.Tensor, window_size: int = 0) -> torch.Tensor:
    """Get windowed impulse responses using the frequency sampling method.
    Follows the approach in:
    https://ccrma.stanford.edu/~jos/sasp/Windowing_Desired_Impulse_Response.html

    Args:
      magnitudes: Frequency transfer curve. Float32 Tensor of shape [batch,
        n_frames, n_frequencies] or [batch, n_frequencies]. The frequencies of the
        last dimension are ordered as [0, f_nyqist / (n_frequencies -1), ...,
        f_nyquist], where f_nyquist is (sample_rate / 2). Automatically splits the
        audio into equally sized frames to match frames in magnitudes.
      window_size: Size of the window to apply in the time domain. If window_size
        is less than 1, it defaults to the impulse_response size.

    Returns:
      impulse_response: Time-domain FIR filter of shape
        [batch, frames, window_size] or [batch, window_size].

    Raises:
      ValueError: If window size is larger than fft size.
    """
    # Get the IR (zero-phase form).
    # a real, even signal has a real, even Fourier transform (even => h(n)=h(-n))
    # since we define the one-sided magnitude response and assume the phase to be zero,
    # the result of the iDFT is an even impulse response, i.e. zero-phase form.
    # magnitudes = magnitudes.unsqueeze(-1)  # add last dimension for real and imaginary parts
    # magnitudes = torch.cat([magnitudes, torch.zeros_like(magnitudes)], dim=-1)
    impulse_response = torch.fft.irfft(magnitudes, dim=-1)

    # Window and put in causal form.
    impulse_response = apply_window_to_impulse_response(impulse_response, window_size)

    return impulse_response


def apply_window_to_impulse_response(
    impulse_response: torch.Tensor, window_size: int = 0, causal: bool = False
) -> torch.Tensor:
    """Apply a window to an impulse response and put in causal form.

    Args:
      impulse_response: A series of impulse responses frames to window, of shape
        [batch, n_frames, ir_size].
      window_size: Size of the window to apply in the time domain. If window_size
        is less than 1, it defaults to the impulse_response size.
      causal: Impulse response input is in causal form (peak in the middle).

    Returns:
      impulse_response: Windowed impulse response in causal form, with last
        dimension cropped to window_size if window_size is greater than 0 and less
        than ir_size.
    """

    restore_batch_dim = False
    if impulse_response.ndim == 2:
        impulse_response = impulse_response[:, None, :]
        restore_batch_dim = True

    impulse_response = impulse_response.type(torch.float32)

    # If IR is in causal form, put it in zero-phase form.
    if causal:
        impulse_response = torch.roll(
            impulse_response, shifts=(impulse_response.shape[-1]) // 2, dims=-1
        )

    # Get a window for better time/frequency resolution than rectangular.
    # Window defaults to IR size, cannot be bigger.
    ir_size = int(impulse_response.shape[-1])
    if (window_size <= 0) or (window_size > ir_size):
        window_size = ir_size
    window = torch.hann_window(window_size, device=impulse_response.device)

    # Zero pad the window and put in in zero-phase form.
    padding = ir_size - window_size
    if padding > 0:
        half_idx = (window_size + 1) // 2
        window = torch.cat([window[half_idx:], torch.zeros([padding]), window[:half_idx]], dim=0)
    else:
        window = torch.roll(window, shifts=(len(window)) // 2, dims=-1)

    # Apply the window, to get new IR (both in zero-phase form).
    impulse_response = window[None, None, :] * impulse_response

    # Put IR in causal form and trim zero padding.
    if padding > 0:
        first_half_start = (ir_size - (half_idx - 1)) + 1
        second_half_end = half_idx + 1
        impulse_response = torch.cat(
            [impulse_response[..., first_half_start:], impulse_response[..., :second_half_end]],
            dim=-1,
        )
    else:
        # equivalent to impulse_response = tf.signal.fftshift(impulse_response, axes=-1)
        impulse_response = torch.roll(
            impulse_response, shifts=(impulse_response.shape[-1]) // 2, dims=-1
        )

    if restore_batch_dim:
        impulse_response = impulse_response[:, 0, :]
    return impulse_response


def get_fft_size(frame_size: int, ir_size: int, power_of_2: bool = True) -> int:
    """Calculate final size for efficient FFT.
    Args:
      frame_size: Size of the audio frame.
      ir_size: Size of the convolving impulse response.
      power_of_2: Constrain to be a power of 2. If False, allow other 5-smooth
        numbers. TPU requires power of 2, while GPU is more flexible.
    Returns:
      fft_size: Size for efficient FFT.
    """
    convolved_frame_size = ir_size + frame_size - 1
    if power_of_2:
        # Next power of 2.
        fft_size = int(2 ** np.ceil(np.log2(convolved_frame_size)))
    else:
        fft_size = int(fftpack.helper.next_fast_len(convolved_frame_size))
    return fft_size


def fft_convolve(
    audio: torch.Tensor,
    impulse_response: torch.Tensor,
    padding: str = "same",
    delay_compensation: int = -1,
    cross_fade: bool = False,
) -> torch.Tensor:
    """Filter audio with frames of time-varying impulse responses.
    Time-varying filter. Given audio [batch, n_samples], and a series of impulse
    responses [batch, n_frames, n_impulse_response], splits the audio into frames,
    applies filters, and then overlap-and-adds audio back together.
    Applies non-windowed non-overlapping STFT/ISTFT to efficiently compute
    convolution for large impulse response sizes.

    Args:
      audio: Input audio. Tensor of shape [batch, audio_timesteps].
      impulse_response: Finite impulse response to convolve. Can either be a 2-D
        Tensor of shape [batch, ir_size], or a 3-D Tensor of shape [batch,
        ir_frames, ir_size]. A 2-D tensor will apply a single linear
        time-invariant filter to the audio. A 3-D Tensor will apply a linear
        time-varying filter. Automatically chops the audio into equally shaped
        blocks to match ir_frames.
      padding: Either 'valid' or 'same'. For 'same' the final output to be the
        same size as the input audio (audio_timesteps). For 'valid' the audio is
        extended to include the tail of the impulse response (audio_timesteps +
        ir_timesteps - 1).
      delay_compensation: Samples to crop from start of output audio to compensate
        for group delay of the impulse response. If delay_compensation is less
        than 0 it defaults to automatically calculating a constant group delay of
        the windowed linear phase filter from frequency_impulse_response().

    Returns:
      audio_out: Convolved audio. Tensor of shape
          [batch, audio_timesteps + ir_timesteps - 1] ('valid' padding) or shape
          [batch, audio_timesteps] ('same' padding).

    Raises:
      ValueError: If audio and impulse response have different batch size.
      ValueError: If audio cannot be split into evenly spaced frames. (i.e. the
        number of impulse response frames is on the order of the audio size and
        not a multiple of the audio size.)
    """
    audio, impulse_response = audio.type(torch.float32), impulse_response.type(torch.float32)

    # Add a frame dimension to impulse response if it doesn't have one.
    ir_shape = impulse_response.shape
    if len(ir_shape) == 2:
        impulse_response = impulse_response[:, None, :]
        ir_shape = impulse_response.shape

    # Get shapes of audio and impulse response.
    batch_size_ir, n_ir_frames, ir_size = ir_shape
    batch_size, audio_size = audio.shape

    # Validate that batch sizes match.
    if batch_size != batch_size_ir:
        raise ValueError(
            "Batch size of audio ({}) and impulse response ({}) must "
            "be the same.".format(batch_size, batch_size_ir)
        )

    # Cut audio into frames.
    frame_size = int(np.ceil(audio_size / n_ir_frames))
    audio_frames = torch.split(
        audio, frame_size, dim=-1
    )  # tuple of tensors, last one might be shorter
    audio_frames = list(audio_frames)

    # zero-pad last frame if necessary
    last_frame_size = audio_frames[-1].shape[-1]
    if last_frame_size < frame_size:
        pad_length = frame_size - last_frame_size
        last_frame_padded = torch.nn.functional.pad(audio_frames[-1], pad=(0, pad_length))
        audio_frames = audio_frames[:-1] + [last_frame_padded]

    audio_frames = torch.stack(audio_frames, dim=1)  # [batch_size, n_frames, frame_size]

    # Check that number of frames match.
    n_audio_frames = int(audio_frames.shape[1])
    if n_audio_frames != n_ir_frames:
        raise ValueError(
            "Number of Audio frames ({}) and impulse response frames ({}) do not "
            "match. For small hop size = ceil(audio_size / n_ir_frames), "
            "number of impulse response frames must be a multiple of the audio "
            "size.".format(n_audio_frames, n_ir_frames)
        )

    # Pad and FFT the audio and impulse responses.
    fft_size = get_fft_size(frame_size, ir_size, power_of_2=True)
    pad_length_audio = fft_size - frame_size
    audio_frames = torch.nn.functional.pad(audio_frames, pad=(0, pad_length_audio))
    pad_length_ir = fft_size - ir_size
    impulse_response = torch.nn.functional.pad(impulse_response, pad=(0, pad_length_ir))

    # audio_fft = torch.view_as_complex(torch.rfft(audio_frames, signal_ndim=1))
    # ir_fft = torch.view_as_complex(torch.rfft(impulse_response, signal_ndim=1))

    audio_fft = torch.fft.rfft(audio_frames)
    ir_fft = torch.fft.rfft(impulse_response)

    if cross_fade:
        audio_frames_out = cross_fade_time_varying_fir(
            audio_fft, ir_fft, frame_size, ir_size, fft_size
        )
    else:
        # Multiply the FFTs (same as convolution in time).
        audio_ir_fft = audio_fft * ir_fft

        # Take the IFFT to resynthesize audio.
        # audio_ir_fft = torch.view_as_real(audio_ir_fft)
        # audio_frames_out = torch.fft.irfft(audio_ir_fft, n=audio_frames.shape[-1])
        audio_frames_out = torch.fft.irfft(audio_ir_fft, n=audio_frames.shape[-1])

    audio_frames_out = audio_frames_out.transpose(1, 2)  # [batch_size, frame_size, n_frames]

    audio_out_size = (
        n_ir_frames - 1
    ) * frame_size + fft_size  # time domain length after frame-wise convolution

    # same as audio_out = tf.signal.overlap_and_add(audio_frames_out, hop_size)
    audio_out = torch.nn.functional.fold(
        audio_frames_out,
        output_size=(1, audio_out_size),
        kernel_size=(1, fft_size),
        stride=(1, frame_size),
    )
    audio_out = audio_out[:, 0, 0, :]

    # Crop and shift the output audio.
    return crop_and_compensate_delay(audio_out, audio_size, ir_size, padding, delay_compensation)


def cross_fade_time_varying_fir(
    audio_fft, ir_fft, audio_frame_size: int, ir_size: int, fft_size: int
):
    """
    Convolves a signal blockwise with a time-varying FIR filter in the frequency domain
    and cross-fades the transitions between IR frames to reduce audible clicking artefacts.
    Args:
        audio_fft: FFT of audio blocks. torch.Tensor [batch_size, n_frames, fft_size].
        ir_fft:  FFT of IRs. torch.Tensor [batch_size, n_frames, fft_size].
        audio_frame_size: length of the audio frames (before zero-padding to fft_size).
        ir_size: length of the IRs (before zero-padding to fft_size).
        fft_size: size of the FFT applied to the audio frames and the IRs.

    Returns:

    """

    # each frame is convolved with its IR
    audio_ir_fft = audio_fft * ir_fft

    # each frame is convolved with the IR of the previous frame
    audio_ir_fft_prev = audio_fft * torch.roll(ir_fft, shifts=1, dims=1)

    # Take the IFFT to resynthesize audio.
    audio_ir_fft = torch.view_as_real(audio_ir_fft)
    audio_frames_out = torch.fft.irfft(audio_ir_fft, dim=1, n=(fft_size,))

    audio_ir_fft_prev = torch.view_as_real(audio_ir_fft_prev)
    audio_frames_out_prev = torch.fft.irfft(audio_ir_fft_prev, dim=1, n=(fft_size,))

    convolved_frame_size = audio_frame_size + ir_size - 1  # signal length after convolution

    overlap = (
        ir_size - 1
    )  # overlap of blocks for overlap add. In this part the fade-in and -out will be applied

    pi = 3.141592653589793

    # create fade-in and fade-out functions. First blocks don't get a fade.
    # fade_in = sin^2(pi * n / (2 * overlap))
    # fade_out = cos^2(pi * n / (2 * overlap))
    # fade_in + fade_out = 1
    fade_in = torch.ones_like(audio_frames_out)
    fade_in[:, 1:, :overlap] = (
        torch.sin(pi * torch.linspace(0, overlap, overlap) / (2 * overlap)) ** 2
    )
    fade_out = torch.zeros_like(audio_frames_out_prev)
    fade_out[:, 1:, :overlap] = (
        torch.cos(pi * torch.linspace(0, overlap, overlap) / (2 * overlap)) ** 2
    )
    # apply fading
    audio_frames_out = audio_frames_out * fade_in
    audio_frames_out_prev = audio_frames_out_prev * fade_out

    audio_frames_out = audio_frames_out + audio_frames_out_prev

    return audio_frames_out


def crop_and_compensate_delay(
    audio: torch.Tensor, audio_size: int, ir_size: int, padding: str, delay_compensation: int
) -> torch.Tensor:
    """Crop audio output from convolution to compensate for group delay.

    Args:
      audio: Audio after convolution. Tensor of shape [batch, time_steps].
      audio_size: Initial size of the audio before convolution.
      ir_size: Size of the convolving impulse response.
      padding: Either 'valid' or 'same'. For 'same' the final output to be the
        same size as the input audio (audio_timesteps). For 'valid' the audio is
        extended to include the tail of the impulse response (audio_timesteps +
        ir_timesteps - 1).
      delay_compensation: Samples to crop from start of output audio to compensate
        for group delay of the impulse response. If delay_compensation < 0 it
        defaults to automatically calculating a constant group delay of the
        windowed linear phase filter from frequency_impulse_response().

    Returns:
      Tensor of cropped and shifted audio.

    Raises:
      ValueError: If padding is not either 'valid' or 'same'.
    """
    # Crop the output.
    if padding == "valid":
        crop_size = ir_size + audio_size - 1
    elif padding == "same":
        crop_size = audio_size
    else:
        raise ValueError("Padding must be 'valid' or 'same', instead " "of {}.".format(padding))

    # Compensate for the group delay of the filter by trimming the front.
    # For an impulse response produced by frequency_impulse_response(),
    # the group delay is constant because the filter is linear phase.
    total_size = int(audio.shape[-1])
    crop = total_size - crop_size
    start = (ir_size - 1) // 2 - 1 if delay_compensation < 0 else delay_compensation
    end = crop - start
    return audio[:, start:-end]


# extension to the DDSP library
def slope_frequency_response(decay_per_octave_db, n_freqs, f_ref):
    """

    Args:
        decay_per_octave_db: torch.Tensor [batch_size, n_frames, 1] the slope of the
            frequency response as reduction per octave in dB
        n_freqs: torch.Tensor int, number of frequency bands to consider between 0 and 8000 Hz
        f_ref: torch.Tensor [1,]. reference frequency to compute de decay per octave from.

    Returns:
        amplitude_factors: torch.Tensor [batch_size, n_frames, n_frequencies].
            frequency dependent factors for pressure signal so that -X db/octave
            from f_ref is obtained.
    """
    # # Check if decay_per_octave_db has 3 dims and add it if not
    # if len(decay_per_octave_db.shape) == 2:
    #     decay_per_octave_db = decay_per_octave_db.unsqueeze(2)

    freqs = torch.linspace(0, 8000, n_freqs).to(decay_per_octave_db)[None, None, :]
    freqs[:, :, 0] += torch.tensor(1e-7).to(freqs)  # avoid log2(0) below

    # factor for pressure signal that results in -'decay' dB SPL
    # (- 'decay' = 20 * log10(a_0))
    a_0 = 10 ** (-decay_per_octave_db / 20)

    # frequencies below f_ref have factor 1
    # other frequencies get a factor such that -decay_per_octave_db db/octave from f_ref is obtained
    amplitude_factors = torch.where(
        freqs > f_ref, a_0 ** torch.log2(freqs / f_ref), torch.tensor(1.0).to(decay_per_octave_db)
    )

    return amplitude_factors
