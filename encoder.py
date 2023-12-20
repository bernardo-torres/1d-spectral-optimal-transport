# PESTO architecture comes from https://github.com/SonyCSLParis/pesto 

import torch
import torch.nn as nn
from functools import partial
import numpy as np


class ExpSigmoid(torch.nn.Module):
    def __init__(self, exponent=10.0, max_value=2.0, threshold=1e-7):
        super().__init__()
        self.exponent = exponent
        self.max_value = max_value
        self.threshold = threshold

    def forward(self, x):
        x = x.type(torch.float32)
        exponent = torch.tensor(self.exponent, dtype=torch.float32, device=x.device)
        return self.max_value * torch.sigmoid(x) ** torch.log(exponent) + self.threshold


class ToeplitzLinear(nn.Conv1d):
    def __init__(self, in_features, out_features):
        super(ToeplitzLinear, self).__init__(
            in_channels=1,
            out_channels=1,
            kernel_size=in_features + out_features - 1,
            padding=out_features - 1,
            bias=False,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super(ToeplitzLinear, self).forward(input.unsqueeze(-2)).squeeze(-2)


def fc(ch_in=256, ch_out=256):
    layers = [
        torch.nn.Linear(ch_in, ch_out),
        torch.nn.LayerNorm(ch_out),  # normalization is done over the last dimension
        torch.nn.LeakyReLU(),
    ]
    return torch.nn.Sequential(*layers)


# DDSP
def fc_stack(ch_in=256, ch_out=256, layers=2):
    return torch.nn.Sequential(
        *([fc(ch_in, ch_out)] + [fc(ch_out, ch_out) for i in range(layers - 1)])
    )


def get_padding(input_size, output_size, kernel_size, stride=1, dilation=1):
    """
    Calculate the padding needed for a 1D convolution to maintain a specific output size.

    Parameters:
    - input_size (int): The size of the input.
    - output_size (int): The desired size of the output.
    - kernel_size (int): The size of the kernel.
    - stride (int): The stride of the convolution. Default is 1.
    - dilation (int): The dilation of the kernel. Default is 1.

    Returns:
    - int: The padding needed to maintain the output size.
    """

    padding = np.floor(
        (stride * (output_size - 1) - input_size + dilation * (kernel_size - 1) + 1) / 2
    )
    return int(padding)


class PESTOEncoder(nn.Module):
    """
    Basic CNN similar to the one in Johannes Zeitler's report,
    but for longer HCQT input (always stride 1 in time)
    Still with 75 (-1) context frames, i.e. 37 frames padded to each side
    The number of input channels, channels in the hidden layers, and output
    dimensions (e.g. for pitch output) can be parameterized.
    Layer normalization is only performed over frequency and channel dimensions,
    not over time (in order to work with variable length input).
    Outputs one channel with sigmoid activation.

    Args (Defaults: BasicCNN by Johannes Zeitler but with 6 input channels):
        n_chan_input:     Number of input channels (harmonics in HCQT)
        n_chan_layers:    Number of channels in the hidden layers (list)
        n_prefilt_layers: Number of repetitions of the prefiltering layer
        residual:         If True, use residual connections for prefiltering (default: False)
        n_bins_in:        Number of input bins (12 * number of octaves)
        n_bins_out:       Number of output bins (12 for pitch class, 72 for pitch, num_octaves * 12)
        a_lrelu:          alpha parameter (slope) of LeakyReLU activation function
        p_dropout:        Dropout probability
    """

    def __init__(
        self,
        n_modes=1,
        estimation_type="soft-argmax",
        output_splits=["frequency"],
        harmonic=False,
        feature_size=512,
        output_size=None,
        n_chan_input=1,
        n_chan_layers=(40, 30, 30, 10, 3),
        n_prefilt_layers=2,
        residual=True,
        n_bins_in=512,
        activation_fn="leaky",
        num_output_layers=1,
        a_lrelu=0.3,
        kernel_size=15,
        **kwargs
    ):
        # SinusoidalEncoder parameters
        self.n_modes = n_modes
        self.estimation_type = estimation_type
        self.output_splits = output_splits
        self.harmonic = harmonic
        self.feature_size = feature_size

        # Output size logic
        self.out_size = output_size if output_size is not None else feature_size

        # PESTOEncoder parameters
        self.n_chan_input = n_chan_input
        n_in = n_chan_input
        self.n_chan_layers = n_chan_layers
        self.n_prefilt_layers = n_prefilt_layers
        self.residual = residual
        self.n_bins_in = n_bins_in
        self.activation_fn = activation_fn
        self.num_output_layers = num_output_layers
        self.a_lrelu = a_lrelu
        self.kernel_size = kernel_size
        if len(self.n_chan_layers) < 5:
            self.n_chan_layers.append(1)
        pre_fc_dim = n_bins_in * self.n_chan_layers[4]
        self.feature_size = pre_fc_dim
        super().__init__(**kwargs)

        self.linear = nn.ModuleDict()
        self.activations = nn.ModuleDict()

        if activation_fn == "relu":
            activation_layer = nn.ReLU
        elif activation_fn == "silu":
            activation_layer = nn.SiLU
        elif activation_fn == "leaky":
            activation_layer = partial(nn.LeakyReLU, negative_slope=a_lrelu)
        else:
            raise ValueError

        # Layer normalization over frequency and channels (harmonics of HCQT)
        self.layernorm = nn.LayerNorm(normalized_shape=[n_in, n_bins_in])

        pad = get_padding(n_bins_in, n_bins_in, kernel_size)

        # Prefiltering
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=n_in,
                out_channels=self.n_chan_layers[0],
                kernel_size=kernel_size,
                padding=pad,
                stride=1,
            ),
            activation_layer(),
        )
        self.n_prefilt_layers = n_prefilt_layers
        self.prefilt_list = nn.ModuleList()
        for p in range(1, n_prefilt_layers):
            self.prefilt_list.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=self.n_chan_layers[0],
                        out_channels=self.n_chan_layers[0],
                        kernel_size=kernel_size,
                        padding=pad,
                        stride=1,
                    ),
                    activation_layer(),
                )
            )
        self.residual = residual

        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.n_chan_layers[0],
                out_channels=self.n_chan_layers[1],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            activation_layer(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.n_chan_layers[1],
                out_channels=self.n_chan_layers[2],
                kernel_size=1,
                padding=0,
                stride=1,
            ),
            activation_layer(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.n_chan_layers[2],
                out_channels=self.n_chan_layers[3],
                kernel_size=1,
                padding=0,
                stride=1,
            ),
            activation_layer(),
            nn.Dropout(),
            nn.Conv1d(
                in_channels=self.n_chan_layers[3],
                out_channels=self.n_chan_layers[4],
                kernel_size=1,
                padding=0,
                stride=1,
            ),
        )

        self.flatten = nn.Flatten(start_dim=1)

        layers = []
        pre_fc_dim = n_bins_in * self.n_chan_layers[4]
        for i in range(num_output_layers - 1):
            layers.extend([ToeplitzLinear(pre_fc_dim, pre_fc_dim), activation_layer()])
        self.pre_fc = nn.Sequential(*layers)
        # self.fc = ToeplitzLinear(pre_fc_dim, output_dim)

        self.linear = nn.ModuleDict()
        self.activations = nn.ModuleDict()
        if "frequency" in self.output_splits:
            n_mean_outs = 1 if self.harmonic else self.n_modes
            self.linear["frequency"] = nn.ModuleList(
                [ToeplitzLinear(self.feature_size, self.out_size) for i in range(n_mean_outs)]
            )

            # self.linear["frequency"] = nn.ModuleList(
            #     [nn.Linear(self.feature_size, self.out_size) for i in range(n_mean_outs)]
            # )
            self.activations["frequency"] = nn.Identity()

        if "gain" in self.output_splits:
            self.linear["gain"] = nn.ModuleList([nn.Linear(self.feature_size, 1)])
            self.activations["gain"] = ExpSigmoid()

        if "weights" in self.output_splits:
            self.linear["weights"] = nn.ModuleList([nn.Linear(self.feature_size, self.n_modes)])
            self.activations["weights"] = nn.Sequential(ExpSigmoid())  # nn.Softmax(dim=-1))

        self.register_buffer("abs_shift", torch.zeros((), dtype=torch.long), persistent=True)

    def forward(self, x, **kwargs):
        r"""

        Args:
            x (torch.Tensor): shape (batch, channels, freq_bins)
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x_norm = self.layernorm(x)

        x = self.conv1(x_norm)
        for p in range(0, self.n_prefilt_layers - 1):
            prefilt_layer = self.prefilt_list[p]
            if self.residual:
                x_new = prefilt_layer(x)
                x = x_new + x
            else:
                x = prefilt_layer(x)
        conv2_lrelu = self.conv2(x)
        conv3_lrelu = self.conv3(conv2_lrelu)

        y_pred = self.conv4(conv3_lrelu)
        y_pred = self.flatten(y_pred)
        y_pred = self.pre_fc(y_pred)
        # y_pred = self.fc(y_pred)  # WARNING: issues when batch size = 1
        # return self.final_norm(y_pred)
        # outputs = self.end_forward(y_pred)

        outputs = {}
        # For each output split, apply linear layer to each mode
        for output_split in self.output_splits:
            # Stack outputs from each mode
            outputs[output_split] = torch.stack(
                [linear(y_pred.clone()) for linear in self.linear[output_split]],
                dim=1,
            )
            outputs[output_split] = outputs[output_split].squeeze(-1)
            outputs[output_split] = outputs[output_split].squeeze(
                1
            )  # Remove if going back to old weights
            # Apply activation function
            outputs[output_split] = self.activations[output_split](outputs[output_split])
            # (batch, n_modes, out_size)
        return outputs

    def predict_pitch(self, logits, temperature=1.0, mask=None):
        """Predict normalized pitch from logits. Range is [0, 1], which corresponds to
            [freq_unit_min, freq_unit_max]
        Args:
            logits (torch.Tensor): logits from network. Shape (batch, n_modes, out_size)
        Returns:
            torch.Tensor: predicted pitch (normalized to [freq_unit_min, freq_unit_max])

        Example:
            If freq_unit_min = 0 and freq_unit_max = 0.2, and the predicted pitch from the network
                is 0.8, then the returned frequency 0.16
        """

        if logits.ndim == 2:
            logits = logits.unsqueeze(1)
        batch_size, n_modes, seq_len = logits.shape

        outputs = {}
        # If network predicts bin probabilities, convert to expectation (do it for each mode)
        if self.estimation_type == "soft-argmax":
            if mask is not None:
                if mask.ndim == 2:
                    mask = mask.unsqueeze(1)
                logits = logits * mask
                logits = logits + 1e-7
            probabilities = torch.softmax(logits / temperature, dim=-1)
            positions = torch.linspace(0, 1, seq_len).to(logits.device)
            expectation = torch.sum(probabilities * positions, dim=-1)
            outputs.update({"pitch_unit": expectation, "probabilities": probabilities})

        # If network predicts expectation directly, return expectation
        elif self.estimation_type == "kernel-soft-argmax":
            # Apply gaussian kernel centered on discrete argmax of logits
            # with std = 0.1
            discrete_argmax = torch.argmax(logits, dim=-1)  # Shape batch, n_modes
            argmax_pos = discrete_argmax / (seq_len - 1)  # Get position in [0, 1]
            positions = torch.linspace(0, 1, seq_len).to(logits.device)
            std = self.kwargs.get("std", 0.025)
            gaussian_kernel = torch.exp(
                -((positions.unsqueeze(0) - argmax_pos.unsqueeze(-1)) ** 2) / (2 * std**2)
            )

            # Normalize kernel
            gaussian_kernel = gaussian_kernel / gaussian_kernel.sum(dim=-1, keepdim=True)

            # Apply kernel to logits
            probabilities = torch.softmax(gaussian_kernel * logits / temperature, dim=-1)
            expectation = torch.sum(probabilities * positions, dim=-1)
            outputs.update(
                {
                    "pitch_unit": expectation,
                    "probabilities": probabilities,
                    "kernel": gaussian_kernel,
                }
            )

        elif self.estimation_type == "regression":
            # Sigmoid to ensure values are in [0, 1]
            regressed = torch.sigmoid(logits).squeeze(-1)  # maybe switch to expsigmoid
            outputs.update({"pitch_unit": regressed})

        return outputs


