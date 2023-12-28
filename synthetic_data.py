import pytorch_lightning as pl
import torch
import numpy as np
import os

from torch.utils.data import DataLoader, random_split
from synths import Sinusoidal


class SimpleSinusoidDataset(pl.LightningDataModule):
    """Generate dataset of sinusoidal functions with random frequencies and amplitudes."""

    def __init__(
        self,
        freq_gen_min,
        freq_gen_max,
        n_samples,
        sample_rate=16000,
        amplitude_min=1,
        amplitude_max=1,
        size=1000,
        batch_size=8,
        batch_size_val=8,
        n_sinusoids=1,
        eval_split=0.2,
        test_split=None,
        n_sinusoids_min=None,
        mask_rand_amplitudes=False,
        harmonic=False,
        num_workers=1,
    ):
        super().__init__()
        self.freq_gen_min = freq_gen_min
        self.freq_gen_max = freq_gen_max
        self.sample_rate = sample_rate
        self.amplitude_min = amplitude_min
        self.amplitude_max = amplitude_max
        self.n_sinusoids = n_sinusoids
        self.n_sinusoids_min = n_sinusoids_min
        self.mask_rand_amplitudes = mask_rand_amplitudes
        self.harmonic = harmonic
        self.size = size
        self.batch_size = batch_size
        self.batch_size_val = batch_size_val
        self.num_workers = num_workers
        self.n_samples = n_samples
        self.n_fake_frames = 16  # length/hop_size, manually set for now
        self.eval_split = eval_split
        self.test_split = test_split

        assert (
            freq_gen_max < sample_rate / 2
        ), "freq_gen_max must be less than sample_rate / 2"
        if self.harmonic:
            # self.synth = Harmonic(n_samples, sample_rate=self.sample_rate, 
            # scale_fn_amplitudes=None, scale_fn_distribution=None,
            #      normalize_below_nyquist=True)
            self.synth = Sinusoidal(
                n_samples,
                sample_rate=self.sample_rate,
                amp_scale_fn=None,
                freq_scale_fn=None,
                harmonic=True,
            )
        else:
            self.synth = Sinusoidal(
                n_samples,
                sample_rate=self.sample_rate,
                amp_scale_fn=None,
                freq_scale_fn=None,
            )

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        n_freqs = 1 if self.harmonic else self.n_sinusoids
        freqs = (
            torch.rand(self.size, n_freqs) * (self.freq_gen_max - self.freq_gen_min)
            + self.freq_gen_min
        )
        amplitudes = (
            torch.rand(self.size, self.n_sinusoids)
            * (self.amplitude_max - self.amplitude_min)
            + self.amplitude_min
        )

        if self.n_sinusoids_min is not None:
            # Generate a tensor with random number of active modes for each element
            n_active_modes = torch.randint(
                low=self.n_sinusoids_min - 1, high=self.n_sinusoids, size=(self.size,)
            )

            if (
                self.mask_rand_amplitudes
            ):  # This randomly selects which n_active_modes will be masked
                # Create a mask for each element in your weights tensor, where first mode is always
                # active
                #  Eg if n_active_modes = 3, then the mask (without the first mode, which iw 1) can 
                # be [1, 0, 1, 0, 0, 1, 0, 0, ...]

                mask = torch.zeros(self.size, self.n_sinusoids - 1).bool()
                for i in range(self.size):
                    mask[
                        i, torch.randperm(self.n_sinusoids - 1)[: n_active_modes[i]]
                    ] = 1

            else:  # This will mask all amplitudes after n_active_modes (sequentially)
                # Eg if n_active_modes = 3, then the mask will be [1, 1, 1, 0, 0, 0, 0, 0, ...]
                mask = torch.arange(1, self.n_sinusoids).expand(
                    self.size, self.n_sinusoids - 1
                ) < n_active_modes.unsqueeze(1)

            mask = torch.cat((torch.ones(self.size, 1).bool(), mask), dim=1)

            # Apply the mask to your weights tensor (this will set the non-active weights to zero)
            amplitudes = amplitudes * mask.float()

        sinusoids = self.generate_sinusoids(freqs, amplitudes)
        self.data = TensorDictDataset(
            sinusoids, thetas={"frequency": freqs, "weights": amplitudes}
        )
        if self.test_split is not None:
            self.train, self.val, self.test = random_split(
                self.data,
                [
                    int((1 - self.eval_split - self.test_split) * self.size),
                    int(self.eval_split * self.size),
                    int(self.test_split * self.size),
                ],
            )
        else:
            self.train, self.val = random_split(
                self.data,
                [
                    int((1 - self.eval_split) * self.size),
                    int(self.eval_split * self.size),
                ],
            )

    def train_dataloader(self):
        return DataLoader(
            self.train, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val, batch_size=self.batch_size_val, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return [
            DataLoader(
                self.test, batch_size=self.batch_size_val, num_workers=self.num_workers
            )
        ]

    def save_dataset(self, save_path):
        save_dict = {
            'train_tensors': self.train.dataset.tensors,
            'train_thetas': self.train.dataset.thetas,
            'val_tensors': self.val.dataset.tensors,
            'val_thetas': self.val.dataset.thetas
        }
        
        if self.test_split is not None:
            save_dict.update({
                'test_tensors': self.test.dataset.tensors,
                'test_thetas': self.test.dataset.thetas
            })
            
        torch.save(save_dict, save_path)

    def generate_sinusoids(self, fundamental_freqs, weights):
        """
        Args:
            fundamental_freqs: A tensor of shape [batch, n_sinusoids] of fundamental frequencies in 
            Hz or [batch, 1] if harmonic=True
            amplitudes: A tensor of shape [batch, n_sinusoids] of amplitudes.
        """

        n_frames = (
            self.n_fake_frames
        )  # Verify this is correct, I think it does not matter since n_samples is fixed on the
          #synth
        if self.harmonic:
            harmonic_distribution = weights.unsqueeze(1).repeat(1, n_frames, 1)
            f0_hz = fundamental_freqs.unsqueeze(1).repeat(1, n_frames, 1)
            signal = self.synth(harmonic_distribution, f0_hz)
            # signal = signal / (torch.max(torch.abs(signal), dim=-1, keepdim=True)[0] + 1e-7)
            return signal

        else:
            # make the amplitudes sum to 1
            # weights = weights / torch.sum(weights, dim=-1, keepdim=True)
            amplitudes = weights.unsqueeze(1).repeat(1, n_frames, 1)
            # sum to 1
            amplitudes = amplitudes / torch.sum(amplitudes, dim=-1, keepdim=True)

            f0_hz = fundamental_freqs.unsqueeze(1).repeat(1, n_frames, 1)
            return self.synth(amplitudes, f0_hz)


class TensorDictDataset(torch.utils.data.Dataset):
    def __init__(self, tensors, thetas=None, normalize=True):
        # Conver to torch if numpy or list
        if isinstance(tensors, np.ndarray):
            tensors = torch.from_numpy(tensors)
        elif isinstance(tensors, list):
            for i, t in enumerate(tensors):
                if isinstance(t, np.ndarray):
                    tensors[i] = torch.from_numpy(t)
            tensors = torch.stack(tensors)

        # Check if elements of theta are numpy, convert to torch 32 if so
        if thetas is not None:
            for k, v in thetas.items():
                if isinstance(v, np.ndarray):
                    thetas[k] = torch.from_numpy(v)
                elif isinstance(v, list):
                    thetas[k] = torch.stack([torch.from_numpy(t) for t in v])
                thetas[k] = thetas[k].float()

        # Convert to 32
        if tensors.dtype == torch.float64:
            tensors = tensors.float()

        self.tensors = tensors
        self.thetas = thetas
        self.normalize = normalize

    def __getitem__(self, index):
        tensor = self.tensors[index]
        # Normalize to be between 0.9 and 0.9
        if self.normalize:
            tensor = tensor / (tensor.abs().max() + 1e-7)
            tensor = tensor * 0.9

        outputs = {"x": tensor}
        if self.thetas is not None:
            outputs.update({k: v[index] for k, v in self.thetas.items()})
        return outputs

    def __len__(self):
        return len(self.tensors)


class PreloadedSinusoidDataset(torch.utils.data.Dataset):
    """Load dataset from a saved file."""

    def __init__(self, 
                 data,
                 normalize=True,):
        self.data = data
        self.normalize = normalize

    def __getitem__(self, index):
        signal = self.data[index]['x']
        # Normalize signal to be between 0.9 and 0.9
        if self.normalize:
            signal = signal / (signal.abs().max() + 1e-7)

        outputs = {"x": signal}
        outputs.update({k: v[index] for k, v in self.data[index].items() if k != 'x'})
        return outputs

    def __len__(self):
        return len(self.data)


class PreloadedSinusoidDataModule(pl.LightningDataModule):
    """DataModule for preloaded sinusoid dataset."""

    def __init__(self, data_path, batch_size=8, batch_size_val=8, num_workers=1):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.batch_size_val = batch_size_val
        self.num_workers = num_workers

    def setup(self, stage=None):
        data_dict = torch.load(self.data_path)
        
        self.train = TensorDictDataset(data_dict['train_tensors'], data_dict['train_thetas'])
        self.val = TensorDictDataset(data_dict['val_tensors'], data_dict['val_thetas'])
        
        if 'test_tensors' in data_dict:
            self.test = TensorDictDataset(data_dict['test_tensors'], data_dict['test_thetas'])
        else:
            self.test = None

    def train_dataloader(self):
        return DataLoader(
            self.train, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val, batch_size=self.batch_size_val, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return (
            DataLoader(
                self.test, batch_size=self.batch_size_val, num_workers=self.num_workers
            )
            if self.test
            else None
        )


# Usage example:

if __name__ == "__main__":

    # n_samples: 4096  # Verify why it bugs with 8192 and not 8193
    # # sample_rate: 8000
    # freq_gen_min: 40
    # freq_gen_max: 1950
    # amplitude_min: 0.4
    # amplitude_max: 1
    # size: 5000
    # # n_samples: 256 # defined by spectrum_size
    # n_sinusoids: 8
    # n_sinusoids_min: 1
    # harmonic: true
    # eval_split: 0.2
    # test_split: 0.1

    # Create and setup the dataset
    dataset = SimpleSinusoidDataset(
        freq_gen_min=40,
        freq_gen_max=1950,
        n_samples=4096,
        amplitude_min=0.4,
        amplitude_max=1,
        size=4000,
        batch_size=8,
        batch_size_val=8,
        n_sinusoids=8,
        eval_split=0.2,
        test_split=0.1,
        n_sinusoids_min=1,
        harmonic=True,
    )

    dataset.setup()

    # Save the dataset
    dataset.save_dataset("saved_dataset.pth")
    print("Dataset saved.")

    # Load the dataset
    loaded_data = torch.load("saved_dataset.pth")
    dataset = PreloadedSinusoidDataset(loaded_data)
    print("Dataset loaded.")
