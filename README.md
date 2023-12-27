# Unsupervised Harmonic Parameter Estimation Using Differentiable DSP and Spectral Optimal Transport

[Paper (arXiv)](https://arxiv.org/abs/2312.14507)

This is the repository for our paper "[Unsupervised Harmonic Parameter Estimation Using Differentiable DSP and Spectral Optimal Transport.](https://arxiv.org/abs/2312.14507)" 

We introduce a loss function for comparing spectra *horizontally* inspired by optimal transport. It computes the one dimensional Wasserstein distance between the spectra of two signals, which gives a measure of energy displacement along the frequency axis. By computing the gradient of this loss function with respect to the parameters of a signal processor (such as an sinusoidal oscillator), we can improve frequency localization/estimation compared to traditional *vertical* spectral losses (such as the Multi-Scale Spectral loss).


### Summary
- **Spectral Optimal Transport**: We use a loss function inspired by optimal transport to compare the spectra of two signals
In the paper we test this loss function on an autoencoding task aimed at estimating the parameters of a harmonic synthesizer (fundamental frequency and amplitudes) and at obtaining good reconstruction.

The [loss function](https://github.com/bernardo-torres/1d-spectral-optimal-transport/blob/4ba751d4cc7b7427ce8a1e7e9ae8320d799deeff/losses.py#L89) was largely based on [POT](https://pythonot.github.io/)s implementation of the 1D Wasserstein distance.

- **Lightweight pitch estimator**: Our encoder uses a lightweight architecture based on [PESTO](https://github.com/SonyCSLParis/pesto/tree/master) to estimate the parameters of a harmonic synthesizer (fundamental frequency and amplitudes).
- **Differentiable DSP**: Our decoder is a harmonic synthesizer from [DDSP](https://openreview.net/pdf?id=B1x1ma4tDr) that synthesizes audio from fundamental frequency and amplitude parameters. Even though the decoder is not trained, we code it using pytorch and using AutoDiff we can compute the gradient of the loss function w.r.t. its input parameters (fundamental frequency and amplitudes).


## Running Experiments

We recommend installing the dependencies in a virtual environment. We provide the conda environment file [environment.yml](environment.yml) for easy installation.

```bash
conda env create -f environment.yml
conda activate sot
```

### Paper experiments

Checkpoints are available in the [checkpoints](checkpoints) folder. Each subfolder is an experiment as described in the paper and we provide checkpoints and configuration files for each of the 5 runs with different random seeds.

To run an experiment, run the following command:
```bash
python train.py --config checkpoints/<experiment_name>/config.yaml
```
To reproduce the result table from the paper, run the following command:
```bash
python eval_paper.py
```

## Data Description
The synthetic data used for training, evaluation, and testing is available at [data](data). You can use PreloadedSinusoidDataModule in [synthetic_data.py](synthetic_data.py) to load it easily.

## Citation
If you find our work useful or use it in your research, you can cite it using:

```bibtex
@article{torres2023unsupervised,
      title={Unsupervised Harmonic Parameter Estimation Using Differentiable DSP and Spectral Optimal Transport}, 
      author={Torres, Bernardo and Peeters, Geoffroy and Richard, Ga{\"e}l},
      journal={arXiv preprint arXiv:2312.14507},
      year={2023},
}

```
