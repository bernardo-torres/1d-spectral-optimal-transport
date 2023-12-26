# Unsupervised Harmonic Parameter Estimation Using Differentiable DSP and Spectral Optimal Transport

[Paper](https://arxiv.org/abs/2312.14507)

This is the repository for our paper titled "Unsupervised Harmonic Parameter Estimation Using Differentiable DSP and Spectral Optimal Transport." 

We introduce a loss function for comparing spectra \textit{horizontally} inspired by optimal transport. It computes the one dimensional Wasserstein distance between the spectra of two signals, which gives a measure of energy displacement along the frequency axis. By computing the gradient of this loss function with respect to the parameters of a signal processor (such as an sinusoidal oscillator), we can improve frequency localization/estimation compared to traditional \textit{vertical} spectral losses (such as the Multi-Scale spectral loss).


### Key Components
- **Lightweight estimator**: We use a lightweight encoder based on PESTO to estimate the parameters of a harmonic synthesizer (fundamental frequency, and amplitudes).
- **Differentiable DSP**: We use a harmonic synthesizer from DDSP, which allows the gradient-based optimization of DSP parameters.
- **Spectral Optimal Transport**: We use a loss function inspired by optimal transport to compare the spectra of two signals.


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
