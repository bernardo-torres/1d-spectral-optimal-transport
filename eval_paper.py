# Import necessary libraries
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
import os
from contextlib import redirect_stdout, redirect_stderr
import io
import random
import yaml

import pytorch_lightning as pl
from pytorch_lightning import Trainer as plTrainer
from synthetic_data import PreloadedSinusoidDataModule
from trainer import Trainer

# Initialize IO for capturing stdout and stderr
f_stdout = io.StringIO()
f_stderr = io.StringIO()

# Function to load checkpoints
def load_checkpoint_from_run_id(run_id: int, type):
    logs_path = "checkpoints"
    experiments = [f for f in os.listdir(logs_path) if os.path.isdir(os.path.join(logs_path, f))]
    run_paths = []
    for experiment in experiments:
        run_paths += [os.path.join(logs_path, experiment, f) for f in os.listdir(os.path.join(logs_path, experiment)) if os.path.isdir(os.path.join(logs_path, experiment, f))]

    run_path = [f for f in run_paths if str(run_id) in f]
    if len(run_path) == 0:
        raise Exception(f'No run_id {run_id} in {logs_path}')
    run_path = run_path[0]
    checkpoint_folder_path = os.path.join(run_path, "checkpoints")
    checkpoints = [f for f in os.listdir(checkpoint_folder_path) if type in f]
    if len(checkpoints) > 1:
        raise Exception(f'More than one checkpoint with type {type} in name')
    
    checkpoint = [f for f in checkpoints if type in f][0]
    checkpoint_path = os.path.join(checkpoint_folder_path, checkpoint)
    checkpoint = Trainer.load_from_checkpoint(checkpoint_path)
    
    train_config_path = os.path.join(run_path, "train_config.yaml")
    with open(train_config_path, 'r') as stream:
        try:
            train_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
        name = train_config['name']
        seed = train_config['seed_everything']
    
    return checkpoint.eval(), {'name': name, 'seed': seed}

# Function to rename keys in metrics
def rename_keys(metrics):
    metrics['LSD'] = round(metrics.pop('test_metrics/log_spectral_distance'), 2)
    metrics['MSE'] = round(metrics.pop('test_metrics/mse'), 2)
    metrics['MSS'] = round(metrics.pop('test_metrics/mss'), 2)
    metrics['OD'] = round(metrics.pop('test_metrics/octave_difference'), 2) * -1
    metrics['RPA'] = round(metrics.pop('test_metrics/raw_pitch_accuracy')*100, 2)
    metrics['RCA'] = round(metrics.pop('test_metrics/raw_chroma_accuracy')*100, 2)
    if 'test_metrics/diff_activated_partials' in metrics:
        metrics.pop('test_metrics/diff_activated_partials')
    return {k: v for k, v in metrics.items() if 'loss' not in k}

# Function to set seed
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(seed, workers=True)

# Set the seed
set_seed(234123)

# Setup data module
synthetic_datamodule = PreloadedSinusoidDataModule(data_path="data/40_1950_4096_04_1_4000_8_1_harmonic.pth")
synthetic_datamodule.setup()

# Function to evaluate synthetic data
@torch.inference_mode()
def eval_synthetic(model):
    trainer = plTrainer(gpus=1)
    metrics = trainer.test(model, datamodule=synthetic_datamodule)
    return rename_keys(metrics[0])

# Function to run runs
@torch.inference_mode()
def run_runs(run_ids, category):
    metrics = {}
    for run_id in tqdm(run_ids):
        print(f"Run ID: {run_id}")
        with redirect_stdout(f_stdout), redirect_stderr(f_stderr):
            checkpoint, metadata = load_checkpoint_from_run_id(run_id, category)
            name = '-'.join(metadata['name'].split('-')[0:-1])
            seed = metadata['seed']
            _metrics = eval_synthetic(checkpoint)
            name_seed = f"{name}_{seed}"
            metrics[name_seed] = _metrics
            metrics[name_seed]['name'] = name
            metrics[name_seed]['seed'] = seed
            metrics[name_seed]['run_id'] = run_id
        print(f"Ran {name_seed}")
    return metrics

# Define categories and file paths
category = 'best-lsd'
# Create results folder if it does not exist
if not os.path.exists('results'):
    os.makedirs('results')
results_csv = f"results/synthetic_results_{category}.csv"
paper_results_csv = f"results/synthetic_results_paper_{category}.csv"
all_runs = {}

# Baselines
baseline_run_ids = ['24k6gcvr', 'j34nzynm', 'p3gl8g41', 'mnd5a1id', 'zxc53gr9']
all_runs['MSS-Lin'] = baseline_run_ids
metrics = run_runs(baseline_run_ids, category)
df = pd.DataFrame.from_dict(metrics, orient='index')
df.to_csv(results_csv, mode='a', header=not os.path.exists(results_csv))

runs = ['vb2gupng', '8ilummup', 't7xiqbi7', 'snzfsaay', 'ld9m0re0']
all_runs['MSS-LogLin'] = runs
metrics = run_runs(runs, category)
df = pd.DataFrame.from_dict(metrics, orient='index')
df.to_csv(results_csv, mode='a', header=not os.path.exists(results_csv))

run_ids = ['w3pj33i5', 'rcfmhtj4', 'evcknw5r', 'f44i9icd', 'pketj6f1']
all_runs['SOT-512'] = run_ids
metrics = run_runs(run_ids, category)
df = pd.DataFrame.from_dict(metrics, orient='index')
df.to_csv(results_csv, mode='a', header=not os.path.exists(results_csv))

run_ids = ['djl02k78', 'dq3u8gfz', 'tlrh627e', 'eulq6s1k', 'axlgb4wk']
all_runs['SOT-512-LS'] = run_ids
metrics = run_runs(run_ids, category)
df = pd.DataFrame.from_dict(metrics, orient='index')
df.to_csv(results_csv, mode='a', header=not os.path.exists(results_csv))

run_ids = ['4utsx63v', 'up2qw81d', 'c6m8cytv', 'bmc2htff', 'any432ya']
all_runs['SOT-2048'] = run_ids
metrics = run_runs(run_ids, category)

df = pd.DataFrame.from_dict(metrics, orient='index')
# Save to csv, append if file exists
df.to_csv(results_csv, mode='a', header=not os.path.exists(results_csv))
# Organize results
df = pd.read_csv(results_csv, index_col=0)


run_ids = ['f35j6lll', '2nsxm6gi', 'mw4v97ls', '4u1cbkwf', 'bf4oik7u']
all_runs['SOT-NoCut'] = run_ids
metrics = run_runs(run_ids, category)
df = pd.DataFrame.from_dict(metrics, orient='index')
df.to_csv(results_csv, mode='a', header=not os.path.exists(results_csv))

run_ids = ['tiye34tk', 'qopjdrer', 'kpjc0swl', 'ljodmk4w', '3ixmkm41']
all_runs['SOT-SingleScale'] = run_ids
metrics = run_runs(run_ids, category)
df = pd.DataFrame.from_dict(metrics, orient='index')
df.to_csv(results_csv, mode='a', header=not os.path.exists(results_csv))

df = pd.read_csv(results_csv, index_col=0)
# Data processing
df_mean = df.groupby('name').mean()[['LSD', 'RPA', 'RCA', 'OD']]
df_median = df.groupby('name').median()[['LSD', 'RPA', 'RCA', 'OD']]
df_std = df.groupby('name').std()[['LSD', 'RPA', 'RCA', 'OD']]

df_mean_ = df_mean.round(1).astype(str) + ' (' + df_std.round(1).astype(str) + ')'
df_median_ = df_median.round(1)
dfs = {'Mean (STD)': df_mean_, 'Median': df_median_}
df_final = pd.concat(dfs, axis=1).reset_index().rename(columns={'index': 'name'}).set_index('name')

df_final = pd.concat(dfs, axis=1)
# Remove column containing name
df_final = df_final.reset_index()
# remove index
df_final = df_final.rename(columns={'index': 'name'})
# Find best element in each column, replace by \textbf{value}
# Find the second best element in each column, replace by \emph{value}
df_final = df_final.set_index('name')
for col in df_final.columns:
    # Check if not the fitst column
    # print(col)
    if 'name' not in col:
        if 'LSD' in col:
            if 'Mean (STD)' in col:
                indx = df_mean['LSD'].idxmin()
                # Get index of second best
                indx2 = df_mean['LSD'].drop(indx).idxmin()

                df_final.loc[indx, ('Mean (STD)', 'LSD')] =  '\\textbf{' + df_final.loc[indx, ('Mean (STD)', 'LSD')] + '}'
                df_final.loc[indx2, ('Mean (STD)', 'LSD')] =  '\\emph{' + df_final.loc[indx2, ('Mean (STD)', 'LSD')] + '}'
            elif 'Median' in col:
                indx = df_median['LSD'].idxmin()
                indx2 = df_median['LSD'].drop(indx).idxmin()
        
                df_final.loc[indx, ('Median', 'LSD')] =  '\\textbf{' + str(df_final.loc[indx, ('Median', 'LSD')]) + '}'
                df_final.loc[indx2, ('Median', 'LSD')] =  '\\emph{' + str(df_final.loc[indx2, ('Median', 'LSD')]) + '}'
    
        elif 'MSS' in col:
            if 'Mean (STD)' in col:
                indx = df_mean['MSS'].idxmin()
                indx2 = df_mean['MSS'].drop(indx).idxmin()
                df_final.loc[indx, ('Mean (STD)', 'MSS')] =  '\\textbf{' + str(df_final.loc[indx, ('Mean (STD)', 'MSS')]) + '}'
                df_final.loc[indx2, ('Mean (STD)', 'MSS')] =  '\\emph{' + str(df_final.loc[indx2, ('Mean (STD)', 'MSS')]) + '}'
            elif 'Median' in col:
                indx = df_median['MSS'].idxmin()
                indx2 = df_median['MSS'].drop(indx).idxmin()
                df_final.loc[indx, ('Mean (STD)', 'MSS')] =  '\\textbf{' + str(df_final.loc[indx, ('Mean (STD)', 'MSS')]) + '}'
                df_final.loc[indx2, ('Mean (STD)', 'MSS')] =  '\\emph{' + str(df_final.loc[indx2, ('Mean (STD)', 'MSS')]) + '}'

            # print(df_final.loc[df_final[col] == df_final[col].min(), col])
        elif 'OD' in col:
            # # Min absoulte value
            # df_final.loc[df_final[col].abs() == df_final[col].abs().min(), col] = '\\textbf{' + df_final[col].astype(str) + '}'
            continue
        elif 'RPA' in col:
            if 'Mean (STD)' in col:
                indx = df_mean['RPA'].idxmax()
                indx2 = df_mean['RPA'].drop(indx).idxmax()
                df_final.loc[indx, ('Mean (STD)', 'RPA')] =  '\\textbf{' + str(df_final.loc[indx, ('Mean (STD)', 'RPA')]) + '}'
                df_final.loc[indx2, ('Mean (STD)', 'RPA')] =  '\\emph{' + str(df_final.loc[indx2, ('Mean (STD)', 'RPA')]) + '}'

            elif 'Median' in col:
                indx = df_median['RPA'].idxmax()
                indx2 = df_median['RPA'].drop(indx).idxmax()
                df_final.loc[indx, ('Median', 'RPA')] =  '\\textbf{' + str(df_final.loc[indx, ('Median', 'RPA')]) + '}'
                df_final.loc[indx2, ('Median', 'RPA')] =  '\\emph{' + str(df_final.loc[indx2, ('Median', 'RPA')]) + '}'
            # df_final.loc[df_final[col] == df_final[col].max(), col] = '\\textbf{' + df_final[col].astype(str) + '}'
        elif 'RCA' in col:
            if 'Mean (STD)' in col:
                indx = df_mean['RCA'].idxmax()
                indx2 = df_mean['RCA'].drop(indx).idxmax()
                df_final.loc[indx, ('Mean (STD)', 'RCA')] =  '\\textbf{' + str(df_final.loc[indx, ('Mean (STD)', 'RCA')]) + '}'
                df_final.loc[indx2, ('Mean (STD)', 'RCA')] =  '\\emph{' + str(df_final.loc[indx2, ('Mean (STD)', 'RCA')]) + '}'
            elif 'Median' in col:
                indx = df_median['RCA'].idxmax()
                indx2 = df_median['RCA'].drop(indx).idxmax()
                df_final.loc[indx, ('Median', 'RCA')] =  '\\textbf{' + str(df_final.loc[indx, ('Median', 'RCA')]) + '}'
                df_final.loc[indx2, ('Median', 'RCA')] =  '\\emph{' + str(df_final.loc[indx2, ('Median', 'RCA')]) + '}'
            # df_final.loc[df_final[col] == df_final[col].max(), col] = '\\textbf{' + df_final[col].astype(str) + '}'
        else:
            # df_final.loc[df_final[col] == df_final[col].max(), col] = '\\textbf{' + df_final[col].astype(str) + '}'
            continue


# Save to csv, replace if file exists
df_final.to_csv(paper_results_csv, mode='w', header=True)
