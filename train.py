import torch
import yaml
import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI
from trainer import Trainer
from save_config import save_config_properly
import wandb
import os
from utils import get_cqt_n_bins
from features import get_default_cqt_args
import random
import numpy as np

import sys

sys.path.append("..")  # Necessary I think in order for cli to see every subpath in config files


class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--ckpt_path", default=None)  # For resuming training
        parser.add_argument(
            "--ckpt_dirpath", default=None
        )  #  If None, will be wandb.run.dir/checkpoints

        # parser.add_argument("--sample_rate", default=16000)
        parser.add_argument("--project", default="unnamed-project")
        parser.add_argument("--name", default="run")
        parser.add_argument("--group", default="no-group")
        parser.add_argument("--dataset", default="")
        parser.add_argument("--log_dir", default="logs")

        parser.add_argument("configs", nargs="+", type=str, default=[])

        parser.add_argument("--sample_rate", default=16000, type=int)
        parser.add_argument("--loss", default="", type=str)

        parser.add_argument("--batch_size", default=16, type=int)
        parser.add_argument("--num_workers", default=4, type=int)

        parser.link_arguments("sample_rate", "data.init_args.sample_rate")
        parser.link_arguments("sample_rate", "model.sample_rate")
        parser.link_arguments("sample_rate", "model.decoder.init_args.sample_rate")
        parser.link_arguments("batch_size", "data.init_args.batch_size")
        parser.link_arguments("batch_size", "data.init_args.batch_size_val")
        parser.link_arguments("num_workers", "data.init_args.num_workers")

        # Set logger to pytorch_lightning.loggers.WandbLogger
        parser.set_defaults(
            {"trainer.logger": {"class_path": "pytorch_lightning.loggers.WandbLogger",
                                "init_args": {"log_model": False }}}
        )

        # Model encoder and decoder are identity by default
        identity = {"class_path": "torch.nn.Identity"}
        parser.set_defaults(
            {
                "model.encoder": identity,
                "model.decoder": identity,
                "model.loss_fn": identity,
            }
        )

    def before_instantiate_classes(self) -> None:
        if self.config["project"] != "unnamed-project":
            self.config["trainer"]["logger"]["init_args"]["project"] = self.config["project"]

        group = self.config["group"]
        # Append loss to the group
        group = f"{group}-{self.config['loss']}" if self.config["loss"] != "" else group
        # Append dataset name to the group
        group = f"{group}-{self.config['dataset']}" if self.config["dataset"] != "" else group
        self.config["trainer"]["logger"]["init_args"]["group"] = group
        # Compose run name, add dataset name and overfit_batches if set
        if self.config["name"] != "run":
            # if trainer overfit_batches is set, append it to the name
            if self.config["trainer"]["overfit_batches"] > 0:
                self.config[
                    "name"
                ] = f"{self.config['name']}-overfit{self.config['trainer']['overfit_batches']}"
            # if overfit_one_sample  in datamodule is true, append it to the name
            if self.config["data"]["init_args"].get("overfit_one_sample", False):
                self.config["name"] = f"{self.config['name']}-overfit_one_sample"
            

            # Append temperature to the run name
            if self.config["model"].get("temperature", 1.0) != 1.0:
                t = str(self.config['model']['temperature'])[2:]
                # remove 
                self.config["name"] = f"{self.config['name']}-t{t}"
            
            # Append encoder kernel size to the run name
            if self.config["model"]["encoder"]["init_args"].get("kernel_size", 15) != 15:
                self.config["name"] = f"{self.config['name']}-k{self.config['model']['encoder']['init_args']['kernel_size']}"

            # Append if rolloff is used to the run name
            if self.config["model"]["decoder"]["init_args"].get("apply_roll_off", False):
                self.config["name"] = f"{self.config['name']}-ro"

            self.config["trainer"]["logger"]["init_args"]["name"] = self.config["name"]
            

        # Add log_dir to logger. It is log_dir/project/group
        self.config["trainer"]["logger"]["init_args"]["save_dir"] = os.path.join(
            self.config["log_dir"], self.config["project"], self.config["group"]
        )

        # Compose wandb dir as logs/project/group/wandb, create it if it does not exist
        wandb_dir = os.path.join(
            self.config["log_dir"],
            self.config["project"],
            self.config["group"],
            "wandb",
        )

        os.makedirs(wandb_dir, exist_ok=True)

        # Encoder
        default_cqt_kwargs = get_default_cqt_args(self.config["sample_rate"])

        bins_per_semitone = self.config["model"]["feature_extractor"].get(
            "bins_per_semitone", default_cqt_kwargs["bins_per_semitone"]
        )
        fmin = self.config["model"]["feature_extractor"].get("fmin", default_cqt_kwargs["fmin"])
        n_bins_in_encoder = get_cqt_n_bins(
            self.config["sample_rate"], fmin=fmin, bins_per_semitone=bins_per_semitone
        )
        self.config["model"]["encoder"]["init_args"]["n_bins_in"] = n_bins_in_encoder
        self.config["model"]["encoder"]["init_args"]["output_size"] = n_bins_in_encoder


def run_cli():

    cli = CLI(
        model_class=Trainer,
        datamodule_class=pl.LightningDataModule,  # pl.LightningDataModule,
        subclass_mode_data=True,
        save_config_kwargs={
            "overwrite": True,
        },  # to overwrite saved config file
        run=False,  # only instantiate, does not run fit
    )

    # Set seed
    set_seed(cli.config["seed_everything"])

    wandb.config.update({"model": dict(cli.config["model"])}, allow_val_change=True)
    wandb.config.update({"data": dict(cli.config["data"])}, allow_val_change=True)

    cli.config["ckpt_dirpath"] = (
        os.path.join(wandb.run.dir, "checkpoints")
        if cli.config["ckpt_dirpath"] is None
        else cli.config["ckpt_dirpath"]
    )

    # Ignore checkpoints in wandb
    os.environ["WANDB_IGNORE_GLOBS"] = "*.ckpt"

    # Get all checkpoint callbacks and change their dirpath to ckpt_dirpath if None
    best_val_loss_callback = None
    for callback in cli.trainer.callbacks:
        if isinstance(callback, pl.callbacks.ModelCheckpoint):
            if callback.dirpath is None:
                callback.dirpath = cli.config["ckpt_dirpath"]
            else:
                print(
                    "Warning: ModelCheckpoint dirpath is not None, not changing it to ckpt_dirpath"
                )
            # if callback.monitor == "loss/val":
            if callback.monitor == "val_metrics/log_spectral_distance":
                best_val_loss_callback = callback

    # This makes sure that the config used to launch training is saved in the correct directory
    save_config_properly(cli)

    # For loading checkpoint
    ckpt_path = cli.config["ckpt_path"]
    if ckpt_path is not None:
        step = torch.load(ckpt_path, map_location="cpu")["global_step"]
        cli.trainer.fit_loop.epoch_loop._batches_that_stepped = step

    cli.trainer.fit(cli.model, cli.datamodule, ckpt_path=ckpt_path)

    # Load best checkpoint and test
    model = cli.model
    if best_val_loss_callback is not None:
        model = Trainer.load_from_checkpoint(best_val_loss_callback.best_model_path)
    cli.trainer.test(model, cli.datamodule)

 

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # torch.use_deterministic_algorithms(True, warn_only=True)
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(seed, workers=True)

if __name__ == "__main__":

    old_argv = sys.argv[1:]
    

    file = old_argv[1]
    assert file.endswith(".yaml"), f"Config file {file} must end with .yaml"
    # Load the master configuration file
    with open(file, "r") as file:
        master_config = yaml.safe_load(file)
    
    if "configs" not in master_config:
        run_cli()
        exit(0)

    # Alter argv to include each config file in master_config
    # eg if master_config = {'configs': ['a.yaml', 'b': 'b.yaml']}
    # argv = ['--config', 'a.yaml', '--config', 'b.yaml']
    print(file)
    argv = []
    for config in master_config["configs"]:
        argv.extend(["--config", config])
    sys.argv = [sys.argv[0]] + argv

    # Add the rest of the arguments
    sys.argv.extend(old_argv)

    run_cli()
