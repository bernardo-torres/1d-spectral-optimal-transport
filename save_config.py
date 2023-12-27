# This code alters the save_config method of LightningCLI to save the config.yaml file in the project folder
# This was needed in pytorch-lightning 1.9.3, but may not be needed in future versions
import os
import sys

def save_config_properly(cli):
        """
        Saves config.yaml in project folder
        """
  
        trainer = cli.trainer
        parser = cli._parser(cli.subcommand)
        config = cli.config.get(str(cli.subcommand), cli.config)
        config_filename = cli.save_config_kwargs.get('config_filename', "config.yaml")
        overwrite = cli.save_config_kwargs.get('overwrite', False)
        multifile = cli.save_config_kwargs.get('multifile', False)
     
        log_dir = trainer.log_dir  # this broadcasts the directory
        assert log_dir is not None
        config_path = os.path.join(log_dir, config_filename)

        # fs = get_filesystem(log_dir)
        
        if not overwrite:
            # check if the file exists on rank 0
            file_exists = os.path.isfile(config_path) if trainer.is_global_zero else False
            # broadcast whether to fail to all ranks
            file_exists = trainer.strategy.broadcast(file_exists)
            if file_exists:
                raise RuntimeError(
                    f"expected {config_path} to NOT exist. Aborting to avoid overwriting"
                    " results of a previous run. You can delete the previous config file,"
                    " set `LightningCLI(save_config_callback=None)` to disable config saving,"
                    " or set `LightningCLI(save_config_overwrite=True)` to overwrite the config file."
                )

        project = cli.trainer.logger.experiment._project
        run_id = cli.trainer.logger.experiment._run_id
        wandb_dir = cli.trainer.logger.experiment.dir
        log_dir_experiment = os.path.join(log_dir, project, run_id)
        config_path_2 = os.path.join(log_dir_experiment, config_filename)
        config_path = os.path.join(log_dir, config_filename)

        config_path_3 = os.path.join(wandb_dir, f'train_{config_filename}')

        # save the file on rank 0
        if trainer.is_global_zero:
            # save only on rank zero to avoid race conditions.
            # the `log_dir` needs to be created as we rely on the logger to do it usually
            # but it hasn't logged anything at this point
            os.makedirs(log_dir, exist_ok=True)
            parser.save(
                config, config_path, skip_none=False, overwrite=overwrite, multifile=multifile
            )

            # os.makedirs(log_dir_experiment, exist_ok=True)
            # parser.save(
            #     config, config_path_2, skip_none=False, overwrite=overwrite, multifile=multifile
            # )

            os.makedirs(wandb_dir, exist_ok=True)
            parser.save(
                config, config_path_3, skip_none=False, overwrite=overwrite, multifile=multifile
            )