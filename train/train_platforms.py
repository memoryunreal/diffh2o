# Copyright (c) Meta Platforms, Inc. and affiliates.

import os

class TrainPlatform:
    def __init__(self, save_dir):
        pass

    def report_scalar(self, name, value, iteration, group_name=None):
        pass

    def report_args(self, args, name):
        pass

    def close(self):
        pass


class ClearmlPlatform(TrainPlatform):
    def __init__(self, save_dir):
        from clearml import Task
        path, name = os.path.split(save_dir)
        self.task = Task.init(project_name='motion_diffusion',
                              task_name=name,
                              output_uri=path)
        self.logger = self.task.get_logger()

    def report_scalar(self, name, value, iteration, group_name):
        self.logger.report_scalar(title=group_name, series=name, iteration=iteration, value=value)

    def report_args(self, args, name):
        self.task.connect(args, name=name)

    def close(self):
        self.task.close()


class TensorboardPlatform(TrainPlatform):
    def __init__(self, save_dir):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=save_dir)

    def report_scalar(self, name, value, iteration, group_name=None):
        self.writer.add_scalar(f'{group_name}/{name}', value, iteration)

    def close(self):
        self.writer.close()


class NoPlatform(TrainPlatform):
    def __init__(self, save_dir):
        pass


class WandbPlatform(TrainPlatform):
    """Weights & Biases logging platform.

    Uses environment variables for configuration:
    - WANDB_BASE_URL: wandb server URL (for local wandb)
    - WANDB_API_KEY: API key
    - WANDB_ENTITY: user/team name
    - WANDB_PROJECT: project name
    - WANDB_RUN_NAME: run name

    Source your wandb_init.sh before running training.
    """
    def __init__(self, save_dir, config=None):
        import wandb

        # Get run name from env or use save_dir name
        run_name = os.environ.get('WANDB_RUN_NAME', os.path.basename(save_dir))
        project = os.environ.get('WANDB_PROJECT', 'diffh2o')
        entity = os.environ.get('WANDB_ENTITY', None)

        # Initialize wandb run
        self.run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            config=config,
            dir=save_dir,
            resume='allow',
        )
        print(f"WandB initialized: {self.run.url}")

    def report_scalar(self, name, value, iteration, group_name=None):
        import wandb
        if group_name:
            wandb.log({f'{group_name}/{name}': value}, step=iteration)
        else:
            wandb.log({name: value}, step=iteration)

    def report_args(self, args, name):
        import wandb
        if hasattr(args, '__dict__'):
            wandb.config.update(args.__dict__, allow_val_change=True)
        else:
            wandb.config.update(dict(args), allow_val_change=True)

    def close(self):
        import wandb
        wandb.finish()

