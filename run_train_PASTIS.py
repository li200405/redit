import argparse
import logging
import logging.config
import os
import sys
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from omegaconf import OmegaConf

from lib import config_utils, data_utils, utils
from lib.formatter import RawFormatter
from lib.logger import prepare_logger
from lib.models.vmamba_blocks import selective_scan_backend_name
from diffusers.schedulers import DDIMScheduler

parser = ArgumentParser(
    description='RESTORE-DiT: Reliable satellite image time series reconstruction by multimodal sequential diffusion transformer (Training)',
    formatter_class=RawFormatter
)
parser.add_argument(
    'config_file', type=str, help='yaml configuration file to augment/overwrite the settings in configs/default.yaml'
)
parser.add_argument(
    '--save_dir', type=str, required=True, help='Path to the directory where models and logs should be saved'
)
parser.add_argument('--wandb', action='store_true', default=False, help='Use Weights & Biases instead of TensorBoard')
parser.add_argument('--wandb_project', type=str, default='utilise', help='Wandb project name')


def setup_distributed() -> tuple[bool, int, int]:
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    if world_size <= 1:
        return False, 0, 1
    if not torch.cuda.is_available():
        raise RuntimeError('DDP requires CUDA/NCCL for this training entry point.')

    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    return True, dist.get_rank(), world_size


def main(args: argparse.Namespace) -> None:
    distributed, rank, world_size = setup_distributed()
    is_main_process = rank == 0
    if torch.cuda.is_available():
        # Input resolution is fixed during Chongqing training, so convolution
        # autotuning and TF32 can safely improve throughput on Ampere+ cards.
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
    prog_name = 'RESTORE-DiT: Reliable satellite image time series reconstruction by multimodal sequential diffusion transformer (Training)'
    if is_main_process:
        print('\n{}\n{}\n'.format(prog_name, '=' * len(prog_name)))

    if not os.path.exists(args.config_file):
        raise FileNotFoundError(f'ERROR: Cannot find the yaml configuration file: {args.config_file}')

    # Import the user configuration file
    cfg_custom = config_utils.read_config(args.config_file)

    if not cfg_custom:
        sys.exit(1)

    # Augment/overwrite the default parameter settings with the runtime arguments given by the user
    cfg_default = config_utils.read_config('configs/default_train.yaml')
    config = OmegaConf.merge(cfg_default, cfg_custom)
    config.output.output_directory = args.save_dir

    if args.wandb:
        config.wandb = OmegaConf.create()
        config.wandb.project = args.wandb_project

    # Only rank 0 creates the experiment directory, then shares it with every
    # DDP worker so logs and checkpoints have one consistent location.
    if is_main_process:
        config.output.experiment_folder = utils.create_output_directory(config)
    if distributed:
        experiment_folder = [config.output.experiment_folder if is_main_process else None]
        dist.broadcast_object_list(experiment_folder, src=0)
        config.output.experiment_folder = experiment_folder[0]
        dist.barrier()

    # Set up the logger
    log_file = os.path.join(config.output.experiment_folder, 'run.log') if config.output.experiment_folder else None
    logger = prepare_logger(
        'root_logger',
        level=logging.INFO,
        log_to_console=is_main_process,
        log_file=log_file if is_main_process else None,
    )

    # Print runtime arguments to the console
    if is_main_process:
        logger.info('Configuration file: %s', args.config_file)
        logger.info('\nSettings\n--------\n')
        config_utils.print_config(config, logger=logger)
        if distributed:
            logger.info(
                'DDP enabled: %d GPUs, per-GPU batch=%d, effective global batch=%d',
                world_size,
                config.training_settings.batch_size,
                config.training_settings.batch_size * world_size,
            )

    if config.misc.random_seed is not None:
        utils.set_seed(config.misc.random_seed)

    # ------------------------------------------------- Data loaders ------------------------------------------------- #
    if is_main_process:
        logger.info('\nInitialize data loader (training set)...')
    train_loader = data_utils.get_dataloader(
        config, phase='train', pin_memory=config.misc.pin_memory, drop_last=True, logger=logger
    )
    if is_main_process:
        logger.info('Initialize data loader (validation set)...\n')
    val_loader = None

    if is_main_process:
        logger.info('Number of training samples: %d', train_loader.dataset.__len__())

    # ----------------------------------------- Prepare the output directory ----------------------------------------- #
    logger.info('\nPrepare output folders and files\n--------------------------------\n')

    # Save the path of the checkpoint directory
    config.output.checkpoint_dir = os.path.join(config.output.experiment_folder, 'checkpoints')
    if is_main_process:
        os.makedirs(config.output.checkpoint_dir, exist_ok=True)
        logger.info('Model weights will be stored in: %s\n', config.output.checkpoint_dir)
    if distributed:
        dist.barrier()

    # Write the runtime configuration to file
    if is_main_process:
        config_file = os.path.join(config.output.experiment_folder, 'config.yaml')
        config_utils.write_config(config, config_file)

    # ----------------------------------------------- Define the model ----------------------------------------------- #
    if is_main_process:
        logger.info('\nModel Architecture\n------------------\n')
        logger.info('Architecture: %s', config.method.model_type)


    # input_dim = train_loader.dataset.num_channels
    input_dim = 10   # 4 for RGB_NIR

    model, args_model = utils.get_model(config, input_dim, logger)

    if is_main_process:
        logger.info('Number of trainable parameters: %d\n', utils.count_model_parameters(model))
        logger.info('VMamba selective-scan backend: %s\n', selective_scan_backend_name())

    # Log model parameters to file
    if is_main_process:
        config_file = os.path.join(config.output.experiment_folder, 'model_config.yaml')
        config_utils.write_config(OmegaConf.create({config.method.model_type: args_model}), config_file)

    # Write model architecture to txt file
    if is_main_process and config.output.plot_model_txt:
        file = os.path.join(config.output.experiment_folder, 'model_parameters.txt')
        logger.info('Writing model architecture to file: %s\n', file)
        utils.write_model_structure_to_file(
            file, model, config.training_settings.batch_size, train_loader.dataset.max_seq_length, input_dim,
            train_loader.dataset.image_size
        )

    if torch.cuda.is_available():
        device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', '0')))
        model = model.to(device)
    if distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[int(os.environ['LOCAL_RANK'])],
            gradient_as_bucket_view=True,
            static_graph=True,
        )
    # --------------------------------------------------- Training --------------------------------------------------- #
    if is_main_process:
        logger.info('\nPrepare training\n----------------\n')
        logger.info('Python version: %s', sys.version)
        logger.info('Torch version: %s', torch.__version__)
        logger.info('CUDA version: %s\n', torch.version.cuda)

    # Get optimizer and learning rate scheduler
    optimizer = utils.get_optimizer(config, model, logger)
    scheduler = utils.get_scheduler(config, optimizer, train_loader.dataset.__len__(), logger)

    noise_scheduler = DDIMScheduler(num_train_timesteps=1000)

    if config.misc.random_seed is not None:
        utils.set_seed(config.misc.random_seed)

    # Initialize the trainer and start training
    trainer = utils.get_trainer(config, train_loader, val_loader, model, noise_scheduler, optimizer, scheduler)
    try:
        trainer.train()
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)

    main(parser.parse_args())
