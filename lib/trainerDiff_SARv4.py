import logging
import time
import torch
from omegaconf import DictConfig, OmegaConf
from typing import Any, Dict, Optional, Tuple
from torch import Tensor
from PIL import Image
import numpy as np
import prodict
from prodict import Prodict
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from lib import logger, visutils
from lib.logger import AverageMeter
import wandb

from tqdm.auto import tqdm
from pathlib import Path
import os


def seconds_to_dd_hh_mm_ss(seconds_elapsed: int) -> Tuple[int, int, int, int]:
    days = seconds_elapsed // (24 * 3600)
    seconds_remainder = seconds_elapsed % (24 * 3600)
    hours = seconds_remainder // 3600
    seconds_remainder %= 3600
    minutes = seconds_remainder // 60
    seconds_remainder %= 60
    seconds = seconds_remainder

    return days, hours, minutes, seconds


def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid

def evaluate(output_dir, epoch, pipeline, val_batch):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size = 1,
        generator=torch.manual_seed(0),
    ).images

    output = pipeline(val_batch['y'], val_batch['masks'], generator=torch.manual_seed(0)).images

    # Make a grid out of the images
    image_grid = make_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")

class Trainer:
    def __init__(
            self,
            args: DictConfig,
            train_loader: torch.utils.data.dataloader.DataLoader,
            val_loader: torch.utils.data.dataloader.DataLoader,
            model,
            noise_scheduler,
            optimizer,
            scheduler
    ):
        self.args = args
        self.use_wandb = bool('wandb' in args)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.CTHW = self.args.data.ifCTHW

        self.dataloader = {'train': train_loader, 'val': val_loader}
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model.to(self.device)
        self.args.accum_iter = self.args.get('accum_iter', 1)  # accumulate gradients for `accum_iter` iterations

        self.target_quality_args = self.args.get('target_quality', {})
        self.landcover_loss_args = self.args.get('landcover_loss', {})
        self.restoration_loss_args = self.args.get('restoration_loss', {})
        # self.compute_metrics = EvalMetrics(self.args.metrics)

        # Losses: Initialize statistics
        self.train_stats = self._stats_meter(stats_type='loss')
        self.val_stats = self._stats_meter(stats_type='loss')

        # Losses: Initialize metrics
        self.train_metrics = self._stats_meter(stats_type='metrics')
        self.val_metrics = self._stats_meter(stats_type='metrics')

        self.best_loss = np.inf
        self.epoch_best_loss = np.nan

        os.makedirs(self.args.save_dir, exist_ok=True)
        os.makedirs(self.args.checkpoint_dir, exist_ok=True)
        self.args.path_model_best = os.path.join(self.args.checkpoint_dir, 'Model_best.pth')
        self.args.path_model_last = os.path.join(self.args.checkpoint_dir, 'Model_last.pth')
        self.logger = logger.prepare_logger('train_logger', level=logging.INFO, log_to_console=True,
                                            log_file=os.path.join(args.save_dir, 'training.log'))

        self.noise_scheduler = noise_scheduler

        # Set up wandb
        if self.use_wandb:
            os.makedirs(self.args.wandb.dir, exist_ok=True)
            wandb.init(**self.args.wandb, settings=wandb.Settings(start_method="fork"))
            wandb.config.update(OmegaConf.to_container(self.args))
            self.writer = None

            # Define the wandb summary metrics
            # for key, value in self.args.metrics.items():
            #     if key == 'masked_metrics':
            #         pass
            #     elif value:
            #         wandb.define_metric(f"train_metrics/{key}", summary=OBJECTIVE[key])
            #         wandb.define_metric(f"val_metrics/{key}", summary=OBJECTIVE[key])

            # wandb.define_metric('train/total_loss', summary=OBJECTIVE['total_loss'])
            # wandb.define_metric('val/total_loss', summary=OBJECTIVE['total_loss'])
        else:
            os.makedirs(os.path.join(self.args.save_dir, 'tb'), exist_ok=True)
            self.writer = SummaryWriter(log_dir=os.path.join(self.args.save_dir, 'tb'))

        # Resume training
        if self.args.resume and self.args.pretrained_path:
            self._resume(path=self.args.pretrained_path)
        else:
            self.logger.info('\nTraining from scratch.\n')
            self.epoch = 0
            self.iter = 0

    def _stats_meter(self, stats_type: str) -> prodict.Prodict:
        meters = Prodict()
        stats = self._stats_dict(stats_type)
        for key, _ in stats.items():
            meters[key] = AverageMeter()

        return meters

    def _get_lr(self, group: int = 0) -> float:
        return self.optimizer.param_groups[group]['lr']

    def _resume(self, path: str) -> None:
        """
        Resumes training.

        Args:
            path:  str, path of the pretrained model weights.
        """

        if not os.path.isfile(path):
            raise FileNotFoundError(f'No checkpoint found at {path}\n')

        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #
        # if self.args.get('load_scheduler_state_dict', True) and 'scheduler_state_dict' in checkpoint:
        #     self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        #
        # # Extract the last training epoch
        # self.epoch = checkpoint['epoch'] + 1
        # self.iter = checkpoint['iter']
        # self.args.num_epochs += self.epoch
        #
        # # Best validation loss so far
        # self.best_loss = checkpoint['best_loss']
        # self.epoch_best_loss = checkpoint['epoch']
        #
        # self.logger.info('\n\nRestoring the pretrained model from epoch %d.', self.epoch - 1)
        # self.logger.info('Successfully loaded pretrained model weights from %s.\n', path)
        # self.logger.info('Current best loss %.4f\n', self.best_loss)

        self.epoch = 0
        self.iter = 0

    def _log_iter_epoch(self) -> None:
        if self.use_wandb:
            wandb.log({'epoch': self.epoch}, step=self.iter)
        else:
            self.writer.add_scalar('epoch', self.epoch, self.iter)

    def _log_learning_rate(self) -> None:
        if self.use_wandb:
            wandb.log({'log_lr': np.log10(self._get_lr()), 'epoch': self.epoch}, step=self.iter)
        else:
            self.writer.add_scalar('log_lr', np.log10(self._get_lr()), self.epoch)

    def _stats_dict(self, stats_type: str) -> prodict.Prodict:
        stats = Prodict()

        if stats_type == 'metrics':
            masked_metrics = self.args.metrics.masked_metrics
            for key, value in self.args.metrics.items():
                if key == 'masked_metrics':
                    pass
                elif value:
                    if masked_metrics and key != 'ssim':
                        stats[f'masked_{key}'] = np.inf
                    else:
                        stats[key] = np.inf

        elif stats_type == 'loss':
            for key in (
                'reconstruction_loss',
                'reliability_loss',
                'building_edge_loss',
                'spatial_gradient_loss',
                'spectral_angle_loss',
                'temporal_difference_loss',
                'total_loss',
            ):
                stats[key] = np.inf

        return stats

    def _save_checkpoint(self, filepath: str) -> None:
        state = {
            'epoch': self.epoch,
            'iter': self.iter,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'best_epoch': self.epoch_best_loss
        }

        if self.scheduler is not None:
            state['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(state, filepath)

    def _log_stats_meter(self, phase: str) -> None:
        if self.use_wandb:
            if phase == 'train':
                wandb.log({
                    'train_losses/' + k: v.avg for k, v in self.train_stats.items()
                }, step=self.iter)
            else:
                stats = {'val_losses/' + k: v.avg for k, v in self.val_stats.items()}
                stats['epoch'] = self.epoch
                wandb.log(stats, step=self.iter)

        else:
            if phase == 'train':
                for k, v in self.train_stats.items():
                    self.writer.add_scalar('train_losses/' + k, v.avg, self.iter)
            else:
                for k, v in self.val_stats.items():
                    self.writer.add_scalar('val_losses/' + k, v.avg, self.iter)

    def train(self) -> None:
        # Log gradients and model parameters
        if self.use_wandb and self.args.get('log_gradients', False):
            wandb.watch(self.model, log='all')

        self.logger.info('\nStart training...\n')
        start_time = time.time()

        with tqdm(range(self.epoch, self.args.num_epochs), leave=True) as tnr:
            tnr.set_description("Epoch")
            tnr.set_postfix(epoch=self.epoch, training_loss=np.nan)
            for _ in tnr:
                if self.scheduler is not None:
                    self._log_learning_rate()

                # -------------------------------- TRAINING -------------------------------- #
                self.train_epoch(tnr)

                # After the epoch if finished, update the learning rate scheduler
                if self.scheduler is not None:
                    self._log_learning_rate()

                    if self.scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                        self.scheduler.step(self.val_stats.total_loss.avg)
                    else:
                        self.scheduler.step()

                # Save the model at the selected interval and validate the model
                if (self.epoch + 1) % self.args.checkpoint_every_n_epochs == 0:
                    name = 'Model_after_' + str(self.epoch + 1) + '_epochs.pth'
                    self._save_checkpoint(os.path.join(self.args.checkpoint_dir, name))



                self.epoch += 1

        time_elapsed = int(time.time() - start_time)
        self.logger.info(
            '\n\nTraining finished!\nTraining time: %dd %dh %dm %ds' % seconds_to_dd_hh_mm_ss(time_elapsed))
        self.logger.info('\nBest model at epoch: %d', self.epoch_best_loss)
        self.logger.info(f'Validation loss of the best model: {self.best_loss:.4f}')

        # Save the last model
        self._save_checkpoint(self.args.path_model_last)

        if self.use_wandb:
            wandb.finish()

    def train_epoch(self, tnr=None) -> None:
        # Initialize stats meter
        self.train_stats = self._stats_meter(stats_type='loss')
        # self.train_metrics = self._stats_meter(stats_type='metrics')
        self.model.train()

        # Clear gradients
        for param in self.model.parameters():
            param.grad = None

        with tqdm(self.dataloader['train'], leave=False) as tnr_train:
            tnr_train.set_description("Training")
            tnr_train.set_postfix(epoch=self.epoch, training_loss=np.nan, best_loss=self.best_loss)

            for i, batch in enumerate(tnr_train):
                self._log_iter_epoch()
                loss, loss_dict = self.inference_one_batch(batch)

                # Update to stats_meter
                for key, value in loss_dict.items():
                    self.train_stats[key].update(value)

                loss = loss / self.args.accum_iter
                loss.backward()

                # gradient accumulation
                if ((i + 1) % self.args.accum_iter == 0) or (i + 1 == len(self.dataloader['train'])):
                    # Gradient clipping
                    if getattr(self.args, 'gradient_clip_norm', False) and self.args.gradient_clip_norm > 0.:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.gradient_clip_norm)

                    elif getattr(self.args, 'gradient_clip_value', False) and self.args.gradient_clip_value > 0.:
                        torch.nn.utils.clip_grad_value_(self.model.parameters(), self.args.gradient_clip_value)

                    self.optimizer.step()

                    # Clear gradients
                    for param in self.model.parameters():
                        param.grad = None

                self.iter += 1

            tnr_train.set_postfix(
                epoch=self.epoch,
                training_loss=self.train_stats.total_loss.avg,
            )

            if tnr is not None:
                tnr.set_postfix(
                    epoch=self.epoch,
                    training_loss=self.train_stats.total_loss.avg,
                    best_loss=self.best_loss,
                )


            self.logger.info((f'Train:\tEpoch: {self.epoch}\t' + f'learning rate: {self._get_lr():.8f}\t'
                              ''.join([f'{k}: {v.avg:.6f}\t' for k, v in self.train_stats.items()])))

            if self.best_loss > self.train_stats.total_loss.avg:
                self.best_loss = self.train_stats.total_loss.avg
                self.epoch_best_loss = self.epoch
                self._save_checkpoint(self.args.path_model_best)

            # Reset stats and metrics
            for key in self.train_stats:
                self.train_stats[key].reset()



    @staticmethod
    def _rgb_like_channels(frames: Tensor) -> Tensor:
        if frames.shape[2] >= 3:
            indices = torch.tensor([2, 1, 0], device=frames.device)
            return frames.index_select(2, indices)
        return frames

    @staticmethod
    def _sobel_edges(frames: Tensor) -> Tensor:
        b, t, c, h, w = frames.shape
        flat = frames.reshape(b * t, c, h, w)
        kernel_x = flat.new_tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3) / 4.0
        kernel_y = kernel_x.transpose(-1, -2)
        kernel_x = kernel_x.repeat(c, 1, 1, 1)
        kernel_y = kernel_y.repeat(c, 1, 1, 1)
        grad_x = F.conv2d(flat, kernel_x, padding=1, groups=c)
        grad_y = F.conv2d(flat, kernel_y, padding=1, groups=c)
        return (grad_x.abs() + grad_y.abs()).reshape(b, t, c, h, w)

    def _building_edge_loss(
            self,
            prediction: Tensor,
            target: Tensor,
            building_prob: Optional[Tensor],
            loss_weight: Tensor,
    ) -> Tensor:
        if building_prob is None:
            return prediction.new_tensor(0.0)

        prediction_edges = self._sobel_edges(self._rgb_like_channels(prediction))
        target_edges = self._sobel_edges(self._rgb_like_channels(target))
        edge_weight = (loss_weight * building_prob.detach()).expand_as(prediction_edges)
        edge_error = (prediction_edges - target_edges).abs()
        return (
            (edge_error * edge_weight).sum()
            / edge_weight.sum().clamp_min(1.0)
        )

    def _reconstruction_loss(
            self,
            prediction: Tensor,
            target: Tensor,
            valid_mask: Tensor,
    ) -> Tensor:
        loss_type = str(
            self.restoration_loss_args.get('reconstruction_type', 'charbonnier')
        ).lower()
        error = prediction - target
        if loss_type == 'mse':
            pointwise = error.pow(2)
        elif loss_type == 'l1':
            pointwise = error.abs()
        elif loss_type == 'charbonnier':
            eps = float(self.restoration_loss_args.get('charbonnier_eps', 1.e-3))
            pointwise = torch.sqrt(error.pow(2) + eps * eps) - eps
        else:
            raise ValueError('Unsupported reconstruction loss: {}'.format(loss_type))
        return (pointwise * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)

    def _spatial_gradient_loss(
            self,
            prediction: Tensor,
            target: Tensor,
            loss_weight: Tensor,
    ) -> Tensor:
        prediction_edges = self._sobel_edges(self._rgb_like_channels(prediction))
        target_edges = self._sobel_edges(self._rgb_like_channels(target))
        weight = loss_weight.expand_as(prediction_edges)
        return (
            ((prediction_edges - target_edges).abs() * weight).sum()
            / weight.sum().clamp_min(1.0)
        )

    @staticmethod
    def _spectral_angle_loss(
            prediction: Tensor,
            target: Tensor,
            loss_weight: Tensor,
    ) -> Tensor:
        prediction_01 = ((prediction + 1.0) / 2.0).clamp(0.0, 1.0)
        target_01 = ((target + 1.0) / 2.0).clamp(0.0, 1.0)
        cosine = F.cosine_similarity(prediction_01, target_01, dim=2, eps=1.e-6)
        angle_error = 1.0 - cosine
        weight = loss_weight[:, :, 0]
        return (angle_error * weight).sum() / weight.sum().clamp_min(1.0)

    @staticmethod
    def _temporal_difference_loss(
            prediction: Tensor,
            target: Tensor,
            loss_weight: Tensor,
            dates: Optional[Tensor],
    ) -> Tensor:
        if prediction.shape[1] <= 1:
            return prediction.new_tensor(0.0)
        if dates is None:
            intervals = prediction.new_ones(prediction.shape[0], prediction.shape[1] - 1)
        else:
            intervals = (dates[:, 1:] - dates[:, :-1]).abs().to(prediction.dtype)
            intervals = intervals.clamp_min(1.0)
        intervals = intervals[:, :, None, None, None]
        predicted_velocity = (prediction[:, 1:] - prediction[:, :-1]) / intervals
        target_velocity = (target[:, 1:] - target[:, :-1]) / intervals
        pair_weight = torch.maximum(loss_weight[:, 1:], loss_weight[:, :-1])
        pair_weight = pair_weight.expand_as(predicted_velocity)
        return (
            ((predicted_velocity - target_velocity).abs() * pair_weight).sum()
            / pair_weight.sum().clamp_min(1.0)
        )


    def inference_one_batch(
            self, batch: Dict[str, Any]
    ) -> Tuple[Tensor, Dict[str, float]]:

        y_0 = batch['y'].cuda()                #  (B, T, C, H, W)
        mask = batch['masks'].cuda()           #  (B, T, 1, H, W)
        cloud_mask = batch.get('cloud_mask')
        cloud_mask = cloud_mask.cuda() if cloud_mask is not None else torch.zeros_like(mask)
        date = batch['position_days'].cuda()            #  (B, T)           'position_days' or None
        # date = None
        cond = batch['cond'].cuda()            #  (B, T, 3, H, W)
        cond_dense = batch.get('cond_dense')
        cond_dense = cond_dense.cuda() if cond_dense is not None else None
        date_cond_dense = batch.get('position_days_cond_dense')
        date_cond_dense = date_cond_dense.cuda() if date_cond_dense is not None else None
        date_dense_target = batch.get('position_days_s2_raw')
        date_dense_target = date_dense_target.cuda() if date_dense_target is not None else date
        bs, length = y_0.shape[0], y_0.shape[1]

        if self.CTHW:
            y_0 = y_0.permute(0, 2, 1, 3, 4)     #  (B, C, T, H, W)
            mask = mask.permute(0, 2, 1, 3, 4)   #  (B, 1, T, H, W)
            cloud_mask = cloud_mask.permute(0, 2, 1, 3, 4)

        # with torch.cuda.amp.autocast(enabled=self.args.use_amp):  # casts operations to mixed precision

        noise = torch.randn(y_0.shape).to(y_0.device)
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (bs,), device=y_0.device).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_images = self.noise_scheduler.add_noise(y_0, noise, timesteps)

        quality_enabled = self.target_quality_args.get('enabled', True)
        if quality_enabled:
            quality_output = self.model(
                y_0,
                timesteps,
                date=date,
                cond=cond,
                cond_dense=cond_dense,
                date_cond_dense=date_cond_dense,
                date_dense_target=date_dense_target,
                cloud_mask=cloud_mask,
                quality_target=y_0,
                quality_only=True,
            )

            warmup_epochs = max(
                int(self.target_quality_args.get('warmup_epochs', 100)), 1
            )
            quality_strength = min(float(self.epoch + 1) / warmup_epochs, 1.0)
            known_clear = 1.0 - cloud_mask
            learned_reliability = quality_output['reliability']
            reliability = known_clear * (
                (1.0 - quality_strength) + quality_strength * learned_reliability
            )

            # The estimated dirty probability is detached in the input path to
            # prevent the reliability branch from hiding difficult pixels.
            estimated_dirty = (1.0 - reliability).detach()
            quality_mask = torch.maximum(mask, estimated_dirty)
        else:
            quality_output = None
            reliability = 1.0 - cloud_mask
            quality_mask = torch.maximum(mask, cloud_mask)

        model_input = noisy_images * quality_mask + (1. - quality_mask) * y_0
        landcover_loss_enabled = bool(
            self.landcover_loss_args.get('enabled', False)
        )
        model_output = self.model(
            model_input,
            timesteps,
            date=date,
            cond=cond,
            cond_dense=cond_dense,
            date_cond_dense=date_cond_dense,
            date_dense_target=date_dense_target,
            cloud_mask=quality_mask,
            landcover_source=y_0,
            return_aux=landcover_loss_enabled,
        )
        if isinstance(model_output, dict):
            pred_hat = model_output['prediction']
            building_prob = model_output.get('building_prob')
        else:
            pred_hat = model_output
            building_prob = None
        # noise_hat = self.model(noisy_images * mask + (1. - mask) * y_0, timesteps, batch_positions=batch['position_days'])

        if quality_output is not None:
            consensus = quality_output['consensus'].detach()
            support = quality_output['support'].detach()
            corrected_target = reliability * y_0 + (1.0 - reliability) * consensus

            # When no clean temporal reference exists, the supervision weight
            # remains low instead of inventing a high-confidence pseudo target.
            supervision_confidence = (
                reliability.detach() + (1.0 - reliability.detach()) * support
            )
            loss_weight = mask * supervision_confidence
            valid_loss_mask = loss_weight.expand_as(y_0)
            reconstruction_loss = self._reconstruction_loss(
                pred_hat, corrected_target, valid_loss_mask
            )

            reliability_loss = F.binary_cross_entropy(
                quality_output['raw_reliability'].clamp(1e-5, 1.0 - 1e-5),
                quality_output['pseudo_reliability'],
            )
            reliability_loss_w = float(
                self.target_quality_args.get('reliability_loss_w', 0.25)
            )
            reliability_loss = reliability_loss_w * reliability_loss
            edge_target = corrected_target
        else:
            loss_weight = mask * (1. - cloud_mask)
            valid_loss_mask = loss_weight.expand_as(y_0)
            reconstruction_loss = self._reconstruction_loss(
                pred_hat, y_0, valid_loss_mask
            )
            reliability_loss = pred_hat.new_tensor(0.0)
            edge_target = y_0

        building_edge_loss = pred_hat.new_tensor(0.0)
        if landcover_loss_enabled and building_prob is not None:
            edge_loss = self._building_edge_loss(
                pred_hat, edge_target, building_prob, loss_weight
            )
            edge_loss_w = float(
                self.landcover_loss_args.get('building_edge_loss_w', 0.05)
            )
            building_edge_loss = edge_loss_w * edge_loss

        spatial_gradient_loss = self._spatial_gradient_loss(
            pred_hat, edge_target, loss_weight
        ) * float(self.restoration_loss_args.get('spatial_gradient_loss_w', 0.0))
        spectral_angle_loss = self._spectral_angle_loss(
            pred_hat, edge_target, loss_weight
        ) * float(self.restoration_loss_args.get('spectral_angle_loss_w', 0.0))
        temporal_difference_loss = self._temporal_difference_loss(
            pred_hat, edge_target, loss_weight, date_dense_target
        ) * float(self.restoration_loss_args.get('temporal_difference_loss_w', 0.0))

        loss = (
            reconstruction_loss
            + reliability_loss
            + building_edge_loss
            + spatial_gradient_loss
            + spectral_angle_loss
            + temporal_difference_loss
        )
        loss_dict = {
            'reconstruction_loss': reconstruction_loss.detach().item(),
            'reliability_loss': reliability_loss.detach().item(),
            'building_edge_loss': building_edge_loss.detach().item(),
            'spatial_gradient_loss': spatial_gradient_loss.detach().item(),
            'spectral_angle_loss': spectral_angle_loss.detach().item(),
            'temporal_difference_loss': temporal_difference_loss.detach().item(),
            'total_loss': loss.detach().item(),
        }
        return loss, loss_dict
