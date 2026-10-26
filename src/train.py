# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

import copy
from hashlib import sha256
import json
import logging
import math
from pathlib import Path
import sys
import yaml

import numpy as np

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from src.masks.multiblock import MaskCollator as MBMaskCollator
from src.masks.utils import apply_masks
from src.utils.distributed import (
    init_distributed,
    AllReduce
)
from src.utils.logging import (
    CSVLogger,
    gpu_timer,
    grad_logger,
    AverageMeter)
from src.utils.tensors import repeat_interleave_batch
from src.helper import (
    load_checkpoint,
    init_model,
    init_opt,
    initialize_from_released_checkpoint)
from src.datasets.wall_frames import WallFrames, build_frame_index
from src.mylonas_ijepa_wm.wall_layout_manifest import load_manifest

# --
log_timings = True
log_freq = 10
checkpoint_freq = 50
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def make_imagenet1k(**kwargs):
    # Keep torchvision optional for Wall-only runs and local tests.
    from src.datasets.imagenet1k import make_imagenet1k as factory
    return factory(**kwargs)


def make_transforms(**kwargs):
    from src.transforms import make_transforms as factory
    return factory(**kwargs)


def jepa_loss(images, masks_enc, masks_pred, encoder, predictor, target_encoder):
    """The original I-JEPA within-image masked prediction objective."""
    with torch.no_grad():
        targets = target_encoder(images)
        targets = F.layer_norm(targets, (targets.size(-1),))
        batch_size = len(targets)
        targets = apply_masks(targets, masks_pred)
        targets = repeat_interleave_batch(targets, batch_size, repeat=len(masks_enc))
    predictions = predictor(encoder(images, masks_enc), masks_enc, masks_pred)
    return F.smooth_l1_loss(predictions, targets)


def _unwrapped(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def _json_hash(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _wall_config_identity(args):
    settings = copy.deepcopy(args)
    settings["meta"].pop("load_checkpoint", None)
    settings["meta"].pop("read_checkpoint", None)
    return _json_hash(settings)


def _atomic_torch_save(value, path):
    temporary = Path(str(path) + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def evaluate_wall_loss(dataset, batch_size, num_workers, pin_mem, mask_settings,
                       encoder, predictor, target_encoder, device, seed=12345):
    """Evaluate the same fixed masks and frame order every epoch."""
    collator = MBMaskCollator(**mask_settings)
    loader = DataLoader(
        dataset, sampler=SequentialSampler(dataset), batch_size=batch_size,
        # MaskCollator uses a shared counter; main-process collation fixes mask order.
        drop_last=False, collate_fn=collator, num_workers=0,
        pin_memory=pin_mem,
    )
    losses = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        with torch.no_grad():
            for (udata, masks_enc, masks_pred) in loader:
                images = udata[0].to(device)
                enc = [mask.to(device) for mask in masks_enc]
                pred = [mask.to(device) for mask in masks_pred]
                losses.append(float(jepa_loss(images, enc, pred, encoder, predictor, target_encoder)))
    if not losses:
        raise ValueError("Wall validation has no frames")
    return sum(losses) / len(losses)


def main(args, resume_preempt=False):

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    is_wall = args['data'].get('dataset', 'imagenet') == 'wall'
    if args['data'].get('dataset', 'imagenet') not in ('wall', 'imagenet'):
        raise ValueError("data.dataset must be wall or imagenet")
    use_bfloat16 = args['meta']['use_bfloat16']
    model_name = args['meta']['model_name']
    load_model = args['meta']['load_checkpoint'] or resume_preempt
    r_file = args['meta']['read_checkpoint']
    copy_data = args['meta']['copy_data']
    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # -- DATA
    use_gaussian_blur = args['data']['use_gaussian_blur']
    use_horizontal_flip = args['data']['use_horizontal_flip']
    use_color_distortion = args['data']['use_color_distortion']
    color_jitter = args['data']['color_jitter_strength']
    # --
    batch_size = args['data']['batch_size']
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    if is_wall and num_workers != 0:
        raise ValueError("Wall fine-tuning requires data.num_workers: 0 for reproducible masks")
    root_path = args['data'].get('root_path')
    image_folder = args['data'].get('image_folder')
    crop_size = args['data']['crop_size']
    crop_scale = args['data']['crop_scale']
    # --

    # -- MASK
    allow_overlap = args['mask']['allow_overlap']  # whether to allow overlap b/w context and target blocks
    patch_size = args['mask']['patch_size']  # patch-size for model training
    num_enc_masks = args['mask']['num_enc_masks']  # number of context blocks
    min_keep = args['mask']['min_keep']  # min number of patches in context block
    enc_mask_scale = args['mask']['enc_mask_scale']  # scale of context blocks
    num_pred_masks = args['mask']['num_pred_masks']  # number of target blocks
    pred_mask_scale = args['mask']['pred_mask_scale']  # scale of target blocks
    aspect_ratio = args['mask']['aspect_ratio']  # aspect ratio of target blocks
    # --

    # -- OPTIMIZATION
    ema = args['optimization']['ema']
    ipe_scale = args['optimization']['ipe_scale']  # scheduler scale factor (def: 1.0)
    wd = float(args['optimization']['weight_decay'])
    final_wd = float(args['optimization']['final_weight_decay'])
    num_epochs = args['optimization']['epochs']
    warmup = args['optimization']['warmup']
    start_lr = args['optimization']['start_lr']
    lr = args['optimization']['lr']
    final_lr = args['optimization']['final_lr']

    # -- LOGGING
    folder = args['logging']['folder']
    tag = args['logging']['write_tag']
    Path(folder).mkdir(parents=True, exist_ok=True)

    dump = os.path.join(folder, 'params-ijepa.yaml')
    with open(dump, 'w') as f:
        yaml.dump(args, f)
    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')
    if rank > 0:
        logger.setLevel(logging.ERROR)

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f'{tag}_r{rank}.csv')
    save_path = os.path.join(folder, f'{tag}' + '-ep{epoch}.pth.tar')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    load_path = None
    if load_model:
        load_path = os.path.join(folder, r_file) if r_file is not None else latest_path

    # -- make csv_logger
    csv_logger = CSVLogger(log_file,
                           ('%d', 'epoch'),
                           ('%d', 'itr'),
                           ('%.5f', 'loss'),
                           ('%.5f', 'mask-A'),
                           ('%.5f', 'mask-B'),
                           ('%d', 'time (ms)'))

    # -- init model
    encoder, predictor = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name)
    target_encoder = copy.deepcopy(encoder)
    source_identity = None
    if is_wall:
        source_path = args['meta'].get('init_checkpoint')
        if not source_path:
            raise ValueError("Wall fine-tuning requires meta.init_checkpoint")
        source_identity = initialize_from_released_checkpoint(
            source_path, encoder, predictor, target_encoder)

    # -- make data transforms
    mask_settings = dict(
        input_size=crop_size, patch_size=patch_size,
        pred_mask_scale=pred_mask_scale, enc_mask_scale=enc_mask_scale,
        aspect_ratio=aspect_ratio, nenc=num_enc_masks, npred=num_pred_masks,
        allow_overlap=allow_overlap, min_keep=min_keep,
    )
    mask_collator = MBMaskCollator(**mask_settings)

    wall_index = None
    val_dataset = None
    if is_wall:
        raw_dir = Path(args['data']['raw_dir'])
        wall_index = build_frame_index(
            raw_dir, load_manifest(Path(args['data']['episode_manifest'])),
            frame_stride=int(args['data'].get('frame_stride', 5)),
        )
        train_dataset = WallFrames(raw_dir, wall_index, 'train')
        val_dataset = WallFrames(raw_dir, wall_index, 'val')
        unsupervised_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True,
            seed=int(args['meta'].get('seed', 0)),
        )
        unsupervised_loader = DataLoader(
            train_dataset, sampler=unsupervised_sampler, batch_size=batch_size,
            collate_fn=mask_collator, pin_memory=pin_mem,
            num_workers=num_workers, drop_last=True,
        )
        if rank == 0:
            Path(folder, 'frame_index.json').write_text(json.dumps(wall_index, indent=2))
    else:
        transform = make_transforms(
            crop_size=crop_size, crop_scale=crop_scale,
            gaussian_blur=use_gaussian_blur, horizontal_flip=use_horizontal_flip,
            color_distortion=use_color_distortion, color_jitter=color_jitter)
        _, unsupervised_loader, unsupervised_sampler = make_imagenet1k(
                transform=transform, batch_size=batch_size, collator=mask_collator,
                pin_mem=pin_mem, training=True, num_workers=num_workers,
                world_size=world_size, rank=rank, root_path=root_path,
                image_folder=image_folder, copy_data=copy_data, drop_last=True)
    accumulation_steps = int(args['optimization'].get('accumulation_steps', 1)) if is_wall else 1
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be positive")
    ipe = math.ceil(len(unsupervised_loader) / accumulation_steps)
    if not len(unsupervised_loader):
        raise ValueError("I-JEPA training loader has no batches")

    # -- init optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        use_bfloat16=use_bfloat16)
    if dist.is_available() and dist.is_initialized():
        encoder = DistributedDataParallel(encoder, static_graph=True)
        predictor = DistributedDataParallel(predictor, static_graph=True)
        target_encoder = DistributedDataParallel(target_encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False

    # -- momentum schedule
    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale)
                          for i in range(int(ipe*num_epochs*ipe_scale)+1))

    start_epoch = 0
    best_val_loss = float('inf')
    history = []
    optimizer_updates = 0
    wall_identity = None
    if is_wall:
        wall_identity = {
            'manifest_sha256': wall_index['manifest_sha256'],
            'source_checkpoint_sha256': source_identity['sha256'],
            'frame_index_sha256': _json_hash(wall_index),
            'config_sha256': _wall_config_identity(args),
            'frame_stride': wall_index['frame_stride'],
        }
    # -- load training checkpoint
    if load_model:
        if is_wall:
            checkpoint = torch.load(load_path, map_location='cpu', weights_only=True)
            if checkpoint.get('wall_run_identity') != wall_identity:
                raise ValueError('Wall run identity differs from saved checkpoint')
            for name, model in (
                ('encoder', encoder), ('predictor', predictor),
                ('target_encoder', target_encoder),
            ):
                _unwrapped(model).load_state_dict(checkpoint[name], strict=True)
            optimizer.load_state_dict(checkpoint['opt'])
            if scaler is not None and checkpoint.get('scaler') is not None:
                scaler.load_state_dict(checkpoint['scaler'])
            start_epoch = int(checkpoint['epoch'])
            optimizer_updates = int(checkpoint['optimizer_updates'])
            best_val_loss = float(checkpoint['best_val_loss'])
            history = list(checkpoint['history'])
            scheduler._step = optimizer_updates
            wd_scheduler._step = optimizer_updates
        else:
            encoder, predictor, target_encoder, optimizer, scaler, start_epoch = load_checkpoint(
                device=device,
                r_path=load_path,
                encoder=encoder,
                predictor=predictor,
                target_encoder=target_encoder,
                opt=optimizer,
                scaler=scaler)
            for _ in range(start_epoch*ipe):
                scheduler.step()
                wd_scheduler.step()
                next(momentum_scheduler)
                mask_collator.step()

    def save_checkpoint(epoch):
        save_dict = {
            'encoder': encoder.state_dict(),
            'predictor': predictor.state_dict(),
            'target_encoder': target_encoder.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'loss': loss_meter.avg,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr
        }
        if rank == 0:
            torch.save(save_dict, latest_path)
            if (epoch + 1) % checkpoint_freq == 0:
                torch.save(save_dict, save_path.format(epoch=f'{epoch + 1}'))

    def save_wall_checkpoint(epoch, train_loss, val_loss, improved):
        if rank != 0:
            return
        save_dict = {
            'encoder': _unwrapped(encoder).state_dict(),
            'predictor': _unwrapped(predictor).state_dict(),
            'target_encoder': _unwrapped(target_encoder).state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'optimizer_updates': optimizer_updates,
            'wall_run_identity': wall_identity,
            'history': history,
        }
        _atomic_torch_save(save_dict, latest_path)
        if improved:
            _atomic_torch_save(save_dict, Path(folder) / f'{tag}-best.pth.tar')
        temporary = Path(folder) / 'history.json.tmp'
        temporary.write_text(json.dumps(history, indent=2))
        temporary.replace(Path(folder) / 'history.json')

    # -- TRAINING LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch %d' % (epoch + 1))

        # -- update distributed-data-loader epoch
        unsupervised_sampler.set_epoch(epoch)
        if is_wall:
            torch.manual_seed(int(args['meta'].get('seed', 0)) + epoch)
            if hasattr(mask_collator, '_itr_counter'):
                with mask_collator._itr_counter.get_lock():
                    mask_collator._itr_counter.value = epoch * len(unsupervised_loader) - 1
        encoder.train()
        predictor.train()
        target_encoder.eval()

        loss_meter = AverageMeter()
        maskA_meter = AverageMeter()
        maskB_meter = AverageMeter()
        time_meter = AverageMeter()

        for itr, (udata, masks_enc, masks_pred) in enumerate(unsupervised_loader):

            def load_imgs():
                # -- unsupervised imgs
                imgs = udata[0].to(device, non_blocking=True)
                masks_1 = [u.to(device, non_blocking=True) for u in masks_enc]
                masks_2 = [u.to(device, non_blocking=True) for u in masks_pred]
                return (imgs, masks_1, masks_2)
            imgs, masks_enc, masks_pred = load_imgs()
            maskA_meter.update(len(masks_enc[0][0]))
            maskB_meter.update(len(masks_pred[0][0]))

            def train_step():
                nonlocal optimizer_updates
                group_start = (itr // accumulation_steps) * accumulation_steps
                group_size = min(accumulation_steps, len(unsupervised_loader) - group_start)
                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16,
                    enabled=use_bfloat16 and device.type == 'cuda',
                ):
                    loss = AllReduce.apply(jepa_loss(
                        imgs, masks_enc, masks_pred, encoder, predictor, target_encoder))
                scaled_loss = loss / group_size
                if scaler is not None and device.type == 'cuda':
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                _new_lr = optimizer.param_groups[0]['lr']
                _new_wd = optimizer.param_groups[0]['weight_decay']
                grad_stats = None
                update_now = ((itr + 1) % accumulation_steps == 0
                              or itr + 1 == len(unsupervised_loader))
                if update_now:
                    _new_lr = scheduler.step()
                    _new_wd = wd_scheduler.step()
                    grad_stats = grad_logger(encoder.named_parameters())
                    if scaler is not None and device.type == 'cuda':
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                    with torch.no_grad():
                        if is_wall:
                            total_updates = max(1, ipe * num_epochs)
                            m = ema[0] + (ema[1] - ema[0]) * min(
                                optimizer_updates / total_updates, 1.0)
                        else:
                            m = next(momentum_scheduler)
                        for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                            param_k.data.mul_(m).add_((1. - m) * param_q.detach().data)
                    optimizer_updates += 1
                return (float(loss.detach()), _new_lr, _new_wd, grad_stats)
            (loss, _new_lr, _new_wd, grad_stats), etime = gpu_timer(train_step)
            loss_meter.update(loss)
            time_meter.update(etime)

            # -- Logging
            def log_stats():
                csv_logger.log(epoch + 1, itr, loss, maskA_meter.val, maskB_meter.val, etime)
                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    logger.info('[%d, %5d] loss: %.3f '
                                'masks: %.1f %.1f '
                                '[wd: %.2e] [lr: %.2e] '
                                '[mem: %.2e] '
                                '(%.1f ms)'
                                % (epoch + 1, itr,
                                   loss_meter.avg,
                                   maskA_meter.avg,
                                   maskB_meter.avg,
                                   _new_wd,
                                   _new_lr,
                                   torch.cuda.max_memory_allocated() / 1024.**2,
                                   time_meter.avg))

                    if grad_stats is not None:
                        logger.info('[%d, %5d] grad_stats: [%.2e %.2e] (%.2e, %.2e)'
                                    % (epoch + 1, itr,
                                       grad_stats.first_layer,
                                       grad_stats.last_layer,
                                       grad_stats.min,
                                       grad_stats.max))

            log_stats()

            assert not np.isnan(loss), 'loss is nan'

        # -- Save Checkpoint after every epoch
        logger.info('avg. loss %.3f' % loss_meter.avg)
        if is_wall:
            encoder.eval()
            predictor.eval()
            target_encoder.eval()
            val_loss = evaluate_wall_loss(
                val_dataset, batch_size, num_workers, pin_mem, mask_settings,
                encoder, predictor, target_encoder, device,
                seed=int(args['meta'].get('validation_mask_seed', 12345)),
            )
            if not np.isfinite(val_loss):
                raise ValueError('Wall validation loss is not finite')
            improved = val_loss < best_val_loss
            best_val_loss = min(best_val_loss, val_loss)
            history.append({'epoch': epoch + 1, 'train_loss': loss_meter.avg,
                            'val_loss': val_loss, 'optimizer_updates': optimizer_updates})
            logger.info('Wall validation loss %.5f (best %.5f)', val_loss, best_val_loss)
            save_wall_checkpoint(epoch + 1, loss_meter.avg, val_loss, improved)
        else:
            save_checkpoint(epoch+1)


if __name__ == "__main__":
    main()
