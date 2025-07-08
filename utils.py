# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import torch
import torch.distributed as dist
import numpy as np
import re
import requests
try:
    from torch._six import inf
except:
    from torch import inf


def load_checkpoint(config, model, optimizer, lr_scheduler, loss_scaler, logger):
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    max_accuracy = 0.0
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        if 'scaler' in checkpoint:
            loss_scaler.load_state_dict(checkpoint['scaler'])
        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy

def load_pretrained(config, model, logger):
    logger.info(f"==============> Loading weight {config.MODEL.PRETRAINED} for fine-tuning......")
    checkpoint = torch.load(config.MODEL.PRETRAINED, map_location='cpu')
    state_dict = checkpoint['model']

    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete relative_coords_table since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del state_dict[k]

    # bicubic interpolate relative_position_bias_table if not match
    relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
    for k in relative_position_bias_table_keys:
        relative_position_bias_table_pretrained = state_dict[k]
        relative_position_bias_table_current = model.state_dict()[k]
        L1, nH1 = relative_position_bias_table_pretrained.size()
        L2, nH2 = relative_position_bias_table_current.size()
        if nH1 != nH2:
            logger.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                # bicubic interpolate relative_position_bias_table if not match
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                    relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                    mode='bicubic')
                state_dict[k] = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)

    # bicubic interpolate absolute_pos_embed if not match
    absolute_pos_embed_keys = [k for k in state_dict.keys() if "absolute_pos_embed" in k]
    for k in absolute_pos_embed_keys:
        # dpe
        absolute_pos_embed_pretrained = state_dict[k]
        absolute_pos_embed_current = model.state_dict()[k]
        _, L1, C1 = absolute_pos_embed_pretrained.size()
        _, L2, C2 = absolute_pos_embed_current.size()
        if C1 != C1:
            logger.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
                absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                    absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(1, 2)
                state_dict[k] = absolute_pos_embed_pretrained_resized

    # check classifier, if not match, then re-init classifier to zero
    head_bias_pretrained = state_dict['head.bias']
    Nc1 = head_bias_pretrained.shape[0]
    Nc2 = model.head.bias.shape[0]
    if (Nc1 != Nc2):
        if Nc1 == 21841 and Nc2 == 1000:
            logger.info("loading ImageNet-22K weight to ImageNet-1K ......")
            map22kto1k_path = f'data/map22kto1k.txt'
            with open(map22kto1k_path) as f:
                map22kto1k = f.readlines()
            map22kto1k = [int(id22k.strip()) for id22k in map22kto1k]
            state_dict['head.weight'] = state_dict['head.weight'][map22kto1k, :]
            state_dict['head.bias'] = state_dict['head.bias'][map22kto1k]
        else:
            torch.nn.init.constant_(model.head.bias, 0.)
            torch.nn.init.constant_(model.head.weight, 0.)
            del state_dict['head.weight']
            del state_dict['head.bias']
            logger.warning(f"Error in loading classifier head, re-init classifier head to 0")

    msg = model.load_state_dict(state_dict, strict=False)
    logger.warning(msg)

    logger.info(f"=> loaded successfully '{config.MODEL.PRETRAINED}'")

    del checkpoint
    torch.cuda.empty_cache()


def load_pretrained_hjbface_agri17k(config, model, logger):
    if config.MODEL.TYPE is not 'swin':
        checkpoint = torch.load(config.MODEL.PRETRAINED, map_location='cpu')
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        del checkpoint
    else:
        state_dict = load_swin_func(config, model, logger)

    if state_dict is not None:
        state_dict = init_classifier_head(state_dict=state_dict, model_name=config.MODEL.TYPE, model=model)
        logger.warning(f"Transfering in loading classifier head, re-init classifier head to std=0.02")

    msg = model.load_state_dict(state_dict, strict=False)
    logger.warning(msg)

    logger.info(f"=> loaded successfully '{config.MODEL.PRETRAINED}'")

    torch.cuda.empty_cache()


def load_swin_func(config, model, logger):
    checkpoint = torch.load(config.MODEL.PRETRAINED, map_location='cpu')
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete relative_coords_table since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del state_dict[k]

    # bicubic interpolate relative_position_bias_table if not match
    relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
    for k in relative_position_bias_table_keys:
        relative_position_bias_table_pretrained = state_dict[k]
        relative_position_bias_table_current = model.state_dict()[k]
        L1, nH1 = relative_position_bias_table_pretrained.size()
        L2, nH2 = relative_position_bias_table_current.size()
        if nH1 != nH2:
            logger.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                # bicubic interpolate relative_position_bias_table if not match
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                    relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                    mode='bicubic')
                state_dict[k] = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)

    # bicubic interpolate absolute_pos_embed if not match
    absolute_pos_embed_keys = [k for k in state_dict.keys() if "absolute_pos_embed" in k]
    for k in absolute_pos_embed_keys:
        # dpe
        absolute_pos_embed_pretrained = state_dict[k]
        absolute_pos_embed_current = model.state_dict()[k]
        _, L1, C1 = absolute_pos_embed_pretrained.size()
        _, L2, C2 = absolute_pos_embed_current.size()
        if C1 != C1:
            logger.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
                absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                    absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(1, 2)
                state_dict[k] = absolute_pos_embed_pretrained_resized
    
    return state_dict

def init_classifier_head(state_dict, model_name, model):
    if model_name == 'resnet':
        torch.nn.init.constant_(model.fc.bias, 0.) if hasattr(model.fc, 'bias') else None
        torch.nn.init.normal_(model.fc.weight, 0.02)
        del state_dict['fc.weight']
        del state_dict['fc.bias']
    elif model_name == 'convnext':
        torch.nn.init.constant_(model.head.bias, 0.) if hasattr(model.head, 'bias') else None
        torch.nn.init.normal_(model.head.weight, 0.02)
        del state_dict['head.weight']
        del state_dict['head.bias']

    elif model_name == 'vit': 
        torch.nn.init.constant_(model.fc.bias, 0.) if hasattr(model.fc, 'bias') else None
        torch.nn.init.normal_(model.fc.weight, 0.02)
        del state_dict['fc.weight']
        del state_dict['fc.bias']
    elif model_name == 'vgg19': 
        torch.nn.init.constant_(model.classifier[6].bias, 0.) if hasattr(model.classifier[6], 'bias') else None
        torch.nn.init.normal_(model.classifier[6].weight, 0.02)
        del state_dict['classifier.6.weight']
        del state_dict['classifier.6.bias']
    elif model_name == 'vgg16':
        torch.nn.init.constant_(model.fc2.bias, 0.) if hasattr(model.fc2, 'bias') else None
        torch.nn.init.normal_(model.fc2.weight, 0.02)
        del state_dict['fc2.weight']
        del state_dict['fc2.bias']

    elif model_name == 'mobilenetv3_l': 
        torch.nn.init.constant_(model.linear4.bias, 0.) if hasattr(model.linear4, 'bias') else None
        torch.nn.init.normal_(model.linear4.weight, 0.02)
        del state_dict['linear4.weight']
        del state_dict['linear4.bias']
    elif model_name == 'mobilenetv3_s': 
        torch.nn.init.constant_(model.linear4.bias, 0.) if hasattr(model.linear4, 'bias') else None
        torch.nn.init.normal_(model.linear4.weight, 0.02)
        del state_dict['linear4.weight']
        del state_dict['linear4.bias']

    elif model_name == 'mobilenetv2': 
        torch.nn.init.constant_(model.classifier.bias, 0.) if hasattr(model.classifier, 'bias') else None
        torch.nn.init.normal_(model.classifier.weight, 0.02)
        del state_dict['classifier.weight']
        del state_dict['classifier.bias']  

    elif model_name == 'xception': 
        torch.nn.init.constant_(model.fc.bias, 0.) if hasattr(model.fc, 'bias') else None
        torch.nn.init.normal_(model.fc.weight, 0.02)
        del state_dict['fc.weight']
        del state_dict['fc.bias']  
    elif model_name == 'mobilevit': 
        torch.nn.init.constant_(model.fc.bias, 0.) if hasattr(model.fc, 'bias') else None
        torch.nn.init.normal_(model.fc.weight, 0.02)
        del state_dict['fc.weight']
        del state_dict['fc.bias']  
    elif model_name == 'pvt': 
        torch.nn.init.constant_(model.head.bias, 0.) if hasattr(model.head, 'bias') else None
        torch.nn.init.normal_(model.head.weight, 0.02)
        del state_dict['head.weight']
        del state_dict['head.bias']

    elif model_name == 'efficientnet_v2': 
        torch.nn.init.constant_(model.head.classifier.bias, 0.) if hasattr(model.head.classifier, 'bias') else None
        torch.nn.init.normal_(model.head.classifier.weight, 0.02)
        del state_dict['head.classifier.weight']
        del state_dict['head.classifier.bias']

    elif model_name == 'densenet201':
        torch.nn.init.constant_(model.classifier.bias, 0.) if hasattr(model.classifier.bias, 'bias') else None
        torch.nn.init.normal_(model.classifier.weight, 0.02)
        del state_dict['classifier.weight']
        del state_dict['classifier.bias']


    else:
        torch.nn.init.constant_(model.head.bias, 0.)
        torch.nn.init.normal_(model.head.weight, .02)
        del state_dict['head.weight']
        del state_dict['head.bias']
    return state_dict



def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, loss_scaler, logger):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'scaler': loss_scaler.state_dict(),
                  'epoch': epoch,
                  'config': config}

    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")


def save_checkpoint_best(config, epoch, model, max_accuracy, optimizer, lr_scheduler, loss_scaler, logger):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'scaler': loss_scaler.state_dict(),
                  'epoch': epoch,
                  'config': config}

    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_best.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(),
                                                        norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = ampscaler_get_grad_norm(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)



# ------------------------change vit position embedding function -----------------------

def resize_positional_embedding_(posemb, posemb_new, has_class_token=True):
    """Rescale the grid of position embeddings in a sensible manner"""
    from scipy.ndimage import zoom

    # Deal with class token
    ntok_new = posemb_new.shape[1]
    if has_class_token:  # this means classifier == 'token'
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
        ntok_new -= 1
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

    # Get old and new grid sizes
    gs_old = int(np.sqrt(len(posemb_grid)))
    gs_new = int(np.sqrt(ntok_new))
    posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

    # Rescale grid
    zoom_factor = (gs_new / gs_old, gs_new / gs_old, 1)
    posemb_grid = zoom(posemb_grid, zoom_factor, order=1)
    posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
    posemb_grid = torch.from_numpy(posemb_grid)

    # Deal with class token and return
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb

# ----------------------------load pretrained densenet function ---------------------
def densenet_load_state_dict(weights):
    # '.'s are no longer allowed in module names, but previous _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$"
    )

    state_dict = weights
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    return state_dict


# ------------------------load pretrained efficientnetv2 function-----------------------

def load_from_path(model, pretrained_path):
    load_npy(model, np.load(pretrained_path, allow_pickle=True).item())


def load_npy_from_url(url, file_name):
    if not Path(file_name).exists():
        response = requests.get(url)
        with open(file_name, 'wb') as f:
            f.write(response.content)
    return np.load(file_name, allow_pickle=True).item()



def npz_dim_convertor(name, weight):
    weight = torch.from_numpy(weight)
    if 'kernel' in name:
        if weight.dim() == 4:
            if weight.shape[3] == 1:
                # depth-wise convolution 'h w in_c out_c -> in_c out_c h w'
                weight = torch.permute(weight, (2, 3, 0, 1))
            else:
                # 'h w in_c out_c -> out_c in_c h w'
                weight = torch.permute(weight, (3, 2, 0, 1))
        elif weight.dim() == 2:
            weight = weight.transpose(1, 0)
    elif 'scale' in name or 'bias' in name:
        weight = weight.squeeze()
    return weight


def load_npy(model, weight):
    name_convertor = [
        # stem
        ('stem.0.weight', 'stem/conv2d/kernel/ExponentialMovingAverage'),
        ('stem.1.weight', 'stem/tpu_batch_normalization/gamma/ExponentialMovingAverage'),
        ('stem.1.bias', 'stem/tpu_batch_normalization/beta/ExponentialMovingAverage'),
        ('stem.1.running_mean', 'stem/tpu_batch_normalization/moving_mean/ExponentialMovingAverage'),
        ('stem.1.running_var', 'stem/tpu_batch_normalization/moving_variance/ExponentialMovingAverage'),

        # fused layer
        ('block.fused.0.weight', 'conv2d/kernel/ExponentialMovingAverage'),
        ('block.fused.1.weight', 'tpu_batch_normalization/gamma/ExponentialMovingAverage'),
        ('block.fused.1.bias', 'tpu_batch_normalization/beta/ExponentialMovingAverage'),
        ('block.fused.1.running_mean', 'tpu_batch_normalization/moving_mean/ExponentialMovingAverage'),
        ('block.fused.1.running_var', 'tpu_batch_normalization/moving_variance/ExponentialMovingAverage'),

        # linear bottleneck
        ('block.linear_bottleneck.0.weight', 'conv2d/kernel/ExponentialMovingAverage'),
        ('block.linear_bottleneck.1.weight', 'tpu_batch_normalization/gamma/ExponentialMovingAverage'),
        ('block.linear_bottleneck.1.bias', 'tpu_batch_normalization/beta/ExponentialMovingAverage'),
        ('block.linear_bottleneck.1.running_mean', 'tpu_batch_normalization/moving_mean/ExponentialMovingAverage'),
        ('block.linear_bottleneck.1.running_var', 'tpu_batch_normalization/moving_variance/ExponentialMovingAverage'),

        # depth wise layer
        ('block.depth_wise.0.weight', 'depthwise_conv2d/depthwise_kernel/ExponentialMovingAverage'),
        ('block.depth_wise.1.weight', 'tpu_batch_normalization_1/gamma/ExponentialMovingAverage'),
        ('block.depth_wise.1.bias', 'tpu_batch_normalization_1/beta/ExponentialMovingAverage'),
        ('block.depth_wise.1.running_mean', 'tpu_batch_normalization_1/moving_mean/ExponentialMovingAverage'),
        ('block.depth_wise.1.running_var', 'tpu_batch_normalization_1/moving_variance/ExponentialMovingAverage'),

        # se layer
        ('block.se.fc1.weight', 'se/conv2d/kernel/ExponentialMovingAverage'), ('block.se.fc1.bias', 'se/conv2d/bias/ExponentialMovingAverage'),
        ('block.se.fc2.weight', 'se/conv2d_1/kernel/ExponentialMovingAverage'), ('block.se.fc2.bias', 'se/conv2d_1/bias/ExponentialMovingAverage'),

        # point wise layer
        ('block.fused_point_wise.0.weight', 'conv2d_1/kernel/ExponentialMovingAverage'),
        ('block.fused_point_wise.1.weight', 'tpu_batch_normalization_1/gamma/ExponentialMovingAverage'),
        ('block.fused_point_wise.1.bias', 'tpu_batch_normalization_1/beta/ExponentialMovingAverage'),
        ('block.fused_point_wise.1.running_mean', 'tpu_batch_normalization_1/moving_mean/ExponentialMovingAverage'),
        ('block.fused_point_wise.1.running_var', 'tpu_batch_normalization_1/moving_variance/ExponentialMovingAverage'),

        ('block.point_wise.0.weight', 'conv2d_1/kernel/ExponentialMovingAverage'),
        ('block.point_wise.1.weight', 'tpu_batch_normalization_2/gamma/ExponentialMovingAverage'),
        ('block.point_wise.1.bias', 'tpu_batch_normalization_2/beta/ExponentialMovingAverage'),
        ('block.point_wise.1.running_mean', 'tpu_batch_normalization_2/moving_mean/ExponentialMovingAverage'),
        ('block.point_wise.1.running_var', 'tpu_batch_normalization_2/moving_variance/ExponentialMovingAverage'),

        # head
        ('head.bottleneck.0.weight', 'head/conv2d/kernel/ExponentialMovingAverage'),
        ('head.bottleneck.1.weight', 'head/tpu_batch_normalization/gamma/ExponentialMovingAverage'),
        ('head.bottleneck.1.bias', 'head/tpu_batch_normalization/beta/ExponentialMovingAverage'),
        ('head.bottleneck.1.running_mean', 'head/tpu_batch_normalization/moving_mean/ExponentialMovingAverage'),
        ('head.bottleneck.1.running_var', 'head/tpu_batch_normalization/moving_variance/ExponentialMovingAverage'),

        # classifier
        ('head.classifier.weight', 'head/dense/kernel/ExponentialMovingAverage'),
        ('head.classifier.bias', 'head/dense/bias/ExponentialMovingAverage'),

        ('\\.(\\d+)\\.', lambda x: f'_{int(x.group(1))}/'),
    ]

    for name, param in list(model.named_parameters()) + list(model.named_buffers()):
        for pattern, sub in name_convertor:
            name = re.sub(pattern, sub, name)
        if 'dense/kernel' in name and list(param.shape) not in [[1000, 1280], [21843, 1280]]:
            continue
        if 'dense/bias' in name and list(param.shape) not in [[1000], [21843]]:
            continue
        if 'num_batches_tracked' in name:
            continue
        param.data.copy_(npz_dim_convertor(name, weight.get(name)))

