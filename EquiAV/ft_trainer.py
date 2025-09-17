#!/usr/bin/python
#-*- coding: utf-8 -*-
import os
import sys
import time
import wandb
import importlib
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from utils import *

class WrappedModel(nn.Module):
    ## The purpose of this wrapper is to make the model structure consistent between single and multi-GPU
    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.module = model

    def forward(self, data_a, data_v, labels=None):
        return self.module(data_a, data_v, labels)

class EquiAV_ft(nn.Module):
    def __init__(self, trainfunc_ft, model, **kwargs):
        super(EquiAV_ft, self).__init__();

        self.model = model

        EquiAV_Model = importlib.import_module('models.'+model).__getattribute__('MainModel')
        self.__M__ = EquiAV_Model(**kwargs);

        LossFunction = importlib.import_module('loss.'+trainfunc_ft).__getattribute__('LossFunction')
        self.__L__ = LossFunction(**kwargs)

    def forward(self, data_a, data_v, labels=None):
        
        output_ft = self.__M__.forward(data_a, data_v)
        loss_ft = self.__L__.forward(output_ft, labels)

        loss_dict = {'loss': loss_ft}
        outputs = {'output': output_ft}

        return loss_dict, outputs


class ModelTrainer(nn.Module):
    def __init__(self, equiAV, gpu, optimizer='adamw', mixedprec=True, freeze_base=False, no_wandb=False, **kwargs):
        super(ModelTrainer, self).__init__()

        self.__model__  = equiAV
        self.gpu = gpu

        if kwargs['device'] == "cpu":
            self.device = torch.device('cpu')
            self.device_name = "cpu"
        else:
            self.device = torch.device('cuda')
            self.device_name = "cuda"

        self.mixedprec = mixedprec
        self.freeze_base = freeze_base

        self.max_epoch = kwargs['max_epoch']
        
        self.lr = kwargs['lr']
        self.head_lr = kwargs['head_lr']
        self.scheduler = kwargs['scheduler']
        self.warmup_epoch = kwargs['warmup_epoch']
        self.ipe = kwargs['iteration_per_epoch']
        self.ipe_scale = 1.0

        self.model_name = kwargs['model']
        
        # possible mlp layer name list, mlp layers are newly initialized layers in the finetuning stage (i.e., not pretrained) and should use a larger lr during finetuning
        mlp_list = ['__M__.mlp_head.0.weight', '__M__.mlp_head.0.bias', '__M__.mlp_head.1.weight', '__M__.mlp_head.1.bias',
                    '__M__.mlp_head2.0.weight', '__M__.mlp_head2.0.bias', '__M__.mlp_head2.1.weight', '__M__.mlp_head2.1.bias',
                    '__M__.mlp_head_a.0.weight', '__M__.mlp_head_a.0.bias', '__M__.mlp_head_a.1.weight', '__M__.mlp_head_a.1.bias',
                    '__M__.mlp_head_v.0.weight', '__M__.mlp_head_v.0.bias', '__M__.mlp_head_v.1.weight', '__M__.mlp_head_v.1.bias',
                    '__M__.mlp_head_concat.0.weight', '__M__.mlp_head_concat.0.bias', '__M__.mlp_head_concat.1.weight', '__M__.mlp_head_concat.1.bias']

        mlp_params = list(filter(lambda kv: kv[0] in mlp_list, self.__model__.module.named_parameters()))
        base_params = list(filter(lambda kv: kv[0] not in mlp_list, self.__model__.module.named_parameters()))
        
        mlp_params = [i[1] for i in mlp_params]
        base_params = [i[1] for i in base_params]

        if self.freeze_base == True:
            if self.gpu == 0: print('Pretrained backbone parameters are frozen.')
            for param in base_params:
                param.requires_grad = False

        trainables = [p for p in self.__model__.parameters() if p.requires_grad]

        Optimizer = importlib.import_module('optimizer.'+optimizer).__getattribute__('Optimizer')
        self.__optimizer__ = Optimizer([{'params': base_params, 'lr': self.lr}, {'params': mlp_params, 'lr': self.lr * self.head_lr}], gpu=gpu, **kwargs)

        base_lr = self.__optimizer__.param_groups[0]['lr']
        mlp_lr = self.__optimizer__.param_groups[1]['lr']

        if self.gpu == 0:
            print('Total pretrained backbone parameter number is : {:.3f} million\n'.format(sum(p.numel() for p in base_params) / 1e6))
            print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in self.__model__.parameters()) / 1e6))
            print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))
            print('Total newly initialized MLP parameter number is : {:.3f} million'.format(sum(p.numel() for p in mlp_params) / 1e6))
            print('The newly initialized mlp layer uses {:.3f} x larger lr'.format(self.head_lr))
            print(f'Start : base lr: {base_lr}, mlp lr: {mlp_lr}')

        # only for preliminary test, formal exps should use fixed learning rate scheduler
        Scheduler = importlib.import_module('scheduler.'+self.scheduler).__getattribute__('Scheduler')
        self.__scheduler__ = Scheduler(self.__optimizer__, warmup_steps=int(self.warmup_epoch*self.ipe), ref_lr=self.lr, T_max=int(self.ipe_scale*self.max_epoch*self.ipe), **kwargs)

        self.scaler = GradScaler(growth_factor=2, backoff_factor=0.5, growth_interval=2000) if self.mixedprec else None

        # logging
        self.no_wandb = no_wandb
        self.print_freq = kwargs['print_freq']
        self.result_save_path = kwargs['result_save_path']

    def train_network(self, loader=None, evalmode=None, epoch=-1):
        # Setting for the logging
        batch_time = AverageMeter('Time', ':6.2f')
        data_time = AverageMeter('Data', ':6.2f')
        mem = AverageMeter('Mem (GB)', ':6.1f')
        metrics = AverageMeter('Train Loss', ':1.3e') if not evalmode else AverageMeter('Val Loss', ':1.3e')

        progress = ProgressMeter(
            len(loader),
            [batch_time, data_time, mem, metrics],
            prefix="Epoch: [{}]".format(epoch))
        
        # number of model parameters
        param_num = 0
        for p in self.__model__.parameters():
            param_num += p.numel()

        if evalmode:
            self.__model__.eval();
            A_predictions, A_targets, A_loss, syn_A_loss = [], [], [], []
        else:
            self.__model__.train();

        data_iter = 0
        end = time.time()
        
        for data in loader:

            # measure data loading time
            data_time.update(time.time() - end)

            data_a, data_v, _,_,_,_,labels = data

            # transform input to torch cuda tensor if running gpu
            data_a = data_a.to(self.device)          # batch x target_length x num melbins
            data_v = data_v.to(self.device)          # batch x channel x width x height
            labels = labels.to(self.device)

            # ==================== FORWARD PASS ====================
            with autocast(enabled=self.mixedprec):
                loss_dict, outputs = self.__model__(data_a, data_v, labels=labels)
             
            if not evalmode:
                _new_lr = self.__scheduler__.step()

                if self.mixedprec:
                    # mixed precision
                    self.scaler.scale(loss_dict['loss']).backward();
                    self.scaler.step(self.__optimizer__);
                    self.scaler.update();       
                else:
                    # single precision
                    loss_dict['loss'].backward()
                    self.__optimizer__.step();

                self.zero_grad();

                # logging
                metrics.update(loss_dict['loss'], loader.batch_size)
            
            elif evalmode:
                predictions = outputs['output'].to('cpu').detach()

                A_predictions.append(predictions)
                A_targets.append(labels)
                A_loss.append(loss_dict["loss"].to('cpu').detach())
                syn_A_loss.append(loss_dict["loss"].reshape(1).to('cpu').detach())

                metrics.update(loss_dict['loss'], loader.batch_size)
            
            # measure elapsed time and memory
            batch_time.update(time.time() - end)
            end = time.time()
            mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if data_iter % self.print_freq == 0:
                    param_sum = 0
                    for p in self.__model__.parameters():
                        param_sum += torch.pow(p.detach(),2).sum()
                    param_avg = torch.sqrt(param_sum) / param_num

                    if self.gpu == 0:
                        if not self.no_wandb and not evalmode:
                            wandb.log({"Train Loss": metrics.val,
                                'scaler': self.scaler.get_scale() if self.mixedprec else 0,
                                'base_lr': self.__optimizer__.param_groups[0]['lr'],
                                'mlp_lr': self.__optimizer__.param_groups[1]['lr'],
                                'param_avg': param_avg,
                            })

                        log_info = progress.display(data_iter)
                        
                        with open(os.path.join(self.result_save_path, 'log.txt'), 'a') as f:
                            f.write('Eval: '+ '\t'.join(log_info) + '\n' if evalmode else 'Train: '+ '\t'.join(log_info) + '\n')
            data_iter += 1

        if evalmode:
            audio_output = torch.cat(A_predictions)
            target = torch.cat(A_targets)
            loss = np.mean(A_loss)

            stats = calculate_stats(audio_output, target)
            
            return stats, loss

        elif not evalmode:

            sys.stdout.write("\n");
            progress.synchronize()
            
            return metrics.avg

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Save parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def saveParameters(self, path):
        torch.save(self.__model__.module.state_dict(), path);

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Load parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def loadParameters(self, path):
        self_state = self.__model__.module.state_dict()

        if self.device_name == "cpu":
            loaded_state = torch.load(path, map_location="cpu")
        else:
            loaded_state = torch.load(path, map_location="cuda:%d"%self.gpu)
        

        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = origname.replace('__M__.','') if 'cav_mae' not in self.model_name else origname.replace('module','__M__')
                if name not in self_state:
                    if self.gpu == 0: print("{} is not in the model.".format(origname))
                    continue
                else:
                    if self.gpu == 0: print("{} is loaded in the model".format(name))
            else:
                if self.gpu == 0: print("{} is loaded in the model".format(name))

            model_param = self_state[name]

            if model_param.size() != param.size():
                if 'head' in name.lower():  # covers mlp_head, classifier, etc.
                    if model_param.dim() == 2:  # weight
                        num_rows_to_copy = min(model_param.size(0), param.size(0))
                        with torch.no_grad():
                            model_param[:num_rows_to_copy].copy_(param[:num_rows_to_copy])
                        if self.gpu == 0:
                            print(f"Warning: {name} weight mismatch. Copied first {num_rows_to_copy} rows; remaining rows keep current initialization.")
                    elif model_param.dim() == 1:  # bias
                        num_elements_to_copy = min(model_param.size(0), param.size(0))
                        with torch.no_grad():
                            model_param[:num_elements_to_copy].copy_(param[:num_elements_to_copy])
                        if self.gpu == 0:
                            print(f"Warning: {name} bias mismatch. Copied first {num_elements_to_copy} entries; remaining entries keep current initialization.")
                    continue
                else:
                    if self.gpu == 0:
                        print(f"Warning: parameter {name} shape mismatch, model: {model_param.size()}, checkpoint: {param.size()}. Skipping.")
                    continue

            # Exact match: copy entire parameter
            model_param.copy_(param)


    def train_on_single_pair(self, audio_data, video_data, label_data):
        """
        Trains the network on a single audio, video, and label pair.

        Args:
            self: An instance of the class containing the model, optimizer, etc.
                (e.g., your Trainer class).
            audio_data (torch.Tensor): A single audio input tensor.
            video_data (torch.Tensor): A single video input tensor.
            label_data (torch.Tensor): A single label tensor.

        Returns:
            float: The training loss for this single pair.
        """
        print("Beginning to train network on a single pair!")

        # Setting for the logging (simplified for a single pair)
        batch_time = AverageMeter('Time', ':6.2f')
        data_time = AverageMeter('Data', ':6.2f')
        mem = AverageMeter('Mem (GB)', ':6.1f')
        metrics = AverageMeter('Train Loss', ':1.3e')

        # Ensure model is in training mode
        self.__model__.train()

        end = time.time()

        # Measure "data loading" time (simulated for a single pair)
        data_time.update(0) # Since data is directly passed

        # transform input to torch cuda tensor if running gpu
        audio_data = audio_data.to(self.device)  # target_length x num melbins (assuming batch dim is handled by model)
        video_data = video_data.to(self.device)  # channel x width x height (assuming batch dim is handled by model)
        label_data = label_data.to(self.device)

        # Add a batch dimension if your model expects it
        # For example, if your model expects [batch_size, ...] but you're passing [feature_dims, ...]
        audio_data = audio_data.unsqueeze(0) if audio_data.dim() == 2 else audio_data
        video_data = video_data.unsqueeze(0) if video_data.dim() == 3 else video_data
        label_data = label_data.unsqueeze(0) if label_data.dim() == 1 else label_data

        # ==================== FORWARD PASS ====================
        with autocast(enabled=self.mixedprec):
            loss_dict, outputs = self.__model__(audio_data, video_data, labels=label_data)

        _new_lr = self.__scheduler__.step() # You might want to control when scheduler steps

        if self.mixedprec:
            # mixed precision
            self.scaler.scale(loss_dict['loss']).backward()
            self.scaler.step(self.__optimizer__)
            self.scaler.update()
        else:
            # single precision
            loss_dict['loss'].backward()
            self.__optimizer__.step()

        self.zero_grad() # Assuming this resets gradients for the next step

        # logging
        # For a single pair, batch_size is 1
        metrics.update(loss_dict['loss'].item(), 1) # Use .item() to get scalar from tensor

        # measure elapsed time and memory
        batch_time.update(time.time() - end)
        mem.update(torch.cuda.max_memory_allocated() // 1e9)

        print(f"Trained on single pair. Loss: {metrics.val:.4f}")

        # You can return the loss or any other relevant metric
        return metrics.val