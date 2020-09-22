# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
import datetime
import numpy as np
import time
import datetime
import sys
from path import Path
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import json

import matplotlib.pyplot as plt
from utils.official import *
from utils.img_process import tensor2array,tensor2array2
from kitti_utils import *
from layers import *


from datasets import D2V_Dataset
import networks
from utils.logger import TermLogger,AverageMeter
from utils.erodila import rectify
from utils.masks import VarMask,MeanMask,IdenticalMask,float8or
class Trainer:
    def __init__(self, options):
        self.opt = options
        self.start_time = datetime.datetime.now().strftime("%m-%d-%H:%M")
        self.checkpoints_path = Path(self.opt.log_dir)/self.start_time
        #save model and events


        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}#dict
        self.parameters_to_train = []#list

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_foggy_frames = 1




#decoder and encode



        if self.opt.net_arch=='en_decoder':
            # pose encoder
            if self.opt.load_weights_folder:
                pass
            else:
                print("FoggyNet encoder pretrained or scratch: " + self.opt.weights_init)
                print("FoggyNet encoder load:" + self.opt.encoder_path)
            self.models["encoder"] = networks.ResnetEncoder(
                                            self.opt.num_layers,
                                            self.opt.weights_init == "pretrained",
                                            num_input_images=self.num_foggy_frames,
                                            encoder_path=self.opt.encoder_path)
            self.models["encoder"].to(self.device)
            self.parameters_to_train += list(self.models["encoder"].parameters())

            # pose decoder
            self.models["foggy"] = networks.FoggyDecoder(
                self.models["encoder"].num_ch_enc,
                num_input_features=1,
                out_dim=len(self.opt.columns))
            self.models["foggy"].to(self.device)
            self.parameters_to_train += list(self.models["foggy"].parameters())


    #posecnn
        elif self.opt.net_arch=='cnn':
            if self.opt.posecnn_path:

                print("FoggyCNN pretrained or scratch: " + self.opt.weights_init)
                print("FoggyCNN load:" + self.opt.posecnn_path)
            self.models['cnn'] = networks.PoseNet().to(self.device)
            self.parameters_to_train+=list(self.models['cnn'].parameters())





        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)
#load model
        if self.opt.load_weights_folder is not None:
            print("load model from {} instead pretrain".format(self.opt.load_weights_folder))
            self.load_model()


        #print("Training model named:\t  ", self.opt.model_name)
        print("traing files are saved to: ", self.opt.log_dir)
        print("Training is using: ", self.device)
        print("start time: ",self.start_time)



        train_path = Path("./")/"splits"/self.opt.split/options.train_files
        val_path = Path("./")/"splits"/self.opt.split/options.val_files


        train_filenames = readlines(train_path)
        val_filenames = readlines(val_path)
        img_ext = '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs
        self.dataset = D2V_Dataset
        #train loader
        train_dataset = self.dataset(
            data_path=self.opt.data_path,
            gt_path=self.opt.gt_path,
            columns=self.opt.columns,
            filenames=train_filenames,
            height=self.opt.height,
            width=self.opt.width,
            num_scales = 1,
            is_train=True,
            img_ext=img_ext)
        self.train_loader = DataLoader(#train_datasets:KITTIRAWDataset
            dataset=train_dataset,
            batch_size= self.opt.batch_size,
            shuffle= True,
            num_workers=self.opt.num_workers,
            pin_memory=True,
            drop_last=True)

        #val loader
        val_dataset = self.dataset(
            data_path=self.opt.data_path,
            gt_path=self.opt.gt_path,
            columns=self.opt.columns,
            filenames=val_filenames,
            height=self.opt.height,
            width=self.opt.width,
            num_scales=1,
            is_train=False,
            img_ext=img_ext)

        self.val_loader = DataLoader(
            val_dataset,
            self.opt.batch_size,
            True,
            num_workers=self.opt.num_workers,
            pin_memory=True,
            drop_last=True)

        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(self.checkpoints_path/mode)



        print("Using split:{}, train_files:{}, val_files:{}".format( self.opt.split,self.opt.train_files,self.opt.val_files))
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

        #custom

        self.logger = TermLogger(n_epochs=self.opt.num_epochs,
                            train_size=len(self.train_loader),
                            valid_size=len(self.val_loader))
        self.logger.reset_epoch_bar()
        self.columns = self.opt.columns
        self.name_metric =self.opt.name_metric
        self.name_loss = self.opt.name_loss
        self.lambdas = torch.tensor(self.opt.lambdas).to(self.device)

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    #1. forward pass1, more like core
    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """

        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)


        features = self.models["encoder"](inputs["color_aug", 0])
        outputs = self.models["foggy"](features)

        #4.
        loss = ((outputs - 1/inputs['vec_gt']).abs()*self.lambdas).mean()

        return outputs, loss


    def compute_metrics(self,gt, pred):
        """Computation of error metrics between predicted and ground truth depths
        """


        abs_log_rel = torch.abs(torch.log(gt) - torch.log(pred))/torch.log(gt)
        abs_log_rel = abs_log_rel.mean(dim=0).squeeze(0).squeeze(0)

        abs_rel = (torch.abs(gt - pred) / gt).mean(dim=0).squeeze(0).squeeze(0)
        metrics={}
        metrics['abs_rel'] = abs_rel
        metrics['abs_log_rel'] = abs_log_rel


        return metrics


    def terminal_log(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def tb_log(self, mode,  losses=None,metrics=None):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        if losses!=None:
            for key in losses.keys():
                writer.add_scalar("losses/{}".format(key), losses[key], self.step)

        if metrics!=None:
            for key in metrics.keys():
                for idx,component in enumerate(metrics[key]):
                    writer.add_scalar("{}/{}".format(key,self.columns[idx]),component, self.step)





    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = self.checkpoints_path/"models"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(self.checkpoints_path, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """

        save_folder = self.checkpoints_path/"models"/"weights_{}".format(self.epoch)
        save_folder.makedirs_p()

        for model_name, model in self.models.items():
            save_path = save_folder/ "{}.pth".format(model_name)
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
            torch.save(to_save, save_path)

        save_path = save_folder/ "{}.pth".format("adam")
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from load_weights_folder
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)
        print("Loading {} weights...".format(self.opt.models_to_load))

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
    #main cycle
    def epoch_train(self):
        """Run a single epoch of training and validation
        """

        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):
            before_op_time = time.time()
            #model forwardpass
            outputs, loss = self.process_batch(inputs)#
            value = 1/outputs
            self.model_optimizer.zero_grad()
            loss.backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.tb_log_frequency == 0 and self.step < 1000
            late_phase = self.step % 1000 == 0

            #
            self.logger.train_logger_update(batch= batch_idx,time = duration,names=['loss'],values=[float(loss.detach().cpu().numpy())])

            #val, and terminal_val_log, and tb_log
            if early_phase or late_phase:
                for name in self.name_loss.keys():
                    self.name_loss[name] = loss.detach().cpu().numpy()
                self.tb_log(mode="train",  losses =self.name_loss)#terminal log

                if "vec_gt" in inputs:
                    metrics = self.compute_metrics(inputs['vec_gt'], value)
                    for name in self.name_metric.keys():
                        self.name_metric[name] = metrics[name].detach().cpu().numpy()
                    self.tb_log(mode='train', metrics=self.name_metric)


                self.val()


            self.step += 1

        self.model_lr_scheduler.step()

        self.logger.reset_train_bar()
        self.logger.reset_valid_bar()

            #record the metric

    #only 2 methods for public call
    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()

        self.logger.epoch_logger_update(epoch=0,
                                        time=0,
                                        names=self.name_metric.keys(),
                                        values=["{:.4f}".format(float(item[0])) for item in self.name_metric.values()])

        for self.epoch in range(self.opt.start_epoch,self.opt.num_epochs):
            epc_st = time.time()
            self.epoch_train()
            duration = time.time() - epc_st
            self.logger.epoch_logger_update(epoch=self.epoch+1,
                                            time=duration,
                                            names=self.name_metric.keys(),
                                            values=["{:.4f}".format(float(item[0])) for item in self.name_metric.values()])
            if (self.epoch + 1) % self.opt.weights_save_frequency == 0 :
                self.save_model()

    @torch.no_grad()
    def val(self):
        """Validate the model on a single minibatch
        这和之前的常用框架不同， 之前是在train all batches 后再 val all batches，
        这里train batch 再 val batch（frequency）
        """
        self.set_eval()
        try:
            inputs = self.val_iter.__next__()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.__next__()
        time_st = time.time()
        outputs, losses = self.process_batch(inputs)
        duration =time.time() -  time_st
        value = 1/outputs
        self.logger.valid_logger_update(batch=self.val_iter._rcvd_idx,
                                        time=duration*self.opt.tb_log_frequency,
                                        names=self.name_loss.keys(),
                                        values=[float(item) for item in self.name_loss.values()])

        for name in self.name_loss.keys():
            self.name_loss[name] = losses.detach().cpu().numpy()
        self.tb_log(mode="val", losses=self.name_loss)  # terminal log
        if "vec_gt" in inputs:
            metrics = self.compute_metrics(inputs['vec_gt'], value)
            for name in self.name_metric.keys():
                self.name_metric[name] = metrics[name].detach().cpu().numpy()
            self.tb_log(mode='val', metrics=self.name_metric)
            del metrics




        del inputs, outputs, losses
        self.set_train()
