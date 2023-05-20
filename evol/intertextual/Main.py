from __future__ import division
from __future__ import print_function

import time
import argparse
import torch
import torch.nn as nn
from optims import Optim
import util
from util import utils
import lr_scheduler as L
from tqdm import tqdm
import sys
import os
from Data import *
from tool import *
import random

parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
current_dir = os.getcwd()
sys.path.insert(0, parent_dir)
from util.nlp_utils import *
import json


# config
def parse_args():
    parser = argparse.ArgumentParser(description='train.py')
    parser.add_argument('-config', default='config.yaml', type=str,
                        help="config file")
    parser.add_argument('-model', default='base', type=str,
                        choices=['base,para'])
    parser.add_argument('-gpus', default=[1], type=int,
                        help="Use CUDA on the listed devices.")
    parser.add_argument('-restore',
                        type=str, default='',
                        help="restore checkpoint")
    parser.add_argument('-seed', type=int, default=1234,
                        help="Random seed")
    parser.add_argument('-type', default='train', choices=['train', 'eval'],
                        help='train type or eval')
    parser.add_argument('-log', default='', type=str,
                        help="log directory")

    opt = parser.parse_args()
    # 用config.data来得到config中的data选项
    config = util.utils.read_config(opt.config)
    return opt, config


# set opt and config as global variables
args, config = parse_args()
random.seed(args.seed)
np.random.seed(args.seed)


# Training settings

def set_up_logging():
    # log为记录文件
    # config.log是记录的文件夹, 最后一定是/
    # opt.log是此次运行时记录的文件夹的名字
    if not os.path.exists(config.log):
        os.mkdir(config.log)
    if args.log == '':
        log_path = config.log + utils.format_time(time.localtime()) + '/'
    else:
        log_path = config.log + args.log + '/'
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    logging = utils.logging(log_path + 'log.txt')  # 往这个文件里写记录
    logging_csv = utils.logging_csv(log_path + 'record.csv')  # 往这个文件里写记录
    for k, v in config.items():
        logging("%s:\t%s\n" % (str(k), str(v)))
    logging("\n")
    return logging, logging_csv, log_path


logging, logging_csv, log_path = set_up_logging()
use_cuda = torch.cuda.is_available()


def train(model,dataloader_train, scheduler,optim,updates):
    for epoch in range(1, config.epoch + 1):
        total_loss=0.
        start_time = time.time()

        model.train()

        if config.schedule:
            scheduler.step()
            logging("Decaying learning rate to %g\n" % scheduler.get_lr()[0])

        for batch in tqdm(dataloader_train):
            model.zero_grad()
            batch=model(batch)

            loss = loss_CL(batch_emb)
            if torch.isnan(loss):
                break


            loss.backward()
            optim.step()
            updates += 1  # 进行了一次更新
            total_loss += loss.data.item()
            show=100

            if updates%show==0:
                logging("time: %6.3f, epoch: %3d, updates: %8d, train loss: %6.3f\n"
                        % (time.time() - start_time, epoch, updates, total_loss*100 / show))

                total_loss = 0.

        #logging("learning rate to %g" % scheduler.get_lr()[0])
        #print('evaluating after %d updates...\r' % updates)
        if epoch==5:
            save_model(log_path + str(updates) + '_updates_checkpoint.pt', model, optim, updates)

    save_model(log_path + str(updates) + '_updates_checkpoint.pt', model, optim, updates)



def save_model(path, model, optim,updates):
    '''保存的模型是一个字典的形式, 有model, config, optim, updates.'''
    # 如果使用并行的话使用的是model.module.state_dict()
    checkpoints = {
        'model':model.state_dict(),
        'optim':optim,
        'config': config,
        'updates': updates}

    torch.save(checkpoints, path)

def get_dataloader(data, batch_size, max_length):
    print('get dataloader')
    dataloader = DataLoader(data, batch_size, max_length)
    return dataloader()

def main():
    # 设定种子
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    torch.backends.cudnn.benchmark = True
    # model
    print('building model...\n')
    if args.model=='base':
        model = model_CL(use_cuda)

    if args.restore:
        print('loading checkpoint...\n')
        checkpoints = torch.load(os.path.join(log_path, args.restore))
        model.load_state_dict(checkpoints['model'])
    if use_cuda:
        model.cuda()


    '''
    if use_cuda:
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            model = nn.DataParallel(model)

        model.cuda()
    '''
    param_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #param_pretrain= sum(p.numel() for p in model.encoder.parameters() )
    param_count=sum(param.numel() for param in model.parameters())
    param_pretrain=param_count-param_train

    logging('total number of parameters: %d\n\n' % param_count)
    logging('total number of pretrain parameters: %d\n\n' % param_pretrain)
    logging('total number of train parameters: %d\n\n' % param_train)

    print('# parameters:', sum(param.numel() for param in model.parameters()))

    # updates是已经进行了几个epoch, 防止中间出现程序中断的情况.
    if args.restore:
        updates = checkpoints['updates']
    else:
        updates = 0

    #optimizer = optim.Adam(self.params, lr=self.lr)
    optim= Optim(config.optim, config.learning_rate, config.max_grad_norm,
                         lr_decay=config.learning_rate_decay, start_decay_at=config.start_decay_at)

    optim.set_parameters(model.parameters())
    if config.schedule:
        scheduler = L.SetLR(optim.optimizer)

    else:
        scheduler = None


    if args.type=='train':

        start_time = time.time()
        dataloader_train = get_dataloader(config.data,config.batch_size,config.max_length)

        print('loading data...\n')
        print('loading time cost: %.3f' % (time.time() - start_time))

        train(model,  dataloader_train,scheduler,optim,updates)
        #logging("Best acc score: %.2f\n" % (max_acc))

    else:
        print('error')


if __name__ == '__main__':
    main()
