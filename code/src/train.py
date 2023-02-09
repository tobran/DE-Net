import os, sys
import os.path as osp
import time
import random
import argparse
import numpy as np
from PIL import Image
import pprint

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image, make_grid
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data.distributed import DistributedSampler
import multiprocessing as mp

ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))
sys.path.insert(0, ROOT_PATH)
from lib.utils import mkdir_p, get_rank, merge_args_yaml, get_time_stamp, load_model_opt, save_args
from lib.perpare import prepare_datasets, prepare_models
from lib.modules import sample_one_batch as sample, test as test, train as train
from lib.datasets_clip import get_fix_data


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='DE-Net')
    parser.add_argument('--cfg', dest='cfg_file', type=str, default='../cfg/bird.yml',
                        help='optional config file')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='number of workers(default: 1)')
    parser.add_argument('--model', type=str, default='model0',
                        help='the model for training')
    parser.add_argument('--resume_epoch', type=int, default=1,
                        help='state epoch') 
    parser.add_argument('--resume_model_path', type=str, default='model',
                        help='the filepath of saved checkpoints to resume')
    parser.add_argument('--clip_path', type=str, default='RN50',
                        help='clip path')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size')
    parser.add_argument('--multi_gpus', type=bool, default=False,
                        help='if use multi-gpu')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='gpu id')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--random_sample', action='store_true',default=True, 
                        help='whether to sample the dataset with random sampler')
    args = parser.parse_args()
    return args

def main(args):
    time_stamp = get_time_stamp()
    stamp = '_'.join([str(args.model),str(args.stamp),str(args.CONFIG_NAME),str(args.imsize),time_stamp])
    args.model_save_file = osp.join(ROOT_PATH, 'saved_models', str(args.CONFIG_NAME), stamp)
    args.img_save_dir = osp.join(ROOT_PATH, 'imgs/{0}'.format(osp.join(str(args.CONFIG_NAME), 'train', stamp)))
    log_dir = osp.join(ROOT_PATH, 'logs/{0}'.format(osp.join(str(args.CONFIG_NAME), 'train', stamp)))
    if (args.multi_gpus==True) and (get_rank() != 0):
        None
    else:
        mkdir_p(osp.join(ROOT_PATH, 'logs'))
        mkdir_p(args.model_save_file)
        mkdir_p(args.img_save_dir)
    # prepare TensorBoard
    if (args.multi_gpus==True) and (get_rank() != 0):
        writer = None
    else:
        writer = SummaryWriter(log_dir)
    # prepare dataloader
    train_dl, valid_dl, test_dl, train_dataset, valid_dataset, test_dataset, sampler = prepare_datasets(args)
    args.n_words = train_dataset.n_words
    # prepare models
    image_encoder, text_encoder, clip4trn, clip4evl, netG, netD, netC = prepare_models(args)
    # prepare fixed data 
    fixed_shape, fixed_sent, fixed_clip = get_fix_data(train_dl, test_dl, text_encoder, clip4trn, args)
    if (args.multi_gpus==True) and (get_rank() != 0):
        None
    else:
        fixed_grid = make_grid(fixed_shape.cpu(), nrow=8, range=(-1, 1), normalize=True)
        writer.add_image('fixed shape', fixed_grid, 0)
        img_name = 'fixed_shape.png'
        img_save_path = osp.join(args.img_save_dir, img_name)
        vutils.save_image(fixed_shape.data, img_save_path, nrow=8, range=(-1, 1), normalize=True)
    # prepare optimizer
    optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0001, betas=(0.0, 0.9))
    D_params = list(netD.parameters()) + list(netC.parameters())
    optimizerD = torch.optim.Adam(D_params, lr=0.0004, betas=(0.0, 0.9))
    # load from checkpoint
    start_epoch = 1
    if args.state_epoch!=1:
        start_epoch = args.state_epoch + 1
        path = osp.join(args.pretrained_model_path, 'state_epoch_%03d.pth'%(args.state_epoch))
        netG, netD, netC, optimizerG, optimizerD = load_model_opt(netG, netD, netC, optimizerG, optimizerD, path, args.multi_gpus)
    # print args
    if (args.multi_gpus==True) and (get_rank() != 0):
        None
    else:
        pprint.pprint(args)
        arg_save_path = osp.join(log_dir, 'args.yaml')
        save_args(arg_save_path, args)
        print("Start Training")
    test_interval = args.test_interval
    gen_interval = args.gen_interval
    save_interval = args.save_interval
    for epoch in range(start_epoch, args.max_epoch):
        if (args.multi_gpus==True):
            sampler.set_epoch(epoch)
        start_t = time.time()
        args.current_epoch = epoch

        # training
        train(train_dl, netG, netD, netC, text_encoder, image_encoder, clip4trn, optimizerG, optimizerD, args)
        torch.cuda.empty_cache()
        # generate example
        if epoch%gen_interval==0:
            if args.text_encoder=='CLIP':
                fixed_txt = fixed_clip
            else:
                fixed_txt = fixed_sent
            sample(fixed_shape, fixed_txt, netG, args.multi_gpus, epoch, args.img_save_dir, args.tb_img_interval, writer)
            torch.cuda.empty_cache()
        # save models
        if epoch%save_interval==0:
            if (args.multi_gpus==True) and (get_rank() != 0):
                None
            else:
                state = {'model': {'netG': netG.state_dict(), 'netD': netD.state_dict(), 'netC': netC.state_dict()}, \
                        'optimizers': {'optimizer_G': optimizerG.state_dict(), 'optimizer_D': optimizerD.state_dict()},\
                        'epoch': epoch}
                torch.save(state, '%s/state_epoch_%03d.pth' % (args.model_save_file, epoch))

        # valid and test
        if epoch%test_interval==0:
            if (args.multi_gpus==True) and (get_rank() != 0):
                None
            else:
                torch.cuda.empty_cache()
                sim_v,diff_v,mse_v,mp_v,rmp_v,c_sim_v,mp_cv,rmp_cv = test(valid_dl,netG,text_encoder,image_encoder,clip4trn,clip4evl,args,mode='Validing')
                torch.cuda.empty_cache()
                sim_t,diff_t,mse_t,mp_t,rmp_t,c_sim_t,mp_ct,rmp_ct = test(test_dl, netG,text_encoder,image_encoder,clip4trn,clip4evl,args,mode='Testing')
                # DAMSM metric
                writer.add_scalars('sim',  {"valid": sim_v,  "test": sim_t  }, epoch)
                writer.add_scalars('diff', {"valid": diff_v, "test": diff_t }, epoch)
                writer.add_scalars('mse',  {"valid": mse_v,  "test": mse_t  }, epoch)
                writer.add_scalars('mp',   {"valid": mp_v,   "test": mp_t   }, epoch)
                writer.add_scalars('rmp',  {"valid": rmp_v,  "test": rmp_t  }, epoch)
                # clip metric
                writer.add_scalars('c_sim',{"valid": c_sim_v,"test": c_sim_t}, epoch)
                writer.add_scalars('mp_c', {"valid": mp_cv,  "test": mp_ct  }, epoch)
                writer.add_scalars('rmp_C',{"valid": rmp_cv, "test": rmp_ct }, epoch)
                # print logs
                print('*'*40)
                print('[%d/%d] diff_v %.3f mse_v %.3f sim_v %.3f MP_v %.3f RMP_v %.3f MP_cv %.3f RMP_cv %.3f c_sim_v %.3f'
                    % (epoch, args.max_epoch, diff_v, mse_v, sim_v, mp_v, rmp_v, mp_cv, rmp_cv, c_sim_v))
                print('[%d/%d] diff_t %.3f mse_t %.3f sim_t %.3f MP_t %.3f RMP_t %.3f MP_ct %.3f RMP_ct %.3f c_sim_t %.3f'
                    % (epoch, args.max_epoch, diff_t, mse_t, sim_t, mp_t, rmp_t, mp_ct, rmp_ct, c_sim_t))
                print('*'*40)
                torch.cuda.empty_cache()


        if (args.multi_gpus==True) and (get_rank() != 0):
            None
        else:
            end_t = time.time()
            print('The epoch %d costs %.2fs'%(epoch, end_t-start_t))
            

if __name__ == "__main__":
    args = merge_args_yaml(parse_args())
    # set seed
    if args.manual_seed is None:
        args.manual_seed = 100
        #args.manualSeed = random.randint(1, 10000)
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    if args.cuda:
        if args.multi_gpus:
            torch.cuda.manual_seed_all(args.manual_seed)
            torch.distributed.init_process_group(backend="nccl")
            local_rank = torch.distributed.get_rank()
            torch.cuda.set_device(local_rank)
            args.device = torch.device("cuda", local_rank)
            args.local_rank = local_rank
        else:
            torch.cuda.manual_seed_all(args.manual_seed)
            torch.cuda.set_device(args.gpu_id)
            args.device = torch.device("cuda")
    else:
        args.device = torch.device('cpu')
    main(args)
