import argparse
import os
import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import datasets
import models
import utils
import utils.few_shot as fs
import utils.utils as mutils
from datasets.samplers import CategoriesSampler


def classifier_save_path(config):
    svname = args.name
    if svname is None:
        svname = 'classifier_{}'.format(config['train_dataset'])
        if not args.ViT:
            svname += '_' + config['model_args']['encoder']
        else:
            svname += '_' + config['model_2_args']['encoder']
        clsfr = config['model_args']['classifier']
        if clsfr != 'linear-classifier':
            svname += '-' + clsfr
    if args.tag is not None:
        svname += '_' + args.tag
    save_path = os.path.join('./save', svname)
    utils.ensure_path(save_path)
    utils.set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))

    yaml.dump(config, open(os.path.join(save_path, 'config.yaml'), 'w'))

    return writer, save_path


def classifier_dataset(writer):
    print("classifier_dataset")
    # train
    train_dataset = datasets.make(config['train_dataset'],
                                  **config['train_dataset_args'])
    train_loader = DataLoader(train_dataset, config['batch_size'], shuffle=True,
                              num_workers=8, pin_memory=True)
    utils.log('train dataset: {} (x{}), {}'.format(
        train_dataset[0][0].shape, len(train_dataset),
        train_dataset.n_classes))
    if config.get('visualize_datasets'):
        utils.visualize_dataset(train_dataset, 'train_dataset', writer)

    # val
    if config.get('val_dataset'):
        eval_val = True
        val_dataset = datasets.make(config['val_dataset'],  # val_dataset
                                    **config['val_dataset_args'])  # val_dataset_args
        val_loader = DataLoader(val_dataset, config['batch_size'],
                                num_workers=8, pin_memory=True)
        utils.log('val dataset: {} (x{}), {}'.format(
            val_dataset[0][0].shape, len(val_dataset),
            val_dataset.n_classes))
        if config.get('visualize_datasets'):
            utils.visualize_dataset(val_dataset, 'val_dataset', writer)
    else:
        eval_val = False

    # few-shot eval
    if config.get('fs_dataset'):
        ef_epoch = config.get('eval_fs_epoch')
        if ef_epoch is None:
            ef_epoch = 5
        eval_fs = True

        fs_dataset = datasets.make(config['fs_dataset'],
                                   **config['fs_dataset_args'])
        utils.log('fs dataset: {} (x{}), {}'.format(
            fs_dataset[0][0].shape, len(fs_dataset),
            fs_dataset.n_classes))
        if config.get('visualize_datasets'):
            utils.visualize_dataset(fs_dataset, 'fs_dataset', writer)

        n_way = 5
        n_query = 15
        n_shots = [1, 5]
        fs_loaders = []
        for n_shot in n_shots:
            fs_sampler = CategoriesSampler(
                fs_dataset.label, 200,
                n_way, n_shot + n_query, ep_per_batch=4)
            fs_loader = DataLoader(fs_dataset, batch_sampler=fs_sampler,
                                   num_workers=8, pin_memory=True)
            fs_loaders.append(fs_loader)
    else:
        eval_fs = False

    return train_dataset, train_loader, eval_val, val_loader, eval_fs, fs_loaders, n_shots, n_way, n_query, ef_epoch



def main(config):
    svname = args.name
    if svname is None:
        svname = 'meta_{}-{}shot'.format(
                config['train_dataset'], config['n_shot'])
        svname += '_' + config['model'] + '-' + config['model_args']['encoder']
    if args.tag is not None:
        svname += '_' + args.tag
    save_path = os.path.join('./save', svname)
    utils.ensure_path(save_path)
    utils.set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))

    yaml.dump(config, open(os.path.join(save_path, 'config.yaml'), 'w'))

    #### Dataset ####

    n_way, n_shot = config['n_way'], config['n_shot']
    n_query = config['n_query']

    if config.get('n_train_way') is not None:
        n_train_way = config['n_train_way']
    else:
        n_train_way = n_way
    if config.get('n_train_shot') is not None:
        n_train_shot = config['n_train_shot']
    else:
        n_train_shot = n_shot
    if config.get('ep_per_batch') is not None:
        ep_per_batch = config['ep_per_batch']
    else:
        ep_per_batch = 1

    # train
    train_dataset = datasets.make(config['train_dataset'],
                                  **config['train_dataset_args'])
    utils.log('train dataset: {} (x{}), {}'.format(
            train_dataset[0][0].shape, len(train_dataset),
            train_dataset.n_classes))
    if config.get('visualize_datasets'):
        utils.visualize_dataset(train_dataset, 'train_dataset', writer)
    train_sampler = CategoriesSampler(
            train_dataset.label, config['train_batches'],
            n_train_way, n_train_shot + n_query,
            ep_per_batch=ep_per_batch)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,
                              num_workers=8, pin_memory=True)

    # tval
    if config.get('tval_dataset'):
        tval_dataset = datasets.make(config['tval_dataset'],
                                     **config['tval_dataset_args'])
        utils.log('tval dataset: {} (x{}), {}'.format(
                tval_dataset[0][0].shape, len(tval_dataset),
                tval_dataset.n_classes))
        if config.get('visualize_datasets'):
            utils.visualize_dataset(tval_dataset, 'tval_dataset', writer)
        tval_sampler = CategoriesSampler(
                tval_dataset.label, 200,
                n_way, n_shot + n_query,
                ep_per_batch=4)
        tval_loader = DataLoader(tval_dataset, batch_sampler=tval_sampler,
                                 num_workers=8, pin_memory=True)
    else:
        tval_loader = None

    # val
    val_dataset = datasets.make(config['val_dataset'],
                                **config['val_dataset_args'])
    utils.log('val dataset: {} (x{}), {}'.format(
            val_dataset[0][0].shape, len(val_dataset),
            val_dataset.n_classes))
    if config.get('visualize_datasets'):
        utils.visualize_dataset(val_dataset, 'val_dataset', writer)
    val_sampler = CategoriesSampler(
            val_dataset.label, 200,
            n_way, n_shot + n_query,
            ep_per_batch=4)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler,
                            num_workers=8, pin_memory=True)

    ########
    # writer, save_path = classifier_save_path(config)

    # #### Dataset ####
    # train_dataset, train_loader, eval_val, val_loader, eval_fs, fs_loaders, n_shots, n_way, n_query, ef_epoch = classifier_dataset(
    #         writer)



    #### Model and optimizer ####
    # model_path = '/home/wzp/few-shot-meta-baseline/checkpoint/fc100/jx_vit_base_patch16_224_in21k-e5005f0a.pth'
    # 加载.pth文件中的模型参数
    # checkpoint = torch.load(model_path)

    if config.get('load'):
        model_sv = torch.load(config['load'])
        model = models.load(model_sv)
    else:
        model = models.make(config['model'], **config['model_args'])
        print("model",model)
        model.encoder.load_state_dict(torch.load(config['load_encoder']))
        print("-------------------------------------")
        print("Model load done ----> vit_base_patch16_224_in21k")

        # if config.get('load_encoder'):
        #     encoder = models.load(torch.load(config['load_encoder'])).encoder
        #     model.encoder.load_state_dict(encoder.state_dict())

    if config.get('_parallel'):
        model = nn.DataParallel(model)

    utils.log('num params: {}'.format(utils.compute_n_params(model)))

    optimizer, lr_scheduler = utils.make_optimizer(
            model.parameters(),
            config['optimizer'], **config['optimizer_args'])

    ########
    
    max_epoch = config['max_epoch']
    save_epoch = config.get('save_epoch')
    max_va = 0.
    timer_used = utils.Timer()
    timer_epoch = utils.Timer()

    aves_keys = ['tl', 'ta', 'tvl', 'tva', 'vl', 'va']
    trlog = dict()
    for k in aves_keys:
        trlog[k] = []

    for epoch in range(1, max_epoch + 1):
        timer_epoch.s()
        aves = {k: utils.Averager() for k in aves_keys}

        # train
        model.train()
        if config.get('freeze_bn'):
            utils.freeze_bn(model) 
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        np.random.seed(epoch)
        for data, _ in tqdm(train_loader, desc='train', leave=False):
            x_shot, x_query = fs.split_shot_query(
                    data.cuda(), n_train_way, n_train_shot, n_query,
                    ep_per_batch=ep_per_batch)
            label = fs.make_nk_label(n_train_way, n_query,
                    ep_per_batch=ep_per_batch).cuda()

            logits = model(x_shot, x_query).view(-1, n_train_way)
            loss = F.cross_entropy(logits, label)
            acc = utils.compute_acc(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            aves['tl'].add(loss.item())
            aves['ta'].add(acc)

            logits = None; loss = None 

        # eval
        model.eval()

        for name, loader, name_l, name_a in [
                ('tval', tval_loader, 'tvl', 'tva'),
                ('val', val_loader, 'vl', 'va')]:

            if (config.get('tval_dataset') is None) and name == 'tval':
                continue

            np.random.seed(0)
            for data, _ in tqdm(loader, desc=name, leave=False):
                x_shot, x_query = fs.split_shot_query(
                        data.cuda(), n_way, n_shot, n_query,
                        ep_per_batch=4)
                label = fs.make_nk_label(n_way, n_query,
                        ep_per_batch=4).cuda()

                with torch.no_grad():
                    logits = model(x_shot, x_query).view(-1, n_way)
                    loss = F.cross_entropy(logits, label)
                    acc = utils.compute_acc(logits, label)
                
                aves[name_l].add(loss.item())
                aves[name_a].add(acc)

        _sig = int(_[-1])

        # post
        if lr_scheduler is not None:
            lr_scheduler.step()

        for k, v in aves.items():
            aves[k] = v.item()
            trlog[k].append(aves[k])

        t_epoch = utils.time_str(timer_epoch.t())
        t_used = utils.time_str(timer_used.t())
        t_estimate = utils.time_str(timer_used.t() / epoch * max_epoch)
        utils.log('epoch {}, train {:.4f}|{:.4f}, tval {:.4f}|{:.4f}, '
                'val {:.4f}|{:.4f}, {} {}/{} (@{})'.format(
                epoch, aves['tl'], aves['ta'], aves['tvl'], aves['tva'],
                aves['vl'], aves['va'], t_epoch, t_used, t_estimate, _sig))

        writer.add_scalars('loss', {
            'train': aves['tl'],
            'tval': aves['tvl'],
            'val': aves['vl'],
        }, epoch)
        writer.add_scalars('acc', {
            'train': aves['ta'],
            'tval': aves['tva'],
            'val': aves['va'],
        }, epoch)

        if config.get('_parallel'):
            model_ = model.module
        else:
            model_ = model

        training = {
            'epoch': epoch,
            'optimizer': config['optimizer'],
            'optimizer_args': config['optimizer_args'],
            'optimizer_sd': optimizer.state_dict(),
        }
        save_obj = {
            'file': __file__,
            'config': config,

            'model': config['model'],
            'model_args': config['model_args'],
            'model_sd': model_.state_dict(),

            'training': training,
        }
        torch.save(save_obj, os.path.join(save_path, 'epoch-last.pth'))
        torch.save(trlog, os.path.join(save_path, 'trlog.pth'))

        if (save_epoch is not None) and epoch % save_epoch == 0:
            torch.save(save_obj,
                    os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

        if aves['va'] > max_va:
            max_va = aves['va']
            torch.save(save_obj, os.path.join(save_path, 'max-va.pth'))

        writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--ViT', action='store_true')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True
        config['_gpu'] = args.gpu

    utils.set_gpu(args.gpu)
    main(config)

