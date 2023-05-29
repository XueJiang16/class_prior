from utils import log
import torch
import torchvision as tv
import time

import numpy as np
from utils.test_utils import arg_parser, get_measures
import os
import json

from sklearn.linear_model import LogisticRegressionCV
from torch.autograd import Variable
import resnet
from collections import Counter
from PIL import Image
from funcs import *

class IDDataset(torch.utils.data.Dataset):
    #ImageNet
    def __init__(self, path, data_ann, transform, loader=None):
        super().__init__()
        self.data_ann = data_ann
        self.loader = loader
        self.transform = transform
        with open(self.data_ann) as f:
            samples = [x.strip().rsplit(' ', 1) for x in f.readlines()]
        self.file_list = []
        self.label_list = []
        for filename, gt_label in samples:
            self.file_list.append(os.path.join(path, filename))
            self.label_list.append(int(gt_label))
        self.file_list.sort()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        path = self.file_list[item]
        sample = Image.open(path)
        if sample.mode != 'RGB':
            sample = sample.convert('RGB')
        label = self.label_list[item]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

class IDDataset2(torch.utils.data.Dataset):
    #INaturalist
    def __init__(self, path, data_ann, transform, loader=None):
        super().__init__()
        self.data_ann = data_ann
        self.loader = loader
        self.transform = transform
        with open(self.data_ann) as f:
            ann = json.load(f)
        images = ann['images']
        images_dict = dict()
        for item in images:
            images_dict[item['id']] = item['file_name']
        annotations = ann['annotations']
        samples = []
        for item in annotations:
            samples.append([images_dict[item['image_id']], item['category_id']])
        self.file_list = []
        self.label_list = []
        for filename, gt_label in samples:
            self.file_list.append(os.path.join(path, filename))
            self.label_list.append(int(gt_label))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        path = self.file_list[item]
        sample = Image.open(path)
        if sample.mode != 'RGB':
            sample = sample.convert('RGB')
        label = self.label_list[item]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label


def make_id_ood(args, logger):
    """Returns train and validation datasets."""
    crop = 480
    val_tx = tv.transforms.Compose([
        tv.transforms.Resize((crop, crop)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize([123.675/255, 116.28/255, 103.53/255],
                                [58.395/255, 57.12/255, 57.375/255]),
    ])
    id_ann = './meta/val_labeled.txt'
    in_set = IDDataset(args.in_datadir, id_ann, val_tx)
    out_set = tv.datasets.ImageFolder(args.out_datadir, val_tx)

    logger.info(f"Using an in-distribution set with {len(in_set)} images.")
    logger.info(f"Using an out-of-distribution set with {len(out_set)} images.")

    in_loader = torch.utils.data.DataLoader(
        in_set, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    out_loader = torch.utils.data.DataLoader(
        out_set, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    return in_set, out_set, in_loader, out_loader


def run_eval_custom(model, in_loader, out_loader, logger, args, num_classes):
    # switch to evaluate mode
    model.eval()

    logger.info("Running test...")
    logger.flush()
    cls_idx = []
    label_filename = args.id_cls
    with open(label_filename, 'r') as f:
        for line in f.readlines():
            segs = line.strip().split(' ')
            cls_idx.append(int(segs[-1]))
    cls_idx = np.array(cls_idx, dtype='int')
    label_stat = Counter(cls_idx)
    cls_num = [-1 for _ in range(num_classes)]
    for i in range(num_classes):
        cat_num = int(label_stat[i])
        cls_num[i] = cat_num
    target = cls_num / np.sum(cls_num)

    if args.score == 'MSP':
        logger.info("Processing in-distribution data...")
        in_scores, id_labels = iterate_data_msp(in_loader, model)
        logger.info("Processing out-of-distribution data...")
        out_scores, _ = iterate_data_msp(out_loader, model)
    elif args.score == 'RP_MSP':
        logger.info("Processing in-distribution data...")
        in_scores, id_labels = iterate_data_msp_custom(in_loader, model, target)
        logger.info("Processing out-of-distribution data...")
        out_scores, _ = iterate_data_msp_custom(out_loader, model, target)
    elif args.score == 'ODIN':
        logger.info("Processing in-distribution data...")
        in_scores, id_labels = iterate_data_odin(in_loader, model, args.epsilon_odin, args.temperature_odin)
        logger.info("Processing out-of-distribution data...")
        out_scores, _ = iterate_data_odin(out_loader, model, args.epsilon_odin, args.temperature_odin)
    elif args.score == 'RW_ODIN':
        logger.info("Processing in-distribution data...")
        in_scores, id_labels = iterate_data_odin_custom(in_loader, model, args.epsilon_odin, args.temperature_odin, target, mode='linear')
        logger.info("Processing out-of-distribution data...")
        out_scores, _ = iterate_data_odin_custom(out_loader, model, args.epsilon_odin, args.temperature_odin, target, mode='linear')
    elif args.score == 'Energy':
        logger.info("Processing in-distribution data...")
        in_scores, id_labels = iterate_data_energy(in_loader, model, args.temperature_energy)
        logger.info("Processing out-of-distribution data...")
        out_scores, _ = iterate_data_energy(out_loader, model, args.temperature_energy)
    elif args.score == 'RW_Energy':
        logger.info("Processing in-distribution data...")
        in_scores, id_labels = iterate_data_energy_custom(in_loader, model, args.temperature_energy, target,
                                                            mode='linear')
        logger.info("Processing out-of-distribution data...")
        out_scores, _ = iterate_data_energy_custom(out_loader, model, args.temperature_energy, target,
                                                    mode='linear')
    elif args.score == 'Mahalanobis':
        sample_mean, precision, lr_weights, lr_bias, magnitude = np.load(
            os.path.join(args.mahalanobis_param_path, 'results.npy'), allow_pickle=True)
        sample_mean = [s.cuda() for s in sample_mean]
        precision = [p.cuda() for p in precision]

        regressor = LogisticRegressionCV(cv=2).fit([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
                                                   [0, 0, 1, 1])

        regressor.coef_ = lr_weights
        regressor.intercept_ = lr_bias

        temp_x = torch.rand(2, 3, 480, 480)
        temp_x = Variable(temp_x).cuda()
        temp_list = model(x=temp_x, layer_index='all')[1]
        num_output = len(temp_list)

        logger.info("Processing in-distribution data...")
        in_scores, id_labels = iterate_data_mahalanobis(in_loader, model, num_classes, sample_mean, precision,
                                                        num_output, magnitude, regressor)
        logger.info("Processing out-of-distribution data...")
        out_scores, _ = iterate_data_mahalanobis(out_loader, model, num_classes, sample_mean, precision,
                                                 num_output, magnitude, regressor)
    elif args.score == 'GradNorm':
        logger.info("Processing in-distribution data...")
        in_scores, id_labels = iterate_data_gradnorm(in_loader, model, args.temperature_gradnorm, num_classes)
        logger.info("Processing out-of-distribution data...")
        out_scores, _ = iterate_data_gradnorm(out_loader, model, args.temperature_gradnorm, num_classes)

    elif args.score == 'RP_GradNorm':
        logger.info("Processing in-distribution data...")
        in_scores, id_labels = iterate_data_gradnorm_custom(in_loader, model, args.temperature_gradnorm, num_classes, target)
        logger.info("Processing out-of-distribution data...")
        out_scores, _ = iterate_data_gradnorm_custom(out_loader, model, args.temperature_gradnorm, num_classes, target)
    else:
        raise ValueError("Unknown score type {}".format(args.score))

    gather_in_scores = [torch.zeros_like(in_scores) - 1 for _ in range(torch.distributed.get_world_size())]
    gather_out_scores = [torch.zeros_like(out_scores)-1 for _ in range(torch.distributed.get_world_size())]
    gather_labels = [torch.zeros_like(id_labels)-1 for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(gather_in_scores, in_scores)
    torch.distributed.all_gather(gather_out_scores, out_scores)
    torch.distributed.all_gather(gather_labels, id_labels)
    in_scores = torch.stack(gather_in_scores)
    out_scores = torch.stack(gather_out_scores)
    labels = torch.stack(gather_labels)
    in_scores = in_scores[in_scores != -1]
    out_scores = out_scores[out_scores!=-1]
    labels = labels[labels!=-1]
    in_scores = in_scores.cpu().numpy()
    out_scores = out_scores.cpu().numpy()
    labels = labels.cpu().numpy()
    if args.local_rank == 0:
        id_cls = []
        cls_idx = []
        label_filename = args.id_cls
        with open(label_filename, 'r') as f:
            for line in f.readlines():
                segs = line.strip().split(' ')
                cls_idx.append(int(segs[-1]))
        cls_idx = np.array(cls_idx, dtype='int')
        res = Counter(cls_idx)
        for item in labels:
            id_cls.append(res[item])
        id_cls = np.array(id_cls)
        in_examples = in_scores.reshape((-1, 1))
        out_examples = out_scores.reshape((-1, 1))
        id_cls = id_cls.reshape((-1, 1))
        auroc, aupr_in, aupr_out, fpr95 = get_measures(in_examples, out_examples)

        logger.info('============Overall Results for {}============'.format(args.score))
        logger.info('AUROC: {}'.format(auroc))
        logger.info('AUPR (In): {}'.format(aupr_in))
        logger.info('AUPR (Out): {}'.format(aupr_out))
        logger.info('FPR95: {}'.format(fpr95))
        logger.info('quick data: {},{},{},{}'.format(auroc,aupr_in,aupr_out,fpr95))
        logger.flush()


def main(args):
    logger = log.setup_logger(args)
    torch.backends.cudnn.benchmark = True
    if  args.score == 'GradNorm':
        args.batch = 1
    torch.set_default_tensor_type(torch.FloatTensor)
    torch.cuda.set_device(args.local_rank)
    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True

    torch.distributed.init_process_group(backend="nccl")

    in_set, out_set, in_loader, out_loader = make_id_ood(args, logger)

    in_sampler = torch.utils.data.distributed.DistributedSampler(in_set)
    in_loader = torch.utils.data.DataLoader(
        in_set, batch_size = args.batch, shuffle = False,
        num_workers = args.workers, pin_memory = True, drop_last = False, sampler=in_sampler)
    out_sampler = torch.utils.data.distributed.DistributedSampler(out_set)
    out_loader = torch.utils.data.DataLoader(
        out_set, batch_size = args.batch, shuffle = False,
        num_workers = args.workers, pin_memory = True, drop_last = False, sampler=out_sampler)


    logger.info(f"Loading model from {args.model_path}")

    if 'resnet152' in args.model_path:
        model = resnet.resnet152()
    elif 'resnet50' in args.model_path:
        model = resnet.resnet50()
    else:
        model = resnet.resnet101()
   
    b = dict()
    a = torch.load(args.model_path)
    for k, v in a["state_dict"].items():
        b[".".join(k.split(".")[1:])] = v
    model.load_state_dict(b)


    model = model.eval().cuda()

    start_time = time.time()
    run_eval_custom(model, in_loader, out_loader, logger, args, num_classes=1000)

    end_time = time.time()

    logger.info("Total running time: {}".format(end_time - start_time))


if __name__ == "__main__":
    parser = arg_parser()

    parser.add_argument("--in_datadir", help="Path to the in-distribution data folder.")
    parser.add_argument("--out_datadir", help="Path to the out-of-distribution data folder.")

    parser.add_argument('--score', default='MSP')
    parser.add_argument("--id_cls", default='', help="id label file")
    parser.add_argument("--sample_a", default='0', type=str, help='distribution parameter')

    parser.add_argument('--local-rank', type=int, default=0, help='node rank for distributed training')

    # arguments for ODIN
    parser.add_argument('--temperature_odin', default=1000, type=int,
                        help='temperature scaling for odin')
    parser.add_argument('--epsilon_odin', default=0.0, type=float,
                        help='perturbation magnitude for odin')

    # arguments for Energy
    parser.add_argument('--temperature_energy', default=1, type=int,
                        help='temperature scaling for energy')

    # arguments for Mahalanobis
    parser.add_argument('--mahalanobis_param_path', default='checkpoints/finetune/tune_mahalanobis',
                        help='path to tuned mahalanobis parameters')

    # arguments for GradNorm
    parser.add_argument('--temperature_gradnorm', default=1, type=int,
                        help='temperature scaling for GradNorm')

    main(parser.parse_args())
