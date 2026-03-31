from __future__ import print_function

import argparse
import csv
import functools
import json
import math
import os
import random
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import StanfordCars, Food101, SUN397, EuroSAT, Caltech256, Country211, Flowers102, PCAM, FGVCAircraft
from torchvision.datasets import *

import clip
from autoattack import AutoAttack

from models.prompters import TokenPrompter, NullPrompter
from utils import accuracy, AverageMeter, ProgressMeter, save_checkpoint
from utils import cosine_lr, convert_models_to_fp32, refine_classname
from utils import one_hot_embedding


best_acc1 = 0
device = "cuda" if torch.cuda.is_available() else "cpu"

CIFAR100_MEAN = (0.48145466, 0.4578275, 0.40821073)
CIFAR100_STD = (0.26862954, 0.26130258, 0.27577711)

upper_limit, lower_limit = 1.0, 0.0


def parse_option():
    parser = argparse.ArgumentParser('Robust Fine-Tuning for CLIP with CIFAR100 / CIFAR100-LT')

    # logging / io
    parser.add_argument('--print_freq', type=int, default=20)
    parser.add_argument('--save_freq', type=int, default=200)
    parser.add_argument('--validate_freq', type=int, default=1)
    parser.add_argument('--model_dir', type=str, default='./save/models')
    parser.add_argument('--result_dir', type=str, default='./save/results')
    parser.add_argument('--image_dir', type=str, default='./save/images')
    parser.add_argument('--filename', type=str, default=None)
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--resume', type=str, default=None)

    # runtime
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', action='store_true', default=True)

    # train schedule
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--optim', type=str, default='sgd', choices=['sgd'])
    parser.add_argument('--learning_rate', type=float, default=1e-7)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--warmup', type=int, default=1000)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--patience', type=int, default=1000)
    parser.add_argument('--last_num_ft', type=int, default=-1,
                        help='-1 means fine-tune all visual encoder params; otherwise last N params only')

    # adversarial train / eval
    parser.add_argument('--train_eps', type=float, default=2.0)
    parser.add_argument('--train_numsteps', type=int, default=5)
    parser.add_argument('--train_stepsize', type=float, default=1.0)
    parser.add_argument('--test_eps', type=float, default=2.0)
    parser.add_argument('--test_numsteps', type=int, default=5)
    parser.add_argument('--test_stepsize', type=float, default=1.0)
    parser.add_argument('--CW', action='store_true')
    parser.add_argument('--autoattack', action='store_true')
    parser.add_argument('--no_adv_train', action='store_true',
                        help='if set, train on clean images only')

    # model
    parser.add_argument('--model', type=str, default='clip')
    parser.add_argument('--arch', type=str, default='vit_b32')
    parser.add_argument('--method', type=str, default='null_patch', choices=['null_patch'])
    parser.add_argument('--prompt_size', type=int, default=30)
    parser.add_argument('--add_prompt_size', type=int, default=0)
    parser.add_argument('--imagenet_root', type=str, default=None)
    parser.add_argument('--mix_alpha', type=float, default=-1)

    # dataset / experiment design
    parser.add_argument('--root', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='cifar100', choices=[
        'cifar10', 'cifar100', 'cifar100_lt', 'cifar100_balanced_subset', 'ImageNet'
    ])
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--imbalance_factor', type=float, default=100.0,
                        help='imbalance factor for CIFAR100-LT, e.g. 10, 50, 100')
    parser.add_argument('--lt_seed', type=int, default=0,
                        help='seed for building LT or balanced subsets')
    parser.add_argument('--max_samples_per_class', type=int, default=500,
                        help='for CIFAR100 train split this should remain 500 unless custom data used')
    parser.add_argument('--balanced_subset_mode', type=str, default='match_lt_total',
                        choices=['match_lt_total'],
                        help='balanced subset budget rule')

    # evaluation control
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate only; load checkpoint if provided, otherwise evaluate current/original CLIP')
    parser.add_argument('--no_finetune', action='store_true',
                        help='skip training and evaluate original CLIP under current attack setting')
    parser.add_argument('--eval_train_split', action='store_true', default=True,
                        help='evaluate train split to inspect overfitting')
    parser.add_argument('--train_eval_max_batches', type=int, default=None,
                        help='optional cap when evaluating train split for speed')
    parser.add_argument('--val_datasets', type=str, nargs='*', default=None,
                        help='override external evaluation datasets; default uses cifar10/cifar100/dtd in train mode')
    parser.add_argument('--save_json_every_eval', action='store_true')

    args = parser.parse_args()

    args.train_eps = args.train_eps / 255.0
    args.test_eps = args.test_eps / 255.0
    args.train_stepsize = args.train_stepsize / 255.0
    args.test_stepsize = args.test_stepsize / 255.0

    args.filename = '{}_{}_{}_{}_{}_{}_{}_lr_{}_decay_{}_bsz_{}_warmup_{}_seed_{}_ltif_{}'.format(
        args.name, args.method, args.prompt_size, args.dataset, args.model, args.arch,
        args.optim, args.learning_rate, args.weight_decay, args.batch_size, args.warmup, args.seed,
        args.imbalance_factor
    )

    return args


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


mu = None
std = None


def init_normalization_tensors():
    global mu, std
    if device == 'cuda':
        mu = torch.tensor(CIFAR100_MEAN, device='cuda').view(3, 1, 1)
        std = torch.tensor(CIFAR100_STD, device='cuda').view(3, 1, 1)
    else:
        mu = torch.tensor(CIFAR100_MEAN).view(3, 1, 1)
        std = torch.tensor(CIFAR100_STD).view(3, 1, 1)


def normalize(X):
    return (X - mu) / std


def clip_img_preprocessing(X):
    img_size = 224
    X = torch.nn.functional.interpolate(X, size=(img_size, img_size), mode='bicubic', align_corners=False)
    X = normalize(X)
    return X


class TextCLIP(nn.Module):
    def __init__(self, model):
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self, text):
        return self.model.encode_text(text)


class ImageCLIP(nn.Module):
    def __init__(self, model):
        super(ImageCLIP, self).__init__()
        self.model = model

    def forward(self, image, prompt_token=None):
        return self.model.encode_image(image, prompt_token)


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, torch.tensor(upper_limit, device=X.device)), torch.tensor(lower_limit, device=X.device))


def create_logits(x1, x2, logit_scale):
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)
    logits_per_x1 = logit_scale * x1 @ x2.t()
    logits_per_x2 = logit_scale * x2 @ x1.t()
    return logits_per_x1, logits_per_x2


def multiGPU_CLIP(model_image, model_text, model, images, text_tokens, prompt_token=None):
    if prompt_token is not None:
        bs = images.size(0)
        prompt_token = prompt_token.repeat(bs, 1, 1)

    img_embed, scale_text_embed = model(images, text_tokens, prompt_token)
    logits_per_image = img_embed @ scale_text_embed.t()
    logits_per_text = scale_text_embed @ img_embed.t()
    return logits_per_image, logits_per_text


def multiGPU_CLIP_image_logits(images, model, text_tokens, prompter=None, add_prompter=None):
    image_tokens = clip_img_preprocessing(images)
    prompt_token = None if add_prompter is None else add_prompter()
    if prompter is not None:
        image_tokens = prompter(image_tokens)
    return multiGPU_CLIP(None, None, model, image_tokens, text_tokens, prompt_token=prompt_token)[0]


def attack_CW_noprompt(prompter, model, model_text, model_image, criterion, X, target, text_tokens, alpha,
                       attack_iters, norm, restarts=1, early_stop=True, epsilon=0):
    delta = torch.zeros_like(X).to(X.device)
    if norm == 'l_inf':
        delta.uniform_(-epsilon, epsilon)
    elif norm == 'l_2':
        delta.normal_()
        d_flat = delta.view(delta.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r / n * epsilon
    else:
        raise ValueError
    delta = torch.clamp(delta, lower_limit - X, upper_limit - X)
    delta.requires_grad = True

    for _ in range(attack_iters):
        images_adv = clip_img_preprocessing(X + delta)
        output, _ = multiGPU_CLIP(model_image, model_text, model, images_adv, text_tokens, None)

        num_class = output.size(1)
        label_mask = one_hot_embedding(target, num_class).to(X.device)
        correct_logit = torch.sum(label_mask * output, dim=1)
        wrong_logit, _ = torch.max((1 - label_mask) * output - 1e4 * label_mask, dim=1)
        loss = -torch.sum(F.relu(correct_logit - wrong_logit + 50))

        loss.backward()
        grad = delta.grad.detach()

        if norm == 'l_inf':
            d = torch.clamp(delta + alpha * torch.sign(grad), min=-epsilon, max=epsilon)
        else:
            g_norm = torch.norm(grad.view(grad.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            scaled_g = grad / (g_norm + 1e-10)
            d = (delta + scaled_g * alpha).view(delta.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(delta)

        d = torch.clamp(d, lower_limit - X, upper_limit - X)
        delta.data[:] = d
        delta.grad.zero_()

    return delta.detach()


def attack_pgd_noprompt(prompter, model, model_text, model_image, criterion, X, target, text_tokens, alpha,
                        attack_iters, norm, restarts=1, early_stop=True, epsilon=0):
    delta = torch.zeros_like(X).to(X.device)
    if norm == 'l_inf':
        delta.uniform_(-epsilon, epsilon)
    elif norm == 'l_2':
        delta.normal_()
        d_flat = delta.view(delta.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r / n * epsilon
    else:
        raise ValueError
    delta = torch.clamp(delta, lower_limit - X, upper_limit - X)
    delta.requires_grad = True

    for _ in range(attack_iters):
        images_adv = clip_img_preprocessing(X + delta)
        output, _ = multiGPU_CLIP(model_image, model_text, model, images_adv, text_tokens, None)
        loss = criterion(output, target)
        loss.backward()
        grad = delta.grad.detach()

        if norm == 'l_inf':
            d = torch.clamp(delta + alpha * torch.sign(grad), min=-epsilon, max=epsilon)
        else:
            g_norm = torch.norm(grad.view(grad.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            scaled_g = grad / (g_norm + 1e-10)
            d = (delta + scaled_g * alpha).view(delta.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(delta)

        d = torch.clamp(d, lower_limit - X, upper_limit - X)
        delta.data[:] = d
        delta.grad.zero_()

    return delta.detach()


def attack_auto(model, images, target, text_tokens, prompter, add_prompter,
                attacks_to_run=('apgd-ce', 'apgd-dlr'), epsilon=0):
    forward_pass = functools.partial(
        multiGPU_CLIP_image_logits,
        model=model, text_tokens=text_tokens,
        prompter=None, add_prompter=None
    )
    adversary = AutoAttack(forward_pass, norm='Linf', eps=epsilon, version='standard', verbose=False)
    adversary.attacks_to_run = list(attacks_to_run)
    x_adv = adversary.run_standard_evaluation(images, target, bs=images.shape[0])
    return x_adv


def get_img_num_per_cls_cifar100_lt(num_classes=100, imb_factor=100.0, max_num=500):
    img_num_per_cls = []
    for cls_idx in range(num_classes):
        num = max_num * ((1.0 / imb_factor) ** (cls_idx / (num_classes - 1.0)))
        img_num_per_cls.append(int(num))
    return img_num_per_cls


def get_balanced_img_num_per_cls(total_count, num_classes=100):
    base = total_count // num_classes
    rem = total_count % num_classes
    nums = [base for _ in range(num_classes)]
    for i in range(rem):
        nums[i] += 1
    return nums


def summarize_class_counts(counts: List[int]) -> Dict[str, float]:
    counts = np.array(counts)
    return {
        'num_classes': int(len(counts)),
        'total_samples': int(counts.sum()),
        'min_class_count': int(counts.min()),
        'max_class_count': int(counts.max()),
        'mean_class_count': float(counts.mean()),
        'median_class_count': float(np.median(counts)),
    }


def make_subset_from_class_counts(dataset, class_counts: List[int], seed: int = 0) -> Tuple[Subset, List[int], List[List[int]]]:
    rng = np.random.RandomState(seed)
    targets = np.array(dataset.targets)
    num_classes = len(class_counts)
    selected_indices = []
    selected_per_class = []

    for c in range(num_classes):
        idx_c = np.where(targets == c)[0]
        rng.shuffle(idx_c)
        take = min(class_counts[c], len(idx_c))
        chosen = idx_c[:take].tolist()
        selected_indices.extend(chosen)
        selected_per_class.append(chosen)

    rng.shuffle(selected_indices)
    subset = Subset(dataset, selected_indices)
    return subset, selected_indices, selected_per_class


def build_cifar100_training_variant(root, transform, variant='standard', imb_factor=100.0,
                                    seed=0, max_num=500):
    base_dataset = CIFAR100(root, transform=transform, download=True, train=True)
    metadata = {
        'variant': variant,
        'imbalance_factor': float(imb_factor),
        'seed': int(seed),
    }

    if variant == 'standard':
        class_counts = [0 for _ in range(100)]
        for t in base_dataset.targets:
            class_counts[t] += 1
        metadata['class_counts'] = class_counts
        metadata['summary'] = summarize_class_counts(class_counts)
        return base_dataset, metadata

    if variant == 'lt':
        class_counts = get_img_num_per_cls_cifar100_lt(num_classes=100, imb_factor=imb_factor, max_num=max_num)
        subset, selected_indices, _ = make_subset_from_class_counts(base_dataset, class_counts, seed=seed)
        metadata['class_counts'] = class_counts
        metadata['selected_indices_count'] = len(selected_indices)
        metadata['summary'] = summarize_class_counts(class_counts)
        return subset, metadata

    if variant == 'balanced_subset':
        lt_class_counts = get_img_num_per_cls_cifar100_lt(num_classes=100, imb_factor=imb_factor, max_num=max_num)
        total_count = sum(lt_class_counts)
        balanced_counts = get_balanced_img_num_per_cls(total_count, num_classes=100)
        subset, selected_indices, _ = make_subset_from_class_counts(base_dataset, balanced_counts, seed=seed)
        metadata['reference_lt_class_counts'] = lt_class_counts
        metadata['class_counts'] = balanced_counts
        metadata['selected_indices_count'] = len(selected_indices)
        metadata['summary'] = summarize_class_counts(balanced_counts)
        return subset, metadata

    raise ValueError(f'Unknown CIFAR100 training variant: {variant}')


def resolve_imagenet_root(args):
    import socket
    if args.imagenet_root is not None:
        return args.imagenet_root, args.imagenet_root

    if socket.gethostname() in ['cv12', 'cv13']:
        imagenet_root = '/local/vondrick/chengzhi/ImageNet-clean'
    elif socket.gethostname() == 'cv11':
        imagenet_root = '/local/*/datasets/ImageNet-clean'
    else:
        imagenet_root = '/proj/*3/scott/datasets/ImageNet-clean'

    return imagenet_root, imagenet_root


def build_transforms():
    preprocess = transforms.Compose([
        transforms.ToTensor()
    ])
    preprocess224 = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    preprocess224_interpolate = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return preprocess, preprocess224, preprocess224_interpolate


def build_train_dataset(args, preprocess, preprocess224):
    train_metadata = {}

    if args.dataset == 'cifar100':
        train_dataset, train_metadata = build_cifar100_training_variant(
            args.root, preprocess, variant='standard',
            imb_factor=args.imbalance_factor, seed=args.lt_seed,
            max_num=args.max_samples_per_class
        )
        train_dataset_name = 'cifar100'

    elif args.dataset == 'cifar100_lt':
        train_dataset, train_metadata = build_cifar100_training_variant(
            args.root, preprocess, variant='lt',
            imb_factor=args.imbalance_factor, seed=args.lt_seed,
            max_num=args.max_samples_per_class
        )
        train_dataset_name = 'cifar100_lt'

    elif args.dataset == 'cifar100_balanced_subset':
        train_dataset, train_metadata = build_cifar100_training_variant(
            args.root, preprocess, variant='balanced_subset',
            imb_factor=args.imbalance_factor, seed=args.lt_seed,
            max_num=args.max_samples_per_class
        )
        train_dataset_name = 'cifar100_balanced_subset'

    elif args.dataset == 'cifar10':
        train_dataset = CIFAR10(args.root, transform=preprocess, download=True, train=True)
        train_dataset_name = 'cifar10'
        train_metadata = {'variant': 'standard'}

    elif args.dataset == 'ImageNet':
        imagenet_root, _ = resolve_imagenet_root(args)
        train_dataset = torchvision.datasets.ImageFolder(
            os.path.join(imagenet_root, 'train'),
            transform=preprocess224
        )
        train_dataset_name = 'ImageNet'
        train_metadata = {'variant': 'standard'}

    else:
        raise ValueError(f'Unsupported training dataset: {args.dataset}')

    return train_dataset, train_dataset_name, train_metadata


def default_val_dataset_names(args):
    if args.val_datasets is not None and len(args.val_datasets) > 0:
        return args.val_datasets

    if args.evaluate:
        return ['SUN397', 'Food101', 'flowers102', 'Caltech101', 'Caltech256']

    # return ['ImageNet','STL10','fgvc_aircraft', 'cifar10', 'dtd','Food101','cifar100']
    return ['STL10','fgvc_aircraft', 'cifar10', 'dtd','Food101','cifar100']

def build_eval_datasets(args, preprocess, preprocess224, preprocess224_interpolate):
    _, imgnet_full = resolve_imagenet_root(args)
    val_dataset_name = default_val_dataset_names(args)
    val_dataset_list = []

    for each in val_dataset_name:
        if each == 'cifar10':
            val_dataset_list.append(CIFAR10(args.root, transform=preprocess, download=True, train=False))
        elif each == 'cifar100':
            val_dataset_list.append(CIFAR100(args.root, transform=preprocess, download=True, train=False))
        elif each == 'Caltech101':
            val_dataset_list.append(Caltech101(args.root, target_type='category', transform=preprocess224, download=True))
        elif each == 'PCAM':
            val_dataset_list.append(PCAM(args.root, split='test', transform=preprocess224, download=True))
        elif each == 'STL10':
            val_dataset_list.append(STL10(args.root, split='test', transform=preprocess, download=True))
        elif each == 'SUN397':
            val_dataset_list.append(SUN397(args.root, transform=preprocess224, download=True))
        elif each == 'StanfordCars':
            val_dataset_list.append(StanfordCars(args.root, split='test', transform=preprocess224, download=True))
        elif each == 'Food101':
            val_dataset_list.append(Food101(args.root, split='test', transform=preprocess224, download=True))
        elif each == 'oxfordpet':
            val_dataset_list.append(OxfordIIITPet(args.root, split='test', transform=preprocess224, download=True))
        elif each == 'EuroSAT':
            val_dataset_list.append(EuroSAT(args.root, transform=preprocess224, download=True))
        elif each == 'Caltech256':
            val_dataset_list.append(Caltech256(args.root, transform=preprocess224, download=True))
        elif each == 'flowers102':
            val_dataset_list.append(Flowers102(args.root, split='test', transform=preprocess224, download=True))
        elif each == 'Country211':
            val_dataset_list.append(Country211(args.root, split='test', transform=preprocess224, download=True))
        elif each == 'dtd':
            val_dataset_list.append(DTD(args.root, split='test', transform=preprocess224, download=True))
        elif each == 'fgvc_aircraft':
            val_dataset_list.append(FGVCAircraft(args.root, split='test', transform=preprocess224, download=True))
        elif each == 'hateful_memes':
            val_dataset_list.append(HatefulMemes(args.root, splits=['test_seen', 'test_unseen'], transform=preprocess224_interpolate))
        elif each == 'ImageNet':
            val_dataset_list.append(torchvision.datasets.ImageFolder(
                os.path.join(imgnet_full, 'val'), transform=preprocess224
            ))
        else:
            raise ValueError(f'Unsupported eval dataset: {each}')

    return val_dataset_name, val_dataset_list


def get_dataset_classnames(dataset, dataset_name):
    if hasattr(dataset, 'clip_prompts'):
        return None, dataset.clip_prompts

    class_names = dataset.classes
    if dataset_name == 'ImageNet':
        from utils import load_imagenet_folder2name
        folder2name = load_imagenet_folder2name('imagenet_classes_names.txt')
        class_names = [folder2name[c] for c in class_names]

    class_names = refine_classname(class_names)
    template = 'This is a photo of a {}'
    texts = [template.format(label) for label in class_names]
    return class_names, texts


def build_texts(train_dataset, train_dataset_name, val_dataset_list, val_dataset_name):
    template = 'This is a photo of a {}'

    if isinstance(train_dataset, Subset):
        base_train_dataset = train_dataset.dataset
    else:
        base_train_dataset = train_dataset

    train_class_names = base_train_dataset.classes
    if train_dataset_name == 'ImageNet':
        from utils import load_imagenet_folder2name
        folder2name = load_imagenet_folder2name('imagenet_classes_names.txt')
        train_class_names = [folder2name[c] for c in train_class_names]

    train_class_names = refine_classname(train_class_names)
    texts_train = [template.format(label) for label in train_class_names]

    texts_list = []
    for cnt, each in enumerate(val_dataset_list):
        _, texts_tmp = get_dataset_classnames(each, val_dataset_name[cnt])
        texts_list.append(texts_tmp)

    assert len(texts_list) == len(val_dataset_list)
    return texts_train, texts_list


def build_model_and_optimizer(args):
    add_prompt_len = 0
    model, _ = clip.load('ViT-B/32', device, jit=False, prompt_len=add_prompt_len)
    model_text, model_image = None, None

    convert_models_to_fp32(model)
    model = torch.nn.DataParallel(model)
    model.eval()

    prompter = NullPrompter()
    add_prompter = TokenPrompter(add_prompt_len)
    prompter = torch.nn.DataParallel(prompter).cuda() if device == 'cuda' else torch.nn.DataParallel(prompter)
    add_prompter = torch.nn.DataParallel(add_prompter).cuda() if device == 'cuda' else torch.nn.DataParallel(add_prompter)

    if args.last_num_ft == -1:
        params = model.module.visual.parameters()
    else:
        params = list(model.module.visual.parameters())[-args.last_num_ft:]

    optimizer = torch.optim.SGD(
        params,
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    criterion = torch.nn.CrossEntropyLoss().to(device)
    return model, model_text, model_image, prompter, add_prompter, optimizer, criterion


def maybe_resume(model, optimizer, args):
    global best_acc1
    args.start_epoch = 0

    if not args.resume:
        return

    if not os.path.isfile(args.resume):
        print(f"=> no checkpoint found at '{args.resume}'")
        return

    print(f"=> loading checkpoint '{args.resume}'")
    if args.gpu is None:
        checkpoint = torch.load(args.resume)
    else:
        loc = f'cuda:{args.gpu}'
        checkpoint = torch.load(args.resume, map_location=loc)

    args.start_epoch = checkpoint['epoch']
    best_acc1 = checkpoint['best_acc1']

    if args.gpu is not None and hasattr(best_acc1, 'to'):
        best_acc1 = best_acc1.to(args.gpu)

    if args.mix_alpha > 0:
        alpha = args.mix_alpha
        checkpoint_ori = torch.load('original_clip.pth.tar')
        theta_ori = checkpoint_ori['vision_encoder_state_dict']
        theta_rob = checkpoint['vision_encoder_state_dict']
        theta = {key: (1 - alpha) * theta_ori[key] + alpha * theta_rob[key] for key in theta_ori.keys()}
        model.module.visual.load_state_dict(theta)
    else:
        model.module.visual.load_state_dict(checkpoint['vision_encoder_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")


def make_loader(dataset, args, shuffle=False):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers,
        shuffle=shuffle,
        sampler=None
    )


def get_attack_for_eval(args, dataset_name):
    binary = ['PCAM', 'hateful_memes']
    attacks_to_run = ['apgd-ce', 'apgd-dlr']
    if dataset_name in binary:
        attacks_to_run = ['apgd-ce']
    return attacks_to_run


def evaluate_single_loader(loader, dataset_name, texts, model, model_text, model_image,
                           prompter, add_prompter, criterion, args,
                           max_batches=None):
    batch_time = AverageMeter('Time', ':6.3f')
    clean_losses = AverageMeter('CleanLoss', ':.4e')
    adv_losses = AverageMeter('AdvLoss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top1_adv = AverageMeter('Adv Acc@1', ':6.2f')

    progress = ProgressMeter(
        len(loader),
        [batch_time, clean_losses, adv_losses, top1, top1_adv],
        prefix=dataset_name + '_Eval: '
    )

    model.eval()
    attacks_to_run = get_attack_for_eval(args, dataset_name)
    test_stepsize = args.test_stepsize
    end = time.time()

    print(f'Start evaluating on {dataset_name}')

    with torch.set_grad_enabled(True):
        for i, (images, target) in enumerate(tqdm(loader)):
            if max_batches is not None and i >= max_batches:
                break

            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            text_tokens = clip.tokenize(texts).to(device)

            with torch.no_grad():
                with autocast():
                    output, _ = multiGPU_CLIP(
                        model_image, model_text, model,
                        clip_img_preprocessing(images), text_tokens, None
                    )
                    loss = criterion(output, target)
                    acc1 = accuracy(output, target, topk=(1,))
                    clean_losses.update(loss.item(), images.size(0))
                    top1.update(acc1[0].item(), images.size(0))

            if args.CW:
                delta = attack_CW_noprompt(
                    prompter, model, model_text, model_image, criterion,
                    images, target, text_tokens,
                    test_stepsize, args.test_numsteps, 'l_inf', epsilon=args.test_eps
                )
                attacked_images = images + delta
            elif args.autoattack:
                attacked_images = attack_auto(
                    model, images, target, text_tokens,
                    None, None, epsilon=args.test_eps, attacks_to_run=attacks_to_run
                )
            else:
                delta = attack_pgd_noprompt(
                    prompter, model, model_text, model_image, criterion,
                    images, target, text_tokens,
                    test_stepsize, args.test_numsteps, 'l_inf', epsilon=args.test_eps
                )
                attacked_images = images + delta

            if torch.isnan(attacked_images).any() or torch.isinf(attacked_images).any():
                print(f'[{dataset_name}] bad attacked_images at batch {i}; skipping batch')
                continue

            with torch.no_grad():
                with autocast():
                    output_adv, _ = multiGPU_CLIP(
                        model_image, model_text, model,
                        clip_img_preprocessing(attacked_images), text_tokens, None
                    )
                    loss_adv = criterion(output_adv, target)
                    acc1_adv = accuracy(output_adv, target, topk=(1,))
                    adv_losses.update(loss_adv.item(), images.size(0))
                    top1_adv.update(acc1_adv[0].item(), images.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
                if args.debug:
                    break

    summary = {
        'dataset': dataset_name,
        'clean_acc': float(top1.avg),
        'robust_acc': float(top1_adv.avg),
        'clean_loss': float(clean_losses.avg),
        'robust_loss': float(adv_losses.avg),
        'num_batches_evaluated': int(min(len(loader), max_batches) if max_batches is not None else len(loader)),
    }

    print(
        dataset_name +
        ' * Robust Acc@1 {robust:.3f} * Clean Acc@1 {clean:.3f}'.format(
            robust=top1_adv.avg,
            clean=top1.avg
        )
    )

    return summary


def validate(val_loader_list, val_dataset_name, texts_list, model, model_text, model_image,
             prompter, add_prompter, criterion, args):
    all_results = []

    for cnt in range(len(val_loader_list)):
        result = evaluate_single_loader(
            loader=val_loader_list[cnt],
            dataset_name=val_dataset_name[cnt],
            texts=texts_list[cnt],
            model=model,
            model_text=model_text,
            model_image=model_image,
            prompter=prompter,
            add_prompter=add_prompter,
            criterion=criterion,
            args=args,
            max_batches=None,
        )
        all_results.append(result)

    mean_robust = float(np.mean([r['robust_acc'] for r in all_results])) if all_results else 0.0
    mean_clean = float(np.mean([r['clean_acc'] for r in all_results])) if all_results else 0.0

    return {
        'datasets': all_results,
        'mean_clean_acc': mean_clean,
        'mean_robust_acc': mean_robust,
    }


def evaluate_training_split(train_loader, train_name, texts_train, model, model_text, model_image,
                            prompter, add_prompter, criterion, args):
    return evaluate_single_loader(
        loader=train_loader,
        dataset_name=f'{train_name}_train_split',
        texts=texts_train,
        model=model,
        model_text=model_text,
        model_image=model_image,
        prompter=prompter,
        add_prompter=add_prompter,
        criterion=criterion,
        args=args,
        max_batches=args.train_eval_max_batches,
    )


def compute_overfitting_report(train_result: Dict, val_result_bundle: Dict):
    report = {}
    if train_result is None:
        return report

    report['train_clean_minus_mean_val_clean'] = float(train_result['clean_acc'] - val_result_bundle['mean_clean_acc'])
    report['train_robust_minus_mean_val_robust'] = float(train_result['robust_acc'] - val_result_bundle['mean_robust_acc'])
    report['possible_clean_overfitting'] = bool(report['train_clean_minus_mean_val_clean'] > 5.0)
    report['possible_robust_overfitting'] = bool(report['train_robust_minus_mean_val_robust'] > 5.0)
    return report


def save_eval_report(args, epoch, train_metadata, train_eval_result, val_result_bundle, overfit_report):
    os.makedirs(args.result_dir, exist_ok=True)
    report = {
        'epoch': epoch,
        'args': vars(args),
        'train_metadata': train_metadata,
        'train_eval_result': train_eval_result,
        'val_result_bundle': val_result_bundle,
        'overfit_report': overfit_report,
    }

    json_path = os.path.join(args.result_dir, f'{args.filename}_epoch_{epoch}_eval.json')
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)

    csv_path = os.path.join(args.result_dir, f'{args.filename}_summary.csv')
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                'epoch', 'train_variant', 'mean_val_clean', 'mean_val_robust',
                'train_clean', 'train_robust', 'clean_gap', 'robust_gap'
            ])
        writer.writerow([
            epoch,
            train_metadata.get('variant', 'unknown') if isinstance(train_metadata, dict) else 'unknown',
            val_result_bundle['mean_clean_acc'],
            val_result_bundle['mean_robust_acc'],
            None if train_eval_result is None else train_eval_result['clean_acc'],
            None if train_eval_result is None else train_eval_result['robust_acc'],
            overfit_report.get('train_clean_minus_mean_val_clean', None),
            overfit_report.get('train_robust_minus_mean_val_robust', None),
        ])


def train(train_loader, texts, model, model_text, model_image, prompter, add_prompter,
          optimizer, scheduler, criterion, scaler, epoch, args):
    global best_acc1

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix='Epoch: [{}]'.format(epoch)
    )

    model.train()
    model.module.visual.train()

    num_batches_per_epoch = len(train_loader)
    alpha = args.train_stepsize
    attack_iters = args.train_numsteps
    end = time.time()

    for i, (images, target) in enumerate(tqdm(train_loader)):
        data_time.update(time.time() - end)

        step = num_batches_per_epoch * epoch + i
        scheduler(step)
        optimizer.zero_grad(set_to_none=True)

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        text_tokens = clip.tokenize(texts).to(device)

        if not args.no_adv_train:
            delta = attack_pgd_noprompt(
                prompter, model, model_text, model_image, criterion,
                images, target, text_tokens,
                alpha, attack_iters, 'l_inf', epsilon=args.train_eps
            )
            tmp = clip_img_preprocessing(images + delta)
        else:
            tmp = clip_img_preprocessing(images)

        with autocast():
            output, _ = multiGPU_CLIP(
                model_image, model_text, model,
                tmp, text_tokens, None
            )
            loss = criterion(output, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        model.module.logit_scale.data = torch.clamp(model.module.logit_scale.data, 0, 4.6052)

        acc1 = accuracy(output, target, topk=(1,))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
            if args.debug:
                break

        if i % args.save_freq == 0 and i > 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': prompter.state_dict(),
                'add_prompter': add_prompter.state_dict(),
                'vision_encoder_state_dict': model.module.visual.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, args)

    return losses.avg, top1.avg


def ensure_dirs(args):
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)
    refined_template = 'this_is_a_photo_of_a_{}'
    args.filename = f'{args.filename}_template_{refined_template}'
    args.model_folder = os.path.join(args.model_dir, args.filename)
    os.makedirs(args.model_folder, exist_ok=True)


def main():
    global best_acc1

    args = parse_option()
    ensure_dirs(args)
    set_seed(args.seed)
    init_normalization_tensors()

    print(args)
    if device == 'cuda':
        print('cuda device count:', torch.cuda.device_count())
        print('current device:', torch.cuda.current_device())

    preprocess, preprocess224, preprocess224_interpolate = build_transforms()

    train_dataset, train_dataset_name, train_metadata = build_train_dataset(args, preprocess, preprocess224)
    val_dataset_name, val_dataset_list = build_eval_datasets(args, preprocess, preprocess224, preprocess224_interpolate)

    train_loader = make_loader(train_dataset, args, shuffle=True)
    train_eval_loader = make_loader(train_dataset, args, shuffle=False)
    val_loader_list = [make_loader(ds, args, shuffle=False) for ds in val_dataset_list]

    texts_train, texts_list = build_texts(train_dataset, train_dataset_name, val_dataset_list, val_dataset_name)

    model, model_text, model_image, prompter, add_prompter, optimizer, criterion = build_model_and_optimizer(args)
    maybe_resume(model, optimizer, args)

    scaler = GradScaler()
    total_steps = len(train_loader) * args.epochs
    scheduler = cosine_lr(optimizer, args.learning_rate, args.warmup, total_steps)

    print('Training dataset metadata:')
    print(json.dumps(train_metadata, indent=2))

    # Original CLIP baseline or evaluation-only mode
    if args.no_finetune:
        print('Running original CLIP evaluation without fine-tuning...')
        val_result_bundle = validate(val_loader_list, val_dataset_name, texts_list, model, model_text, model_image,
                                     prompter, add_prompter, criterion, args)
        train_eval_result = None
        if args.eval_train_split:
            train_eval_result = evaluate_training_split(
                train_eval_loader, train_dataset_name, texts_train,
                model, model_text, model_image, prompter, add_prompter, criterion, args
            )
        overfit_report = compute_overfitting_report(train_eval_result, val_result_bundle)
        save_eval_report(args, epoch=0, train_metadata=train_metadata,
                         train_eval_result=train_eval_result,
                         val_result_bundle=val_result_bundle,
                         overfit_report=overfit_report)
        return

    if args.evaluate:
        print('Running evaluation-only on current/or resumed model...')
        val_result_bundle = validate(val_loader_list, val_dataset_name, texts_list, model, model_text, model_image,
                                     prompter, add_prompter, criterion, args)
        train_eval_result = None
        if args.eval_train_split:
            train_eval_result = evaluate_training_split(
                train_eval_loader, train_dataset_name, texts_train,
                model, model_text, model_image, prompter, add_prompter, criterion, args
            )
        overfit_report = compute_overfitting_report(train_eval_result, val_result_bundle)
        save_eval_report(args, epoch=args.start_epoch, train_metadata=train_metadata,
                         train_eval_result=train_eval_result,
                         val_result_bundle=val_result_bundle,
                         overfit_report=overfit_report)
        return

    epochs_since_improvement = 0

    for epoch in range(args.start_epoch, args.epochs):
        train(train_loader, texts_train, model, model_text, model_image, prompter, add_prompter,
              optimizer, scheduler, criterion, scaler, epoch, args)

        if epoch % args.validate_freq == 0:
            val_result_bundle = validate(
                val_loader_list, val_dataset_name, texts_list,
                model, model_text, model_image, prompter, add_prompter, criterion, args
            )
            acc1_mean = val_result_bundle['mean_robust_acc']
        else:
            val_result_bundle = None
            acc1_mean = best_acc1

        train_eval_result = None
        overfit_report = {}
        if val_result_bundle is not None and args.eval_train_split:
            train_eval_result = evaluate_training_split(
                train_eval_loader, train_dataset_name, texts_train,
                model, model_text, model_image, prompter, add_prompter, criterion, args
            )
            overfit_report = compute_overfitting_report(train_eval_result, val_result_bundle)
            save_eval_report(args, epoch=epoch + 1, train_metadata=train_metadata,
                             train_eval_result=train_eval_result,
                             val_result_bundle=val_result_bundle,
                             overfit_report=overfit_report)

        is_best = acc1_mean > best_acc1
        best_acc1 = max(acc1_mean, best_acc1)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': prompter.state_dict(),
            'add_prompter': add_prompter.state_dict(),
            'vision_encoder_state_dict': model.module.visual.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, args, is_best=is_best)

        if is_best:
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            print(f"There's no improvement for {epochs_since_improvement} epochs.")
            if epochs_since_improvement >= args.patience:
                print('The training halted by early stopping criterion.')
                break


if __name__ == '__main__':
    main()

