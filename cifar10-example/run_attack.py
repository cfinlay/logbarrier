import argparse, yaml
import os, sys

import numpy as np
import torch

from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import Subset

sys.path.append('../')
from logbarrier import Top1Criterion, Attack

from models import ResNeXt34_2x32

def main():
    parser = argparse.ArgumentParser('Attack an example CIFAR10 model with the Log Barrier attack.'
                                      'Writes adversarial distances (and optionally images) to a npz file.')

    groups0 = parser.add_argument_group('Required arguments')
    groups0.add_argument('--data-dir', type=str, required=True,
            help='Directory where CIFAR10 data is saved')

    groups2 = parser.add_argument_group('Optional attack arguments')
    groups2.add_argument('--num-images', type=int, default=1000,metavar='N',
            help='total number of images to attack (default: 1000)')
    groups2.add_argument('--batch-size', type=int, default=200,metavar='N',
            help='number of images to attack at a time (default: 200) ')
    groups2.add_argument('--save-images', action='store_true', default=False,
            help='save perturbed images to a npy file (default: False)')
    groups2.add_argument('--norm', type=str, default='L2',metavar='NORM',
            choices=['L2','Linf'],
            help='The norm measuring distance between images. (default: "L2")')

    groups2.add_argument('--seed', type=int, default=0,
            help='seed for RNG (default: 0)')
    groups2.add_argument('--random-subset', action='store_true',
            default=False, help='use random subset of test images (default: False)')

    group1 = parser.add_argument_group('Attack hyperparameters')
    group1.add_argument('--dt', type=float, default=0.01, help='step size (default: 0.01)')
    group1.add_argument('--alpha', type=float, default=0.1,
            help='initial Lagrange multiplier of log barrier penalty (default: 0.1)')
    group1.add_argument('--beta', type=float, default=0.75,
            help='shrink parameter of Lagrange multiplier after each inner loop (default: 0.75)')
    group1.add_argument('--gamma', type=float, default=0.5,
            help='back track parameter (default: 0.5)')
    group1.add_argument('--max-outer', type=int, default=15, \
            help='maximum number of outer loops (default: 15)')
    group1.add_argument('--max-inner', type=int, default=500,
            help='max inner loop iterations (default: 500)')
    group1.add_argument('--tol', type=float, default=1e-6,
            help='inner loop stopping criterion (default: 1e-6)')
    group1.add_argument('--T', type=float, default=500,
            help='softmax temperature for approximating Linf-norm (default: 500)')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    i = 0
    while os.path.exists('attack%s'%i):
        i +=1
    pth = os.path.join('./','attack%s/'%i)
    os.makedirs(pth, exist_ok=True)

    args_file_path = os.path.join(pth, 'args.yaml')
    with open(args_file_path, 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)

    has_cuda = torch.cuda.is_available()

    # Data loading code
    transform = transforms.Compose([transforms.ToTensor()])
    ds = CIFAR10(args.data_dir, download=True, train=False, transform=transform)

    if args.random_subset:
        Ix = np.random.choice(10000, size=args.num_images, replace=False)
        Ix = torch.from_numpy(Ix)
    else:
        Ix = torch.arange(args.num_images) # Use the first N images of test set

    subset = Subset(ds, Ix)

    loader = torch.utils.data.DataLoader(
                        subset,
                        batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=has_cuda)

    # Retrieve pre trained model
    classes = 10
    model = ResNeXt34_2x32()
    model.load_state_dict(torch.load('models/example-resnext34_2x32.pth.tar', map_location='cpu'))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)


    criterion = lambda x, y: Top1Criterion(x,y,model)
    if args.norm=='L2':
        norm = 2
    elif args.norm=='Linf':
        norm = np.inf

    if has_cuda:
        model = model.cuda()

    params = {'bounds':(0,1),
              'dt':args.dt,
              'alpha':args.alpha,
              'beta':args.beta,
              'gamma':args.gamma,
              'max_outer':args.max_outer,
              'tol':args.tol,
              'max_inner':args.max_inner,
              'T': args.T}

    attack = Attack(model, norm=norm, **params)


    d1 = torch.full((args.num_images,),np.inf)
    d2 = torch.full((args.num_images,),np.inf)
    dinf = torch.full((args.num_images,),np.inf)
    if has_cuda:
        d1 = d1.cuda()
        d2 = d2.cuda()
        dinf = dinf.cuda()

    if args.save_images:
        PerturbedImages = torch.full((args.num_images, 3, 32, 32), np.nan)
        labels = torch.full((args.num_images,),-1, dtype=torch.long)
        if has_cuda:
            PerturbedImages = PerturbedImages.cuda()
            labels = labels.cuda()


    K = 0
    for i, (x, y) in enumerate(loader):
        print('Batch %2d/%d:'%(i+1,len(loader)))

        Nb = len(y)
        if has_cuda:
            x, y = x.cuda(), y.cuda()

        xpert = attack(x,y)

        diff = x - xpert.detach()
        l1 = diff.view(Nb, -1).norm(p=1, dim=-1)
        l2 = diff.view(Nb, -1).norm(p=2, dim=-1)
        linf = diff.view(Nb, -1).norm(p=np.inf, dim=-1)

        ix = torch.arange(K,K+Nb, device=x.device)

        if args.save_images:
            PerturbedImages[ix] = xpert
            labels[ix] = y
        d1[ix] = l1
        d2[ix] = l2
        dinf[ix] = linf

        K+=Nb


    st = 'logbarrier-'+args.norm

    dists = {'index':Ix.cpu().numpy(),
             'l1':d1.cpu().numpy(),
             'l2':d2.cpu().numpy(),
             'linf':dinf.cpu().numpy()}
    if args.save_images:
        dists['perturbed'] = PerturbedImages.cpu().numpy()
        dists['labels'] = labels.cpu().numpy()

    with open(os.path.join(pth, st+'.npz'), 'wb') as f:
        np.savez(f, **dists)

if __name__=="__main__":
    main()
