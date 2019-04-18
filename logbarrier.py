"""Loads a pretrained model, then attacks it.

   This script minimizes the distance of a misclassified image to an original,
   and enforces misclassification with a log barrier penalty.
"""
import argparse
import os, sys

import numpy as np
import pandas as pd
import pickle as pk

import torch
from torch import nn
from torch.autograd import grad


def Top1Criterion(x,y, model):
    """Returns True if model prediction is in top1"""
    return model(x).topk(1)[1].view(-1)==y

def Top5Criterion(x,y, model):
    """Returns True if model prediction is in top5"""
    return (model(x).topk(5)[1]==y.view(-1,1)).any(dim=-1)

def initialize(x,y,criterion,max_iters=1e3, bounds=(0,1)):
    """Generates random perturbations of clean images until images have incorrect label.

    If the image is already mis-classified, then it is not perturbed."""
    xpert = x.clone()
    dt = 0.01

    correct = criterion(xpert,y)
    k=0
    while correct.sum()>0:
        l = correct.sum()
        xpert[correct] = x[correct] + (1.01)**k*dt*torch.randn(l,*xpert.shape[1:],device=xpert.device)
        xpert[correct].clamp_(*bounds)
        correct = criterion(xpert,y)

        k+=1
        if k>max_iters:
            raise ValueError('failed to initialize: maximum iterations reached')

    return xpert

class Attack(object):

    def __init__(self, model, criterion=Top1Criterion, initialize=initialize, norm=2,
            verbose=True, **kwargs):
        """Attack a model using the log barrier constraint to enforce mis-classification.

        Arguments:
            model: PyTorch model, takes batch of inputs and returns logits
            loader: PyTorch dataset DataLoader
            criterion: function which takes in images and labels and a model, and returns
                a boolean vector, which is True is model prediction is correct
                For example, the Top1 or Top5 classification criteria
            initialize (optional): function which takes in images, labels and a model, and
                returns mis-classified images (an initial starting guess) (default: clipped Gaussians)
            norm (optional): norm to measure adversarial distance with (default: 2)
            verbobose (optional): if True (default), display status during attack

        Keyword arguments:
            bounds: tuple, image bounds (default (0,1))
            dt: step size (default: 0.01)
            alpha: initial Lagrange multiplier of log barrier penalty (default: 0.1)
            beta: shrink parameter of Lagrange multiplier after each inner loop (default: 0.75)
            gamma: back track parameter (default: 0.5)
            max_outer: maximum number of outer loops (default: 15)
            tol: inner loop stopping criteria (default: 1e-6)
            max_inner: maximum number of inner loop iterations (default: 500)
            T: softmax temperature in L-infinity norm approximation (default: 500)

        Returns:
            images: adversarial images mis-classified by the model
        """
        super().__init__()
        self.model = model
        self.criterion = lambda x, y: criterion(x,y,model)
        self.initialize = initialize
        self.labels = None
        self.original_images = None
        self.perturbed_images = None

        if not (norm==2 or norm==np.inf):
            raise ValueError('norm must be either 2 or np.inf')
        self.norm = norm
        self.verbose = verbose


        config = {'bounds':(0,1),
                  'dt': 0.01,
                  'alpha':0.1,
                  'beta':0.75,
                  'gamma':0.5,
                  'max_outer':15,
                  'tol':1e-6,
                  'max_inner':int(5e2),
                  'T':500.}
        config.update(kwargs)

        self.hyperparams = config


    def __call__(self, x, y):
        self.labels = y
        self.original_images = x

        config = self.hyperparams
        model = self.model
        criterion = self.criterion

        bounds, dt, alpha0, beta, gamma, max_outer, tol, max_inner, T = (
                config['bounds'], config['dt'], config['alpha'], config['beta'],
                config['gamma'], config['max_outer'], config['tol'], config['max_inner'],
                config['T'])


        Nb = len(y)
        ix = torch.arange(Nb, device=x.device)

        imshape = x.shape[1:]
        PerturbedImages = torch.full(x.shape,np.nan, device=x.device)

        mis0 = criterion(x,y)

        xpert = initialize(x,y,criterion)

        xpert[~mis0]= x[~mis0]
        xold = xpert.clone()
        xbest = xpert.clone()
        diffBest = torch.full((Nb,),np.inf,device=x.device)
        xpert.requires_grad_(True)

        for k in range(max_outer):
            alpha = alpha0*beta**k

            diff = (xpert - x).view(Nb,-1).norm(self.norm,-1)
            update= diff>0
            for j in range(max_inner):
                p = model(xpert).softmax(dim=-1)

                pdiff = p.max(dim=-1)[0] - p[ix,y]
                s = -torch.log(pdiff).sum()
                g = grad(alpha*s,xpert)[0] # TODO: use only one grad when norm==Linf
                if self.norm==2:
                    with torch.no_grad():
                        xpert[update] = xpert[update].mul(1-dt).add(-dt,
                                g[update]).add(dt,x[update]).clamp_(*bounds)
                elif self.norm==np.inf:
                    Nb_ = xpert[update].shape[0]
                    xpert_, x_ = xpert[update].view(Nb_,-1), x[update].view(Nb_,-1)
                    z_ = (xpert_ - x_)
                    z = torch.abs(z_)

                    #smooth approximation of Linf norm
                    ex_ = ((z*T).softmax(dim=-1)*z).sum(dim=-1)

                    ginf = grad(ex_.sum(),xpert_)[0]

                    with torch.no_grad():
                        GradientStep = ginf.view(Nb_,*imshape) + g[update]
                        xpert[update] = xpert[update].add(-dt,GradientStep).clamp(*bounds)

                with torch.no_grad():
                    # backtrack
                    c = criterion(xpert,y)
                    while c.any():
                        xpert.data[c] = xpert.data[c].clone().mul(1-gamma).add(gamma,xold[c])
                        c = criterion(xpert,y)

                    diff = (xpert - x).view(Nb,-1).norm(self.norm,-1)
                    boolDiff = diff <= diffBest
                    xbest[boolDiff] = xpert[boolDiff]
                    diffBest[boolDiff] = diff[boolDiff]

                    iterdiff = (xpert - xold).view(Nb,-1).norm(self.norm,-1)
                    #med = diff.median()

                    xold = xpert.clone()


                if self.verbose:
                    sys.stdout.write('  [%2d outer, %4d inner] median & max distance: (%4.4f, %4.4f)\r'
                        %(k, j, diffBest.median() , diffBest.max()))

                if not iterdiff.abs().max()>tol:
                    break

        if self.verbose:
            sys.stdout.write('\n')

        switched = ~criterion(xbest,y)
        PerturbedImages[switched] = xbest.detach()[switched]

        self.perturbed_images = PerturbedImages

        return PerturbedImages
