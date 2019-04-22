#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 13:20:15 2019

@author: chin-weihuang
"""


from __future__ import print_function
import numpy as np
import torch
import matplotlib.pyplot as plt
from discrim import jsd_objective, Discriminator

# plot p0 and p1
plt.figure()

# empirical
xx = torch.randn(10000)
f = lambda x: torch.tanh(x*2+1) + x*0.75
d = lambda x: (1-torch.tanh(x*2+1)**2)*2+0.75
plt.hist(f(xx), 100, alpha=0.5, density=1)
plt.hist(xx, 100, alpha=0.5, density=1)
plt.xlim(-5,5)
# exact
xx = np.linspace(-5,5,1000)
N = lambda x: np.exp(-x**2/2.)/((2*np.pi)**0.5)
plt.plot(f(torch.from_numpy(xx)).numpy(), d(torch.from_numpy(xx)).numpy()**(-1)*N(xx))
plt.plot(xx, N(xx))
plt.savefig("normal.pdf")


############### import the sampler ``samplers.distribution4'' 
############### train a discriminator on distribution4 and standard gaussian
############### estimate the density of distribution4

#######--- INSERT YOUR CODE BELOW ---#######
from samplers import distribution3, distribution4
import torch as T
from torch import nn
from torch import optim
import seaborn as sns
if T.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def train_discrim(bs=512, iters=10000):
    loss_fcn = jsd_objective
    normal = T.distributions.normal.Normal(0, 1)
    network = Discriminator(1, 256)
    network = network.to(device)
    optimizer = optim.SGD(network.parameters(), 1e-4, 0.99)
    i = 0
    for real, fake in zip(distribution4(bs), distribution3(bs)):
        i += 1
        real = T.tensor(real, device=device).float()
        fake = T.tensor(fake, device=device).float()
        loss, gen_loss, jsd = loss_fcn(network, real, fake)
        loss.backward()
        nn.utils.clip_grad_norm_(network.parameters(), 1)
        optimizer.step()
        optimizer.zero_grad()
        if i > iters:
            break
        elif i % 100 == 0:
            print(loss.item(), jsd.item())

    probe = np.linspace(-10, 10, 10000)
    probe = T.tensor(probe, device=device).float().unsqueeze(-1)

    density = T.exp(normal.log_prob(probe)).squeeze(-1).detach().cpu().numpy()
    dx = T.sigmoid(network(probe)).squeeze(-1).detach().cpu().numpy()
    density_estimate = dx*density/(1 - dx)

    plt.figure()
    plt.plot(probe.cpu().numpy()[:, 0], density_estimate, label="Estimated density")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("Estimated density p(x)")
    plt.title("Estimated pdf of Distribution 4")
    plt.savefig("14_ours.pdf")

    truth = next(distribution4(5000))
    plt.figure()
    sns.distplot(truth)
    plt.plot(probe.cpu().numpy()[:, 0], density_estimate, label="Estimated density")
    plt.title("Approximate Unknown Distribution (sampling estimate)")
    plt.xlabel("Estimated p(x)")
    plt.xlabel("x")
    plt.savefig("14_gt.pdf")

    return network


network = train_discrim(512, 10000)

############### plotting things
############### (1) plot the output of your trained discriminator 
############### (2) plot the estimated density contrasted with the true density

t_xx = T.tensor(xx).to(device).float()
t_xx = t_xx[:, None]
r = network(t_xx)  # evaluate xx using your discriminator; replace xx with the output
r = T.sigmoid(network(t_xx)).squeeze(-1).detach().cpu().numpy()
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.plot(xx,r)
plt.title(r'$D(x)$')

normal = T.distributions.normal.Normal(0, 1)
density = N(xx)

# estimate the density of distribution4 (on xx) using the discriminator;
estimate = r * density / (1 - r + 0.0000001)  # eps for stability
plt.subplot(1,2,2)
plt.plot(xx,estimate)
plt.plot(f(torch.from_numpy(xx)).numpy(), d(torch.from_numpy(xx)).numpy()**(-1)*N(xx))
plt.legend(['Estimated','True'])
plt.title('Estimated vs True')

plt.savefig("14_official.pdf")











