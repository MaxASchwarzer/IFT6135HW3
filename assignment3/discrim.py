import torch as T
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from samplers import distribution1, distribution2, distribution3, distribution4
if T.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
log2 = np.log(2)

def wgan_objective(model, real, fake, lda=10):
    d_real = model(real)
    d_fake = model(fake)

    alpha = T.rand((real.shape[0]), device=real.device)
    for i in range(len(real.shape) - 1):
        alpha.unsqueeze_(-1)
    interps = alpha*real.data + (1 - alpha)*fake.data
    interps.requires_grad = True

    output = model(interps).squeeze(-1)

    grad = T.autograd.grad(output, interps,
                           retain_graph=True, create_graph=True,
                           grad_outputs=T.ones(output.shape[0], device=device))[0]
    grad_penalty = (T.norm(grad + 1e-16, dim=-1) - 1)**2
    # offset_penalty = (d_real.mean() + d_fake.mean())**2

    print(grad_penalty.mean().item(), d_real.mean().item(), d_fake.mean().item())#, offset_penalty.item())
    d_loss = d_fake.mean() - d_real.mean() + lda*grad_penalty.mean()# + offset_penalty
    g_loss = -d_fake.mean()

    return d_loss, g_loss, d_real.mean() - d_fake.mean()


def jsd_objective(model, real, fake):
    d_real = model(real).squeeze(-1)
    d_fake = model(fake).squeeze(-1)

    l_real = F.logsigmoid(d_real)
    l_fake = F.logsigmoid(1 - d_fake)

    # don't include the log2 term, as it's constant wrt parameters
    loss = 0.5*(l_real + l_fake)

    d_loss = -loss.mean()
    g_loss = loss.mean()
    real_jsd = loss.mean() + log2

    return d_loss, g_loss, real_jsd.mean()


def problem3(bs=512, iters=1000):
    divs = []
    for loss_fcn in [jsd_objective, wgan_objective]:
        divs.append([])
        for theta in np.linspace(1, -1, 21, endpoint=True):
            network = Discriminator(2, 256)
            network = network.to(device)
            optimizer = optim.SGD(network.parameters(), 1e-2, 0.9)
            i = 0
            for real, fake in zip(distribution1(0, bs), distribution1(theta, bs)):
                i += 1
                real = T.tensor(real, device=device).float()
                fake = T.tensor(fake, device=device).float()

                dl, gl, div = loss_fcn(network, real, fake)
                dl.sum().backward()
                nn.utils.clip_grad_norm_(network.parameters(), 1)
                optimizer.step()
                optimizer.zero_grad()
                if i > iters:
                    break
                elif i % 99 == 0:
                    print(div.item())

            divs[-1].append(div.detach().cpu().numpy())

    plt.figure()
    plt.scatter(np.linspace(-1, 1, 21, endpoint=True), divs[0], label="JSD")
    plt.legend()
    plt.xlabel("Theta")
    plt.ylabel("Estimated divergence")
    plt.title("Estimated JSD for disjoint distributions")
    plt.savefig("3-1.pdf")

    plt.figure()
    plt.scatter(np.linspace(-1, 1, 21, endpoint=True), divs[1], label="Wasserstein Estimate")
    plt.legend()
    plt.xlabel("Theta")
    plt.ylabel("Estimated divergence")
    plt.title("Estimated Wasserstein distance for disjoint distributions")
    plt.savefig("3-2.pdf")


def problem4(bs=512, iters=10000):
    loss_fcn = jsd_objective
    normal = T.distributions.normal.Normal(0, 1)
    network = Discriminator(1, 128)
    network = network.to(device)
    optimizer = optim.SGD(network.parameters(), 1e-3, 0.9)
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
    density_estimate = dx*density/(1 - dx + 0.0001)

    plt.figure()
    plt.plot(probe.cpu().numpy()[:, 0], density_estimate, label="Estimated density")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("Estimated density p(x)")
    plt.title("Estimated pdf of Distribution 4")
    plt.savefig("4-1.pdf")


if __name__ == "__main__":
    problem3(512)
    problem4(512)


