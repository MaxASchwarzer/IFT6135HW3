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


class Discriminator(nn.Module):
    def __init__(self, indim, hdim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(indim, hdim)
        self.relu_1 = nn.ReLU()
        self.fc2 = nn.Linear(hdim, hdim)
        self.relu_2 = nn.ReLU()
        self.fc3 = nn.Linear(hdim, 1)

    def forward(self, x):
        x1 = self.fc1(x)
        x1 = self.relu_1(x1)

        x2 = self.fc2(x1)
        x2 = self.relu_2(x2)
        x3 = self.fc3(x2)

        return x3


def wgan_objective(model, real, fake, lda=100):
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
                           grad_outputs=T.ones(output.shape[0],
                                               device=device))[0]
    grad_penalty = (T.norm(grad + 1e-16, dim=-1) - 1)**2

    d_loss = d_fake.mean() - d_real.mean() + lda*grad_penalty.mean()
    g_loss = -d_fake.mean()

    # Return: discriminator loss, generator loss, and estimated WSD
    return d_loss, g_loss, d_real.mean() - d_fake.mean()


def jsd_objective(model, real, fake):
    d_real = model(real).squeeze(-1)
    d_fake = model(fake).squeeze(-1)

    l_real = F.logsigmoid(d_real)
    l_fake = T.log(1 - T.sigmoid(d_fake))

    # don't include the log2 term, as it's constant wrt parameters
    loss = 0.5*(l_real + l_fake)

    d_loss = -loss.mean()
    g_loss = loss.mean()
    real_jsd = loss.mean() + log2

    # Return: discriminator loss, generator loss, and estimated JSD
    return d_loss, g_loss, real_jsd.mean()


def problem3(bs=512, iters=1000):
    divs = []
    for loss_fcn in [wgan_objective, jsd_objective]:
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
                    print("Estimated divergence: {}".format(div.item()))

            divs[-1].append(div.detach().cpu().numpy())

    plt.figure()
    plt.scatter(np.linspace(-1, 1, 21, endpoint=True), divs[1], label="JSD")
    plt.legend()
    plt.xlabel("Phi")
    plt.ylabel("Estimated divergence")
    plt.title("Estimated JSD for disjoint distributions")
    plt.savefig("3-1.pdf")

    plt.figure()
    plt.scatter(np.linspace(-1, 1, 21, endpoint=True), divs[0], label="Wasserstein Estimate")
    plt.legend()
    plt.xlabel("Phi")
    plt.ylabel("Estimated divergence")
    plt.title("Estimated Wasserstein distance for disjoint distributions")
    plt.savefig("3-2.pdf")


    plt.figure()
    plt.scatter(np.linspace(-1, 1, 21, endpoint=True), divs[0], label="Wasserstein")
    plt.scatter(np.linspace(-1, 1, 21, endpoint=True), divs[1], label="JSD")
    plt.legend()
    plt.xlabel("Phi")
    plt.ylabel("Estimated divergence")
    plt.title("Estimated divergences for disjoint distributions")
    plt.savefig("3-3.pdf")



if __name__ == "__main__":
    problem3(512)

