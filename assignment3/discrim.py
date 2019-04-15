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


class ResBlock(nn.Module):
    def __init__(self, inchannels, outchannels, stride=2, dropout=0.5, bn=False, ln=False, sn=False, h=None):
        super(ResBlock, self).__init__()
        if inchannels != outchannels:
            if stride > 0:
                self.upscaler = (nn.Conv2d(inchannels, outchannels, 3,
                                                                 stride=stride, padding=1))
                if sn:
                    self.upscaler = nn.utils.spectral_norm(self.upscaler)
            elif stride < 0:
                self.upscaler = (nn.ConvTranspose2d(inchannels, outchannels, 4,
                                                                          padding=1, stride=-stride))
            self.relu0 = nn.LeakyReLU(0.2)
            self.upscale = True
        else:
            self.upscale = False

        self.bn = bn or ln

        self.stride = -stride if stride < 0 else 1/stride
        self.conv1 = (nn.Conv2d(outchannels, outchannels, 3, padding=1))
        self.dropout1 = nn.Dropout(dropout)
        if bn:
            self.bn1 = nn.BatchNorm2d(outchannels)
            self.bn2 = nn.BatchNorm2d(outchannels)
        elif ln:
            self.bn1 = nn.LayerNorm((outchannels, h, h))
            self.bn2 = nn.LayerNorm((outchannels, h, h))
        self.conv2 = (nn.Conv2d(outchannels, outchannels, 3, padding=1))
        self.dropout2 = nn.Dropout(dropout)
        if sn:
            self.conv1 = nn.utils.spectral_norm(self.conv1)
            self.conv2 = nn.utils.spectral_norm(self.conv2)

        self.relu1 = nn.LeakyReLU(0.2)
        self.relu2 = nn.LeakyReLU(0.2)

    def forward(self, x):
        if self.upscale:
            # x = F.interpolate(x, scale_factor=self.stride)
            x = self.upscaler(x)
            x = self.relu0(x)
            if self.bn:
                x = self.bn1(x)
        x1 = self.conv1(x)
        x1 = self.relu1(x1)
        if self.bn:
            x1 = self.bn1(x1)
        x1 = self.dropout1(x1)
        x2 = self.conv2(x1)
        x2 = self.relu2(x2)
        if self.bn:
            x2 = self.bn2(x2)
        x2 = self.dropout2(x2)

        output = x2 + x
        return output


class Generator(nn.Module):
    def __init__(self, zdim, hdim, im_channels=3, blocks=[2, 2, 2, 2], dropout=0.5):
        super(Generator, self).__init__()

        self.initial = nn.Linear(zdim, 16*hdim)

        self.blocks = nn.ModuleList()

        current_dim = hdim
        new_dim = hdim
        h = 4
        for nblocks in blocks:
            for block in range(nblocks):
                if new_dim != current_dim:
                    self.blocks.append(ResBlock(current_dim, new_dim, stride=-2, dropout=dropout, bn=True, sn=False, h=h))
                    current_dim = new_dim
                self.blocks.append(ResBlock(current_dim, current_dim, stride=1, dropout=dropout, bn=True, sn=False, h=h))
            new_dim = current_dim//2
            h *= 2

        self.final = (nn.Conv2d(current_dim, im_channels, 7, padding=3))
        self.final_bn = nn.BatchNorm2d(im_channels)

    def forward(self, x):
        current = self.initial(x)
        current = current.view(x.shape[0], -1, 4, 4)

        for block in self.blocks:
            current = block(current)

        # output = 0.5*self.final_bn(T.sigmoid(self.final(current))) + 0.5
        # output = T.sigmoid(self.final_bn(self.final(current)))
        output = T.sigmoid(self.final(current))

        return output


class ConvDiscriminator(nn.Module):
    def __init__(self, hdim, im_channels=3, blocks=[2, 2, 2, 2], dropout=0):
        super(ConvDiscriminator, self).__init__()
        self.blocks = nn.ModuleList()

        self.im_bn = nn.BatchNorm2d(im_channels)
        self.initial = nn.Conv2d(im_channels, hdim, 7, padding=3)
        self.initial_bn = nn.BatchNorm2d(hdim)
        self.relu = nn.LeakyReLU(0.2)

        h = 32
        current_dim = hdim
        new_dim = current_dim
        for nblocks in blocks:
            for block in range(nblocks):
                if new_dim != current_dim:
                    self.blocks.append(ResBlock(current_dim, new_dim, stride=2, dropout=dropout, ln=True, sn=False, h=h))
                    current_dim = new_dim
                self.blocks.append(ResBlock(current_dim, current_dim, stride=1, dropout=dropout, ln=True, sn=False, h=h))
            new_dim = current_dim*2
            h = h//2

        self.fc1 = (nn.Linear(current_dim*16, current_dim))
        self.fc2 = (nn.Linear(current_dim, 1))

    def forward(self, x):
        # x = self.im_bn(x)
        current = self.initial(x)
        current = self.relu(current)
        # current = self.initial_bn(current)

        for block in self.blocks:
            current = block(current)

        current = self.fc1(current.flatten(1, -1))
        current = self.relu(current)
        return self.fc2(current)


class Discriminator(nn.Module):
    def __init__(self, indim, hdim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(indim, hdim)
        self.relu_1 = nn.ReLU()
        self.fc2 = nn.Linear(hdim, hdim)
        self.relu_2 = nn.ReLU()
        self.fc3 = nn.Linear(hdim, 1)

        nn.init.kaiming_uniform_(self.fc1.weight)
        nn.init.kaiming_uniform_(self.fc2.weight)
        nn.init.kaiming_uniform_(self.fc3.weight)

    def forward(self, x):
        x1 = self.fc1(x)
        x1 = self.relu_1(x1)

        x2 = self.fc2(x1)
        x2 = self.relu_2(x2)
        x3 = self.fc3(x2)

        return x3


def wgan_objective(model, real, fake, lda=5):
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

    # print(grad_penalty.mean().item(), d_real.mean().item(), d_fake.mean().item())
    d_loss = d_fake.mean() - d_real.mean() + lda*grad_penalty.mean()
    g_loss = -d_fake.mean()

    return d_loss, g_loss, d_real.mean() - d_fake.mean()


def jsd_objective(model, real, fake):
    d_real = model(real).squeeze(-1)
    d_fake = model(fake).squeeze(-1)

    l_real = F.logsigmoid(d_real)
    l_fake = F.logsigmoid(1 - d_fake)

    # don't include the log2 term, as it's constant wrt parameters
    loss = 0.5*(l_real + l_fake)

    # if T.sigmoid(d_fake).mean() > 0.6:
    #     import ipdb;
    #     ipdb.set_trace()
    # print(T.sigmoid(d_real).mean().item(), T.sigmoid(d_fake).mean().item())
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


