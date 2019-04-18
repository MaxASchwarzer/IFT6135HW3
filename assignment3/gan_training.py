import torch as T
from torch import nn
from torch import optim
from torch import utils
from torch.utils.data import dataset
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import seaborn as sns
import argparse
import time
import pickle as pkl
import os
from samplers import distribution1, distribution2, distribution3, distribution4
from discrim import wgan_objective
from classify_svhn import get_data_loader
if T.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


class ResBlock(nn.Module):
    def __init__(self, inchannels, outchannels, stride=2, dropout=0.5, bn=False, ln=False, sn=False, h=None):
        super(ResBlock, self).__init__()
        if inchannels != outchannels:
            if stride > 0:
                self.rescaler = nn.Sequential(
                                    nn.Conv2d(inchannels, outchannels, 3, padding=1, stride=stride),
                                    nn.LeakyReLU(0.2)
                                    )
                if sn:
                    self.rescaler = nn.utils.spectral_norm(self.rescaler)
            elif stride < 0:
                self.rescaler = nn.Sequential(
                                    nn.Conv2d(inchannels, outchannels, 3, padding=1, stride=1),
                                    nn.LeakyReLU(0.2),
                                    )
            self.rescale = True
        else:
            self.rescale = False

        self.bn = bn or ln

        self.stride = stride
        self.conv1 = (nn.Conv2d(outchannels, outchannels, 3, padding=1))
        self.dropout1 = nn.Dropout(dropout)
        if bn:
            self.bn0 = nn.BatchNorm2d(outchannels)
            self.bn1 = nn.BatchNorm2d(outchannels)
            self.bn2 = nn.BatchNorm2d(outchannels)
        elif ln:
            self.bn0 = nn.LayerNorm((outchannels, h, h))
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
        if self.rescale:
            x = self.rescaler(x)

            # Can't upsample inside the sequential, sadly
            if self.stride < 0:
                x = F.interpolate(x, scale_factor=-self.stride, mode="bilinear", align_corners=True)

            if self.bn:
                x = self.bn0(x)
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
    def __init__(self, zdim, hdim, im_channels=3, blocks=[2, 2, 2, 2], dropout=0.2):
        super(Generator, self).__init__()

        self.initial = nn.Linear(zdim, 16*hdim)

        self.blocks = nn.ModuleList()

        current_dim = hdim
        new_dim = hdim//2
        h = 8
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

        output = T.sigmoid(self.final(current))
        return output


class ConvDiscriminator(nn.Module):
    def __init__(self, hdim, im_channels=3, blocks=[2, 2, 2, 2], dropout=0):
        super(ConvDiscriminator, self).__init__()
        self.blocks = nn.ModuleList()

        self.initial = nn.Conv2d(im_channels, hdim, 7, padding=3)
        self.relu = nn.LeakyReLU(0.2)

        h = 16
        current_dim = hdim
        new_dim = current_dim*2
        for nblocks in blocks:
            for block in range(nblocks):
                if new_dim != current_dim:
                    self.blocks.append(ResBlock(current_dim, new_dim, stride=2, dropout=dropout, ln=True, sn=False, h=h))
                    current_dim = new_dim
                self.blocks.append(ResBlock(current_dim, current_dim, stride=1, dropout=dropout, ln=True, sn=False, h=h))
            new_dim = current_dim*2
            h = h//2

        self.fc1 = (nn.Linear(current_dim*16, current_dim))
        self.ln1 = nn.LayerNorm(current_dim, elementwise_affine=False)
        self.fc2 = (nn.Linear(current_dim, 1))

    def forward(self, x):
        current = self.initial(x)
        current = self.relu(current)

        for block in self.blocks:
            current = block(current)

        current = self.fc1(current.flatten(1, -1))
        current = self.ln1(current)
        current = self.relu(current)
        return self.fc2(current)


class GAN(nn.Module):
    def __init__(self, zdim=100, channels_g=32, channels_d=32, blocks=[1, 1, 1, 1, 1]):
        super(GAN, self).__init__()
        self.generator = Generator(zdim=zdim, im_channels=3, hdim=channels_g*2**(len(blocks)-1), blocks=blocks)
        self.discriminator = ConvDiscriminator(im_channels=3, hdim=channels_d, blocks=blocks,)
        self.distribution = T.distributions.normal.Normal(0, 1)
        self.d_optim = optim.Adam(self.discriminator.parameters(), 1e-4, betas=(0.5, 0.9), eps=1e-6)
        self.g_optim = optim.Adam(self.generator.parameters(), 1e-4, betas=(0.5, 0.9), eps=1e-6)

        self.zdim = zdim

    def sample(self, batch_size):
        self.train()
        sample_z = self.distribution.sample((batch_size, self.zdim)).to(device)
        sample_images = self.generator(sample_z)
        # print(T.var(sample_images.flatten(1, -1), 0).mean(), T.mean(sample_images.flatten(2, -1), -1).mean(0))

        return sample_images

    def sample_numpy(self, batch_size=32, total=1000, save="GAN_Samples"):
        self.eval()
        with T.no_grad():
            images = []
            for batch in range(int(np.ceil(total/batch_size))):
                sample_z = self.distribution.sample((batch_size, self.zdim)).to(device)
                sample_images = self.generator(sample_z)
                images.append(sample_images)

        images = T.cat([im.detach().cpu() for im in images], 0)[:total]

        if save is not None:
            if not os.path.isdir(save):
                os.mkdir(save)

            for i, im in enumerate(images):
                torchvision.utils.save_image(im, filename=save + "/{}.png".format(i))

            images = images.numpy()
            np.savez(save + "/samples.npz", images=images)

        return images

    def perturb_sample(self, save="perturbed_images"):
        self.eval()
        with T.no_grad():
            sample_z = self.distribution.sample((1, self.zdim)).to(device)
            image_original = self.generator(sample_z)
            images = [image_original]

            for i in range(self.zdim):
                perturbed = sample_z.clone()
                perturbed[:, i] += 1.
                images.append(self.generator(perturbed))

            if not os.path.isdir(save):
                os.mkdir(save)
            for i, im in enumerate(images):
                torchvision.utils.save_image(im, filename=save + "/{}.png".format(i))

            images = [im.detach().cpu().numpy() for im in images]
            images = np.concatenate(images, 0)
            np.savez(save+"/images.npz", images=images)

    def interpolate(self, save="interpolations"):
        self.eval()
        with T.no_grad():
            alphas = np.linspace(0, 1, 11, endpoint=True)
            sample_1 = self.distribution.sample((1, self.zdim)).to(device)
            sample_2 = self.distribution.sample((1, self.zdim)).to(device)
            image_1 = self.generator(sample_1)
            image_2 = self.generator(sample_2)

            im_interps = []
            for alpha in alphas:
                im_interps.append(image_1*alpha + image_2*(1 - alpha))

            latent_interps = []
            for alpha in alphas:
                latent_interps.append(self.generator(sample_1*alpha + sample_2*(1 - alpha)))

            im_interps = T.cat([im.detach().cpu() for im in im_interps], 0)
            latent_interps = T.cat([im.detach().cpu() for im in latent_interps], 0)

            if not os.path.isdir(save):
                os.mkdir(save)
            torchvision.utils.save_image(latent_interps, filename=save + "/latent_interps.png", nrow=16)
            torchvision.utils.save_image(im_interps, filename=save + "/pixel_interps.png", nrow=16)

        np.savez(save, im_interps=im_interps.numpy(), latent_interps=latent_interps.numpy())


def train(flags):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    # Just use the torch dataset for that autodownloading goodness.
    trainvalid = torchvision.datasets.SVHN("./svhn/",
                                            split='train',
                                            transform=transform,
                                            target_transform=None,
                                            download=True)
    trainset_size = int(len(trainvalid) * 0.9)
    trainset, validset = dataset.random_split(
        trainvalid,
        [trainset_size, len(trainvalid) - trainset_size]
    )

    loader = T.utils.data.DataLoader(
        trainset,
        batch_size=flags.batch_size,
        shuffle=True,
        num_workers=2
    )


    if flags.load is not None:
        model = T.load(flags.load)
        with open(flags.load+"log", "rb") as f:
            log = pkl.load(f, encoding="latin1")
    else:
        model = GAN(flags.zdim, flags.channels_g, flags.channels_d, blocks=[flags.blocks for i in range(3)])
        log = []
    model = model.to(device)

    old_time = time.clock()
    i = len(log)*flags.disc_iters
    while i < flags.iters*flags.disc_iters:
        for real, _ in loader:
            i += 1
            if i % flags.disc_iters != 0:
                with T.no_grad():
                    samples = model.sample(real.shape[0])
            else:
                samples = model.sample(real.shape[0])
            real = real.to(device)
            d_loss, g_loss, div = wgan_objective(model.discriminator, real, samples)
            d_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(model.discriminator.parameters(), 1)
            model.d_optim.step()
            model.d_optim.zero_grad()
            if i % flags.disc_iters == 0:
                log.append(div.item())
                model.g_optim.zero_grad()
                g_loss.backward()
                nn.utils.clip_grad_norm_(model.generator.parameters(), 1)
                model.g_optim.step()
                model.g_optim.zero_grad()

            if i > flags.iters:
                break
            if i % flags.print_freq == 0:
                current_time = time.clock()
                per_iter = (current_time - old_time)/flags.print_freq
                old_time = current_time
                print("Iter {0}; Generator gap: {1:.4f}; time per iter: {2:.4f}".format(i, div.item(), per_iter))

            if i % flags.sample_freq == 0:
                file = flags.save.replace(".pt", "") + str(i) + ".png"
                print("Saving samples in {}".format(file))
                torchvision.utils.save_image(samples.detach().cpu(), file, normalize=False)
                raw = samples.detach().cpu().flatten(2, -1).mean((-1)).numpy()
                plt.figure()
                sns.distplot(raw[:, 0])
                sns.distplot(raw[:, 1])
                sns.distplot(raw[:, 2])
                plt.savefig(flags.save + "{}sample_dist.pdf".format(i))
                plt.close('all')
                raw = real.detach().cpu().flatten(2, -1).mean((-1)).numpy()
                plt.figure()
                sns.distplot(raw[:, 0])
                sns.distplot(raw[:, 1])
                sns.distplot(raw[:, 2])
                plt.savefig(flags.save + "{}real_dist.pdf".format(i))
                plt.close('all')
            if i % flags.save_freq == 0:
                print("Saving model in {}".format(flags.save))
                T.save(model, flags.save)
                with open(flags.save+"log", "wb") as f:
                    pkl.dump(log, f)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='WGAN-GP training.')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument("--iters", type=int, default=2000000, help="Number of iters to train for")
    parser.add_argument("--disc_iters", type=int, default=4, help="Number of disc iters per gen iter")
    parser.add_argument("--sample_freq", type=int, default=1000, help="How often to sample")
    parser.add_argument("--print_freq", type=int, default=10, help="How often to print")
    parser.add_argument("--save_freq", type=int, default=1000, help="How often to save")
    parser.add_argument("--load", type=str, default=None, help="File to load model from.  Blank for none.")
    parser.add_argument("--save", type=str, default="wgan.pt", help="File to save model to.")
    parser.add_argument("--zdim", type=int, default=100, help="Dimension of latent")
    parser.add_argument("--blocks", type=int, default=2, help="# of resblocks per pooling")
    parser.add_argument("--channels_g", type=int, default=32, help="Base # of channels in g resnets.")
    parser.add_argument("--channels_d", type=int, default=32, help="Base # of channels in d resnets.")

    flags = parser.parse_args()
    model = train(flags)
    model.sample_numpy()
    model.perturb_sample()
    model.interpolate()




