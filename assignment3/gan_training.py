import torch as T
from torch import nn
from torch import optim
from torch import utils
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import seaborn as sns
import argparse
import time
from samplers import distribution1, distribution2, distribution3, distribution4
if T.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


from discrim import wgan_objective, Generator, ConvDiscriminator
# DIM = 128


# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()
#         preprocess = nn.Sequential(
#             nn.Linear(100, 4 * 4 * 4 * DIM),
#         )
#
#         block1 = nn.Sequential(
#             nn.BatchNorm2d(4 * DIM),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(4 * DIM, 2 * DIM, 2, stride=2),
#             nn.BatchNorm2d(2 * DIM),
#             nn.ReLU(True),
#         )
#         block2 = nn.Sequential(
#             nn.ConvTranspose2d(2 * DIM, DIM, 2, stride=2),
#             nn.BatchNorm2d(DIM),
#             nn.ReLU(True),
#         )
#         deconv_out = nn.ConvTranspose2d(DIM, 3, 2, stride=2)
#
#         self.preprocess = preprocess
#         self.block1 = block1
#         self.block2 = block2
#         self.deconv_out = deconv_out
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, input):
#         output = self.preprocess(input)
#         output = output.view(-1, 4 * DIM, 4, 4)
#         output = self.block1(output)
#         output = self.block2(output)
#         output = self.deconv_out(output)
#         output = self.sigmoid(output)
#         return output.view(-1, 3, 32, 32)
#
#
# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         main = nn.Sequential(
#             nn.Conv2d(3, DIM, 3, 2, padding=1),
#             nn.LeakyReLU(),
#             nn.Conv2d(DIM, 2 * DIM, 3, 2, padding=1),
#             nn.LeakyReLU(),
#             nn.Conv2d(2 * DIM, 4 * DIM, 3, 2, padding=1),
#             nn.LeakyReLU(),
#         )
#
#         self.main = main
#         self.linear = nn.Linear(4*4*4*DIM, 1)
#
#     def forward(self, input):
#         output = self.main(input)
#         output = output.view(-1, 4*4*4*DIM)
#         output = self.linear(output)
#         return output

class GAN(nn.Module):
    def __init__(self, zdim=100, channels_g=32, channels_d=32, blocks=[1, 1, 1, 1, 1]):
        super(GAN, self).__init__()
        self.generator = Generator(zdim=zdim, im_channels=3, hdim=channels_g*2**(len(blocks)-1), blocks=blocks)
        self.discriminator = ConvDiscriminator(im_channels=3, hdim=channels_d, blocks=blocks,)
        self.distribution = T.distributions.normal.Normal(0, 1)
        self.d_optim = optim.Adam(self.discriminator.parameters(), 4e-4, betas=(0.5, 0.9))
        self.g_optim = optim.Adam(self.generator.parameters(), 1e-4, betas=(0.5, 0.9))

        self.zdim = zdim

    def sample(self, batch_size):
        sample_z = self.distribution.sample((batch_size, self.zdim)).to(device)
        sample_images = self.generator(sample_z)
        print(T.var(sample_images.flatten(1, -1), 0).mean(), T.mean(sample_images.flatten(2, -1), -1).mean(0))

        return sample_images

    def sample_numpy(self, batch_size=32, total=1000, save="GAN_Samples"):
        with T.no_grad():
            images = []
            for batch in range(int(np.ceil(batch_size/total))):
                sample_z = self.distribution.sample((batch_size, self.zdim)).to(device)
                sample_images = self.generator(sample_z)
                images.append(sample_images)

        images = np.concatenate(images.detach().cpu().numpy(), 0)[:total]
        if save is not None:
            np.savez(save, images=images)
        return images

    def perturb_sample(self, save="perturbed_images"):
        with T.no_grad():
            sample_z = self.distribution.sample((1, self.zdim)).to(device)
            image_original = self.generator(sample_z)
            images = [image_original]

            for i in range(self.zdim):
                perturbed = T.copy(sample_z)
                perturbed[:, i] + 0.1
                images.append(self.generator(perturbed))

            images = [im.detach().cpu().numpy() for im in images]
            images = np.concatenate(images, 0)
            torchvision.utils.save_image(images)
            np.savez(save, images=images)

    def interpolate(self, save="interpolations"):
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

            im_interps = [im.detach().cpu().numpy() for im in im_interps]
            latent_interps = [im.detach().cpu().numpy() for im in latent_interps]

        np.savez(save, im_interps=im_interps, latent_interps=latent_interps)


def train(flags):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    dataset = torchvision.datasets.SVHN("./svhn/",
                                        split='train',
                                        transform=transform,
                                        target_transform=None,
                                        download=True)
    loader = utils.data.DataLoader(dataset, batch_size=flags.batch_size, shuffle=True)
    if flags.load is not None:
        model = T.load(flags.load)
    else:
        model = GAN(flags.zdim, flags.channels_g, flags.channels_d, blocks=[flags.blocks for i in range(4)])
    model = model.to(device)

    old_time = time.clock()
    i = 0
    while i < flags.iters:
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
                torchvision.utils.save_image(samples.detach().cpu(), file)
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




