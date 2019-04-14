import torch as T
from torch import nn
from torch import optim
from torch import utils
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import argparse
import time
from samplers import distribution1, distribution2, distribution3, distribution4
if T.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


from discrim import wgan_objective, Generator, ConvDiscriminator

class GAN(nn.Module):
    def __init__(self, zdim=100, channels=32, blocks=[1, 1, 1, 1, 1]):
        super(GAN, self).__init__()
        self.generator = Generator(zdim=zdim, im_channels=3, hdim=channels*2**(len(blocks)-1), blocks=blocks)
        self.discriminator = ConvDiscriminator(im_channels=3, hdim=channels, blocks=blocks,)
        self.distribution = T.distributions.normal.Normal(0, 1)
        self.d_optim = optim.Adam(self.discriminator.parameters(), 1e-4, betas=(0.5, 0.9))
        self.g_optim = optim.Adam(self.generator.parameters(), 1e-4, betas=(0.5, 0.9))

        self.zdim = zdim

    def sample(self, batch_size):
        sample_z = self.distribution.sample((batch_size, self.zdim)).to(device)
        sample_images = self.generator(sample_z)
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
    dataset = torchvision.datasets.SVHN("./svhn/",
                                        split='train',
                                        transform=torchvision.transforms.ToTensor(),
                                        target_transform=None,
                                        download=True)
    loader = utils.data.DataLoader(dataset, batch_size=flags.batch_size, shuffle=True)
    if flags.load is not None:
        model = T.load(flags.load)
    else:
        model = GAN(flags.zdim, flags.channels, blocks=[flags.blocks for i in range(5)])
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
                torchvision.utils.save_image(samples[:16].detach().cpu(), file)
            if i % flags.save_freq == 0:
                print("Saving model in {}".format(flags.save))
                T.save(model, flags.save)



    return model



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='WGAN-GP training.')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument("--iters", type=int, default=100000, help="Number of iters to train for")
    parser.add_argument("--disc_iters", type=int, default=4, help="Number of disc iters per gen iter")
    parser.add_argument("--sample_freq", type=int, default=1000, help="How often to sample")
    parser.add_argument("--print_freq", type=int, default=10, help="How often to print")
    parser.add_argument("--save_freq", type=int, default=1000, help="How often to save")
    parser.add_argument("--load", type=str, default=None, help="File to load model from.  Blank for none.")
    parser.add_argument("--save", type=str, default="wgan.pt", help="File to save model to.")
    parser.add_argument("--zdim", type=int, default=100, help="Dimension of latent")
    parser.add_argument("--blocks", type=int, default=2, help="# of resblocks per pooling")
    parser.add_argument("--channels", type=int, default=8, help="Base # of channels in resnets.")

    flags = parser.parse_args()
    model = train(flags)
    model.sample_numpy()
    model.perturb_sample()
    model.interpolate()




