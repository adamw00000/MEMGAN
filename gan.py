# %%
import argparse
import os
import pkbar

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch

os.makedirs("images", exist_ok=True)

# parser = argparse.ArgumentParser()
# parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
# parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
# parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
# parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
# parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
# parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
# parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
# parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
# parser.add_argument("--channels", type=int, default=1, help="number of image channels")
# parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
# opt = parser.parse_args()
# print(opt)

opt = argparse.Namespace(
    n_epochs=100,
    batch_size=128,
    lr=1e-4,
    b1=0.5,
    b2=0.999,
    n_cpu=8,
    latent_dim=256,
    img_size=32,
    channels=1,
    sample_interval=100
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def reparametrization(mu, log_sigma):
    sigma = torch.exp(log_sigma)
    return mu + torch.randn_like(mu) * sigma    


# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()

#         self.init_size = opt.img_size // 4
#         self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

#         self.conv_blocks = nn.Sequential(
#             nn.BatchNorm2d(128),
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(128, 128, 3, stride=1, padding=1),
#             nn.BatchNorm2d(128, 0.8),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(128, 64, 3, stride=1, padding=1),
#             nn.BatchNorm2d(64, 0.8),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
#             nn.Tanh(),
#         )

#     def forward(self, z):
#         out = self.l1(z)
#         out = out.view(out.shape[0], 128, self.init_size, self.init_size)
#         img = self.conv_blocks(out)
#         return img

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def generator_block(in_filters, out_filters, kernel_size=3, stride=1, padding=0,
                relu_slope=0.01, bn_eps=1e-5):
            block = [
                nn.ConvTranspose2d(in_filters, out_filters, kernel_size, stride, padding, bias=False), 
                nn.BatchNorm2d(out_filters, bn_eps),
                nn.LeakyReLU(relu_slope, inplace=True)
            ]
            return block
        
        self.conv_blocks = nn.Sequential(
            *generator_block(opt.latent_dim, 256, kernel_size=4, stride=1),
            *generator_block(256, 128, kernel_size=4, stride=2),
            *generator_block(128, 64, kernel_size=4, stride=1),
            *generator_block(64, 32, kernel_size=4, stride=2),
            *generator_block(32, 32, kernel_size=5, stride=1),
            *generator_block(32, 32, kernel_size=1, stride=1),
            nn.ConvTranspose2d(32, opt.channels, 1, stride=1, bias=False),
        )

        self.output_bias = nn.Parameter(torch.zeros(opt.channels, 32, 32), requires_grad=True)

    def forward(self, z):
        output = self.conv_blocks(z.view(*z.shape, 1, 1))
        output = torch.sigmoid(output + self.output_bias)
        return output


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        def encoder_block(in_filters, out_filters, kernel_size=5, stride=1, padding=0,
                relu_slope=0.01, bn_eps=1e-5):
            block = [
                nn.Conv2d(in_filters, out_filters, kernel_size, stride, padding, bias=False), 
                nn.BatchNorm2d(out_filters, bn_eps),
                nn.LeakyReLU(relu_slope, inplace=True), 
            ]
            return block

        self.model = nn.Sequential(
            *encoder_block(opt.channels, 32, kernel_size=5, stride=1),
            *encoder_block(32, 64, kernel_size=4, stride=2),
            *encoder_block(64, 128, kernel_size=4, stride=1),
            *encoder_block(128, 256, kernel_size=4, stride=2),
            *encoder_block(256, 512, kernel_size=4, stride=1),
            *encoder_block(512, 512, kernel_size=1, stride=1),
            # 2 * latent = (mu, sigma)
            nn.Conv2d(512, 2 * opt.latent_dim, 1, stride=1, bias=True),
            nn.Flatten()
        )

    def forward(self, x):
        out = self.model(x)
        return out


class DiscriminatorXZ(nn.Module):
    def __init__(self):
        super(DiscriminatorXZ, self).__init__()

        def discriminator_block(in_filters, out_filters, kernel_size=3, stride=2, padding=0, bias=False,
                bn=True, dropout=0.2, relu_slope=0.01, bn_eps=1e-5):
            block = [
                nn.Conv2d(in_filters, out_filters, kernel_size, stride, padding, bias=bias), 
                nn.LeakyReLU(relu_slope, inplace=True), 
                nn.Dropout2d(dropout)
            ]
            if bn:
                block.insert(1, nn.BatchNorm2d(out_filters, bn_eps))
            return block

        self.x_block = nn.Sequential(
            *discriminator_block(opt.channels, 32, kernel_size=5, stride=1, bn=False, bias=True),
            *discriminator_block(32, 64, kernel_size=4, stride=2),
            *discriminator_block(64, 128, kernel_size=4, stride=1),
            *discriminator_block(128, 256, kernel_size=4, stride=2),
            *discriminator_block(256, 512, kernel_size=4, stride=1),
        )
        self.z_block = nn.Sequential(
            *discriminator_block(opt.latent_dim, 512, kernel_size=1, stride=1, bn=False),
            *discriminator_block(512, 512, kernel_size=1, stride=1, bn=False),
        )
        self.joint_block = nn.Sequential(
            *discriminator_block(1024, 1024, kernel_size=1, stride=1, bn=False, bias=True),
            *discriminator_block(1024, 1024, kernel_size=1, stride=1, bn=False, bias=True),
        )
        self.adv_layer = nn.Sequential(
            # nn.Linear(1024, 1),
            nn.Conv2d(1024, 1, 1, stride=1, bias=True), 
            nn.Sigmoid(),
            nn.Flatten()
        )

    def forward(self, x, z):
        x_repr = self.x_block(x)
        z_repr = self.z_block(z.view(*z.shape, 1, 1))
        joint_repr = torch.cat([x_repr, z_repr], dim=1)
        out = self.joint_block(joint_repr)
        # out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
encoder = Encoder()
discriminator = DiscriminatorXZ()

generator.to(device)
encoder.to(device)
discriminator.to(device)
adversarial_loss.to(device)

# Initialize weights
generator.apply(weights_init_normal)
encoder.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader
os.makedirs("../../data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_GE = torch.optim.Adam([*generator.parameters(), *encoder.parameters()], lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    kbar = pkbar.Kbar(target=len(dataloader), epoch=epoch, num_epochs=opt.n_epochs, width=8, always_stateful=False)
    for i, (imgs, _) in enumerate(dataloader):

        # Adversarial ground truths
        label_real = Variable(torch.ones((imgs.shape[0], 1)).to(device), requires_grad=False)
        label_fake = Variable(torch.zeros((imgs.shape[0], 1)).to(device), requires_grad=False)

        # Configure input
        imgs_real = Variable(imgs.float().to(device))

        # Prepare discriminator noise
        noise_real = torch.normal(0, 0.1 * (opt.n_epochs - epoch) / opt.n_epochs, imgs_real.shape).to(device)
        noise_fake = torch.normal(0, 0.1 * (opt.n_epochs - epoch) / opt.n_epochs, imgs_real.shape).to(device)

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_GE.zero_grad()

        # Sample noise as generator input
        z_fake = Variable(torch.normal(0, 1, (imgs.shape[0], opt.latent_dim)).float().to(device))
        # Generate a batch of images
        imgs_fake = generator(z_fake)

        # Encode real images
        encoder_out = encoder(imgs_real)
        z_mu, z_log_sigma = encoder_out[:, :opt.latent_dim], encoder_out[:, opt.latent_dim:]
        z_real = reparametrization(z_mu, z_log_sigma)

        # Loss measures generator's ability to fool the discriminator
        # Real images loss: should be classified as fake
        output_real = discriminator(imgs_real + noise_real, z_real)
        loss_real = adversarial_loss(output_real, label_fake)
        # Generated images loss: vice versa
        output_fake = discriminator(imgs_fake + noise_fake, z_fake)
        loss_fake = adversarial_loss(output_fake, label_real)

        loss_g = loss_real + loss_fake

        loss_g.backward()
        optimizer_GE.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Detach generated/encoded vectors for discriminator training
        imgs_fake = imgs_fake.detach()
        z_real = z_real.detach()

        # Measure discriminator's ability to classify real from generated samples
        # Now labels will be correct
        output_real = discriminator(imgs_real + noise_real, z_real)
        output_fake = discriminator(imgs_fake + noise_fake, z_fake)
        loss_real = adversarial_loss(output_real, label_real)
        loss_fake = adversarial_loss(output_fake, label_fake)
        loss_d = loss_real + loss_fake

        loss_d.backward()
        optimizer_D.step()
        
        # if (i + 1) % 100 == 0 or (i + 1) == len(dataloader):
        #     # print(
        #     #     "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
        #     #     % (epoch + 1, opt.n_epochs, i + 1, len(dataloader), loss_d.item(), loss_g.item())
        #     # )
        #     print(f"Epoch: {epoch:4}, Iter: {i:4} ||| " + \
        #         f"D Loss: {loss_d.item():.4f}, G loss: {loss_g.item():.4f}, " + \
        #         f"D(x): {output_real.mean().item():.4f}, " + \
        #         f"D(G(x)): {output_fake.mean().item():.4f}")
        kbar.update(i, values=[("D Loss", loss_d.item()), ("G loss", loss_g.item()), ("D(x)", output_real.mean().item()), ("D(G(x))", output_fake.mean().item())])

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(imgs_fake.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

    # newline for kbar
    print()

# %%
