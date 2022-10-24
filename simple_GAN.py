# %%
import argparse
import os
import pkbar
from datetime import datetime
import numpy as np

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torchvision import datasets
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils

from tensorboardX import SummaryWriter

opt = argparse.Namespace(
    n_epochs=400,
    # n_epochs=10,
    batch_size=128,
    lr=2e-4,
    b1=0.5,
    b2=0.999,
    n_cpu=8,
    latent_dim=64, # should be == dim_memory
    img_size=32,
    channels=3,
    n_memory=100,
    dim_memory=64,
    # sample_interval=400,
    # dataset='CIFAR',
    dataset='MNIST',
)

IMAGE_DIR = f"images_GAN_{opt.dataset}"

os.makedirs(IMAGE_DIR, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

current_time = datetime.now().strftime('%b%d_%H-%M-%S')
logdir = os.path.join('runs', opt.dataset, current_time)
writer = SummaryWriter(logdir)

if opt.dataset == 'MNIST':
    opt.channels = 1
    opt.img_size = 28


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def reparametrization(encoder_out):
    mu, log_sigma = encoder_out[:, :opt.latent_dim], encoder_out[:, opt.latent_dim:]
    sigma = torch.exp(log_sigma)
    return mu + torch.randn_like(mu) * sigma    


def get_log_odds(raw_marginals):
    marginals = torch.clamp(raw_marginals.mean(dim=0), 1e-7, 1 - 1e-7)
    return torch.log(marginals / (1 - marginals))


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def generator_block(in_filters, out_filters, kernel_size=4, stride=2, padding=1,
                bn_eps=1e-5, bias=False):
            block = [
                nn.ConvTranspose2d(in_filters, out_filters, kernel_size, stride, padding, bias=bias), 
                nn.BatchNorm2d(out_filters, bn_eps),
                nn.ReLU(inplace=True), 
            ]
            return block

        ngf = 32
        
        self.conv_blocks = nn.Sequential(
            *generator_block(opt.latent_dim, ngf * 4, 4, 1, 0, bias=False),
            *generator_block(ngf * 4, ngf * 2, 3, 2, 1, bias=False),
            *generator_block(ngf * 2, ngf * 1, 4, 2, 1, bias=False),
            nn.ConvTranspose2d(ngf, opt.channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        img = self.conv_blocks(z.view(*z.shape, 1, 1))
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, kernel_size=4, stride=2, padding=1, bias=False,
                bn=True, dropout=0.2, relu_slope=0.1, bn_eps=1e-5):
            block = [
                nn.Conv2d(in_filters, out_filters, kernel_size, stride, padding, bias=bias), 
                nn.LeakyReLU(relu_slope, inplace=True), 
                nn.Dropout2d(dropout),
            ]
            if bn:
                block.insert(1, nn.BatchNorm2d(out_filters, bn_eps))
            return block

        ndf = 32

        self.x_block = nn.Sequential(
            *discriminator_block(opt.channels, ndf, 4, 2, 1, bn=False),
            *discriminator_block(ndf * 1, ndf * 2, 4, 2, 1),
            *discriminator_block(ndf * 2, ndf * 4, 3, 2, 1),
            *discriminator_block(ndf * 4, 1, 4, 1, 0),
            nn.Flatten(),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.x_block(x)

# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

generator.to(device)
discriminator.to(device)
adversarial_loss.to(device)

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader
if opt.dataset == 'MNIST':
    os.makedirs("../../data/mnist", exist_ok=True)
    dataset_train = datasets.MNIST(
        "../../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                # transforms.Resize(opt.img_size), 
                transforms.ToTensor(), 
                transforms.Normalize([0.5], [0.5])
            ]
        ),
    )
    dataset_test = datasets.MNIST(
        "../../data/mnist",
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                # transforms.Resize(opt.img_size), 
                transforms.ToTensor(), 
                transforms.Normalize([0.5], [0.5])
            ]
        ),
    )

    # https://discuss.pytorch.org/t/how-to-use-one-class-of-number-in-mnist/26276/17
    is_inlier = torch.tensor(dataset_train.targets) == 0
    dataset_train = torch.utils.data.dataset.Subset(dataset_train, np.where(is_inlier)[0])
    # https://discuss.pytorch.org/t/change-labels-in-data-loader/36823/9
    dataset_test.targets = list(np.where(dataset_test.targets == 0, 1, 0))

    dataloader_train = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../../data/mnist",
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.Resize(opt.img_size), 
                    transforms.ToTensor(), 
                    transforms.Normalize([0.5], [0.5])
                ]
            ),
    ),
        batch_size=opt.batch_size,
        shuffle=True,
    )
    dataloader_test = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../../data/mnist",
            train=False,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.Resize(opt.img_size), 
                    transforms.ToTensor(), 
                    transforms.Normalize([0.5], [0.5])
                ]
            ),
        ),
        batch_size=opt.batch_size,
        shuffle=True,
    )
elif opt.dataset == 'CIFAR':
    os.makedirs("./data/cifar", exist_ok=True)
    dataset_train = datasets.CIFAR10(
        "./data/cifar",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize(opt.img_size), 
                transforms.ToTensor(), 
                transforms.Normalize([0.5], [0.5])
            ]
        ),
    )
    dataset_test = datasets.CIFAR10(
        "./data/cifar",
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize(opt.img_size), 
                transforms.ToTensor(), 
                transforms.Normalize([0.5], [0.5])
            ]
        ),
    )

    # https://discuss.pytorch.org/t/how-to-use-one-class-of-number-in-mnist/26276/17
    is_inlier = torch.tensor(dataset_train.targets) == 0
    dataset_train = torch.utils.data.dataset.Subset(dataset_train, np.where(is_inlier)[0])
    # https://discuss.pytorch.org/t/change-labels-in-data-loader/36823/9
    dataset_test.targets = list(np.where(torch.tensor(dataset_test.targets) == 0, 1, 0))

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=opt.batch_size,
        shuffle=True,
    )
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=opt.batch_size,
        shuffle=True,
    )
else:
    raise NotImplementedError()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# For visualization
vis_rows = 8
vis_noise = torch.normal(0, 1, (vis_rows ** 2, opt.latent_dim)).float().to(device)

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    kbar = pkbar.Kbar(target=len(dataloader_train), epoch=epoch, num_epochs=opt.n_epochs, width=8, always_stateful=False)
    for i, (imgs, _) in enumerate(dataloader_train):
        batch_size = imgs.shape[0]

        # Adversarial ground truths
        # label_real = Variable(torch.ones((batch_size, 1)).to(device), requires_grad=False)
        # label_fake = Variable(torch.zeros((batch_size, 1)).to(device), requires_grad=False)
        # Label smoothing
        label_real = Variable(torch.normal(1, 0.1, (batch_size, 1)).to(device), requires_grad=False)
        label_fake = Variable(torch.normal(0, 0.1, (batch_size, 1)).to(device), requires_grad=False)

        # Configure input
        imgs_real = Variable(imgs.float().to(device))

        # Prepare discriminator noise
        noise_real = torch.normal(0, 0.1 * (opt.n_epochs - epoch) / opt.n_epochs, imgs_real.shape).to(device)
        noise_fake = torch.normal(0, 0.1 * (opt.n_epochs - epoch) / opt.n_epochs, imgs_real.shape).to(device)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as generator input
        z_fake = Variable(torch.normal(0, 1, (batch_size, opt.latent_dim)).float().to(device))
        # z_fake = Variable(M.sample_memory(len(imgs)).to(device))
        # sample, alpha_coef = M.sample_memory(len(imgs))
        # z_fake = Variable(sample.to(device))

        # Generate a batch of images
        imgs_fake = generator(z_fake)

        # Encode real images
        # z_real = reparametrization(encoder(imgs_real))

        # Measure discriminator's ability to classify real from generated samples
        # Now labels will be correct
        output_real = discriminator(imgs_real)
        output_fake = discriminator(imgs_fake.detach())
        loss_real = adversarial_loss(output_real, label_real)
        loss_fake = adversarial_loss(output_fake, label_fake)
        loss_d = loss_real + loss_fake

        loss_d.backward()
        optimizer_D.step()

        # -----------------
        #  MEMGAN Losses
        # -----------------


        # -----------------
        #  Train Encoder
        # -----------------

        optimizer_G.zero_grad()

        # Loss measures encoder's ability to fool the discriminator
        # Real images loss: should be classified as fake
        output_real = discriminator(imgs_real)
        loss_real = adversarial_loss(output_real, label_fake)

        # Loss measures generator's ability to fool the discriminator
        # Generated images loss: should be classified as real
        output_fake = discriminator(imgs_fake)
        loss_fake = adversarial_loss(output_fake, label_real)

        loss_g = loss_fake + loss_real

        loss_g.backward()
        optimizer_G.step()

        # -----------------
        #  Process results
        # -----------------

        kbar.update(i + 1, values=[("D Loss", loss_d.item()), ("G loss", loss_g.item()), ("D(x)", output_real.mean().item()), ("D(G(x))", output_fake.mean().item())])

        if i == len(dataloader_train) - 1:
            with torch.no_grad():
                gen_examples = generator(vis_noise).detach().cpu()
                save_image(gen_examples, 
                    os.path.join(IMAGE_DIR, f'{epoch+1}.png'),
                    nrow=vis_rows, normalize=True)

    for metric,val_packed in kbar._values.items():
        value_sum, count = val_packed
        writer.add_scalar(metric, value_sum / count, epoch)
    
    writer.flush()

writer.close()

# %%
