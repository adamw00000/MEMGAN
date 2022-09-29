# %%
import argparse
import os
import pkbar
from datetime import datetime

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch

from tensorboardX import SummaryWriter

opt = argparse.Namespace(
    n_epochs=400,
    batch_size=128,
    lr=1e-4,
    b1=0.5,
    b2=0.999,
    n_cpu=8,
    latent_dim=64,
    img_size=32,
    channels=3,
    # sample_interval=400,
    dataset='CIFAR',
    # dataset='MNIST',
)

IMAGE_DIR = f"images_ALICEL2_{opt.dataset}"

os.makedirs(IMAGE_DIR, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

current_time = datetime.now().strftime('%b%d_%H-%M-%S')
logdir = os.path.join('runs', opt.dataset, current_time)
writer = SummaryWriter(logdir)

if opt.dataset == 'MNIST':
    opt.channels = 1


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def reparametrization(mu, log_sigma):
    sigma = torch.exp(log_sigma)
    return mu + torch.randn_like(mu) * sigma    


def get_log_odds(raw_marginals):
    marginals = torch.clamp(raw_marginals.mean(dim=0), 1e-7, 1 - 1e-7)
    return torch.log(marginals / (1 - marginals))


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        def encoder_block(in_filters, out_filters, kernel_size=5, stride=1, padding=0,
                relu_slope=0.1, bn_eps=1e-5):
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


class MaxOut2D(nn.Module):
    """
    Pytorch implementation of MaxOut on channels for an input that is C x H x W.
    Reshape input from N x C x H x W --> N x H*W x C --> perform MaxPool1D on dim 2, i.e. channels --> reshape back to
    N x C//maxout_kernel x H x W.
    """
    def __init__(self, max_out):
        super(MaxOut2D, self).__init__()
        self.max_out = max_out
        self.max_pool = nn.MaxPool1d(max_out)

    def forward(self, x):
        batch_size = x.shape[0]
        channels = x.shape[1]
        height = x.shape[2]
        width = x.shape[3]
        # Reshape input from N x C x H x W --> N x H*W x C
        x_reshape = torch.permute(x, (0, 2, 3, 1)).view(batch_size, height * width, channels)
        # Pool along channel dims
        x_pooled = self.max_pool(x_reshape)
        # Reshape back to N x C//maxout_kernel x H x W.
        return torch.permute(x_pooled, (0, 2, 1)).view(batch_size, channels // self.max_out, height, width).contiguous()


class DiscriminatorXZ(nn.Module):
    def __init__(self):
        super(DiscriminatorXZ, self).__init__()

        def discriminator_block(in_filters, out_filters, kernel_size=3, stride=2, padding=0, bias=False,
                bn=True, dropout=0.2, relu_slope=0.1, bn_eps=1e-5):
            block = [
                nn.Conv2d(in_filters, out_filters, kernel_size, stride, padding, bias=bias), 
                # nn.LeakyReLU(relu_slope, inplace=True), 
                nn.Dropout2d(dropout)
                # MaxOut2D(2)
            ]
            # if bn:
            #     block.insert(1, nn.BatchNorm2d(out_filters, bn_eps))
            return block

        self.x_block = nn.Sequential(
            *discriminator_block(opt.channels, 32, kernel_size=5, stride=1, dropout=0.2, bn=False, bias=True),
            MaxOut2D(2),
            *discriminator_block(16, 64, kernel_size=4, stride=2, dropout=0.5),
            MaxOut2D(2),
            *discriminator_block(32, 128, kernel_size=4, stride=1, dropout=0.5),
            MaxOut2D(2),
            *discriminator_block(64, 256, kernel_size=4, stride=2, dropout=0.5),
            MaxOut2D(2),
            *discriminator_block(128, 512, kernel_size=4, stride=1, dropout=0.5),
            MaxOut2D(2),
        )
        self.z_block = nn.Sequential(
            *discriminator_block(opt.latent_dim, 512, kernel_size=1, stride=1, dropout=0.2, bn=False),
            MaxOut2D(2),
            *discriminator_block(256, 512, kernel_size=1, stride=1, dropout=0.5, bn=False),
            MaxOut2D(2),
        )
        self.joint_block = nn.Sequential(
            *discriminator_block(512, 1024, kernel_size=1, stride=1, dropout=0.5, bn=False, bias=True),
            MaxOut2D(2),
            *discriminator_block(512, 1024, kernel_size=1, stride=1, dropout=0.5, bn=False, bias=True),
            MaxOut2D(2),
        )
        self.adv_layer = nn.Sequential(
            # nn.Linear(1024, 1),
            # nn.Conv2d(512, 1, 1, stride=1, bias=True), 
            *discriminator_block(512, 1, kernel_size=1, stride=1, dropout=0.5, bn=False, bias=True),
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
if opt.dataset == 'MNIST':
    os.makedirs("../../data/mnist", exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
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
elif opt.dataset == 'CIFAR':
    os.makedirs("./data/cifar", exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
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
        ),
        batch_size=opt.batch_size,
        shuffle=True,
    )
else:
    raise NotImplementedError()

# Optimizers
optimizer_GE = torch.optim.Adam([*generator.parameters(), *encoder.parameters()], lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# For visualization
vis_rows = 8
vis_noise = torch.normal(0, 1, (vis_rows ** 2, opt.latent_dim)).float().to(device)

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    kbar = pkbar.Kbar(target=len(dataloader), epoch=epoch, num_epochs=opt.n_epochs, width=8, always_stateful=False)
    for i, (imgs, _) in enumerate(dataloader):

        # Adversarial ground truths
        # label_real = Variable(torch.ones((imgs.shape[0], 1)).to(device), requires_grad=False)
        # label_fake = Variable(torch.zeros((imgs.shape[0], 1)).to(device), requires_grad=False)
        # Label smoothing
        label_real = Variable(torch.normal(1, 0.1, (imgs.shape[0], 1)).to(device), requires_grad=False)
        label_fake = Variable(torch.normal(0, 0.1, (imgs.shape[0], 1)).to(device), requires_grad=False)

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
        z_fake = Variable(torch.normal(0, 1, (imgs.shape[0], opt.latent_dim)).float().to(device))
        # Generate a batch of images
        imgs_fake = generator(z_fake)

        # Encode real images
        encoder_out = encoder(imgs_real)
        z_mu, z_log_sigma = encoder_out[:, :opt.latent_dim], encoder_out[:, opt.latent_dim:]
        z_real = reparametrization(z_mu, z_log_sigma)

        # Measure discriminator's ability to classify real from generated samples
        # Now labels will be correct
        output_real = discriminator(imgs_real + noise_real, z_real.detach())
        output_fake = discriminator(imgs_fake.detach() + noise_fake, z_fake)
        loss_real = adversarial_loss(output_real, label_real)
        loss_fake = adversarial_loss(output_fake, label_fake)
        loss_d = loss_real + loss_fake

        loss_d.backward()
        optimizer_D.step()
        
        # -----------------
        #  Train Generator
        # -----------------

        optimizer_GE.zero_grad()

        # Cycle consistency
        imgs_recon = generator(z_real)

        # Loss measures generator's ability to fool the discriminator
        # Real images loss: should be classified as fake
        output_real = discriminator(imgs_real + noise_real, z_real)
        loss_real = adversarial_loss(output_real, label_fake)
        # Generated images loss: vice versa
        output_fake = discriminator(imgs_fake + noise_fake, z_fake)
        loss_fake = adversarial_loss(output_fake, label_real)

        # Reconstruction loss
        recon_loss = torch.norm(imgs_recon - imgs_real, p=2)

        loss_g = loss_real + loss_fake
        loss_g_full = loss_g + recon_loss

        loss_g_full.backward()
        optimizer_GE.step()

        # -----------------
        #  Process results
        # -----------------

        kbar.update(i + 1, values=[("D Loss", loss_d.item()), ("G loss", loss_g.item()), ("D(x)", output_real.mean().item()), ("D(G(x))", output_fake.mean().item()), ("Reconstruction loss", recon_loss.item())])

        if i == len(dataloader) - 1:
            with torch.no_grad():
                gen_examples = generator(vis_noise).detach().cpu()
                save_image(gen_examples, 
                    os.path.join(IMAGE_DIR, f'{epoch+1}.png'),
                    nrow=vis_rows, normalize=True)

    for metric,val_packed in kbar._values.items():
        value_sum, count = val_packed
        writer.add_scalar(metric, value_sum / count, epoch)

writer.close()

torch.save(generator, os.path.join(logdir, f'model_generator.torch'))
torch.save(encoder, os.path.join(logdir, f'model_encoder.torch'))
torch.save(discriminator, os.path.join(logdir, f'model_discriminator.torch'))

# %%
