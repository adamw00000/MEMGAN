# %%
import argparse
import os
import pkbar
from datetime import datetime
import numpy as np
from sklearn import metrics

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torchvision import datasets
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils

from spectral import SpectralNorm

from tensorboardX import SummaryWriter

opt = argparse.Namespace(
    n_epochs=400,
    # n_epochs=10,
    batch_size=128,
    lr=1e-4,
    b1=0.5,
    b2=0.999,
    n_cpu=8,
    latent_dim=256, # should be == dim_memory
    img_size=32,
    channels=3,
    n_memory=100,
    dim_memory=256,
    # sample_interval=400,
    training_class=3,
    # dataset='CIFAR',
    dataset='MNIST',
)

IMAGE_DIR = f"images_MEMGAN_{opt.dataset}_v7"

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(os.path.join(IMAGE_DIR, 'RANDOM_SAMPLES'), exist_ok=True)
os.makedirs(os.path.join(IMAGE_DIR, 'MEMORY_GEN_SAMPLES'), exist_ok=True)
os.makedirs(os.path.join(IMAGE_DIR, 'MEMORY_VIS'), exist_ok=True)
os.makedirs(os.path.join(IMAGE_DIR, 'TEST_RECONS'), exist_ok=True)
os.makedirs(os.path.join(IMAGE_DIR, 'TEST_HISTOGRAMS'), exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

current_time = datetime.now().strftime('%b%d_%H-%M-%S')
logdir = os.path.join('runs', opt.dataset, current_time)
writer = SummaryWriter(logdir)

if opt.dataset == 'MNIST':
    opt.img_size = 28
    opt.channels = 1
    opt.n_memory = 50
    opt.dim_memory = opt.latent_dim = 64


# def weights_init_normal(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         nn.init.normal_(m.weight.data, 0.0, 0.02)
#     elif classname.find('BatchNorm') != -1:
#         nn.init.normal_(m.weight.data, 1.0, 0.02)
#         nn.init.constant_(m.bias.data, 0)


def reparametrization(encoder_out):
    mu, log_sigma = encoder_out[:, :opt.latent_dim], encoder_out[:, opt.latent_dim:]
    sigma = torch.exp(log_sigma)
    return mu + torch.randn_like(mu) * sigma    


def get_log_odds(raw_marginals):
    marginals = torch.clamp(raw_marginals.mean(dim=0), 1e-7, 1 - 1e-7)
    return torch.log(marginals / (1 - marginals))


class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim):
        super().__init__()
        
        # Construct the conv layers
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//2 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//2 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        
        # Initialize gamma as 0
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
        
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B * C * W * H)
            returns :
                out : self attention value + input feature 
                attention: B * N * N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        
        proj_query  = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0,2,1) # B * N * C
        proj_key =  self.key_conv(x).view(m_batchsize, -1, width*height) # B * C * N
        energy =  torch.bmm(proj_query, proj_key) # batch matrix-matrix product
        
        attention = self.softmax(energy) # B * N * N
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height) # B * C * N
        out = torch.bmm(proj_value, attention.permute(0,2,1)) # batch matrix-matrix product
        out = out.view(m_batchsize,C,width,height) # B * C * W * H
        
        # Add attention weights onto input
        out = self.gamma*out + x
        # return out, attention
        return out


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def generator_block(in_filters, out_filters, kernel_size=4, stride=2, padding=1,
                bn_eps=1e-5, bias=False):
            block = [
                SpectralNorm(
                    nn.ConvTranspose2d(in_filters, out_filters, kernel_size, stride, padding, bias=bias)
                ), 
                nn.BatchNorm2d(out_filters, bn_eps),
                nn.ReLU(inplace=True), 
            ]
            return block

        ngf = 32
        
        if opt.dataset == 'MNIST':
            self.conv_blocks = nn.Sequential(
                *generator_block(opt.latent_dim, ngf * 8, 1, 1, 0, bias=False),
                *generator_block(ngf * 8, ngf * 4, 4, 1, 0, bias=False),
                *generator_block(ngf * 4, ngf * 2, 3, 2, 1, bias=False),
                *generator_block(ngf * 2, ngf * 1, 4, 2, 1, bias=False),
                Self_Attn(ngf * 1),
                nn.ConvTranspose2d(ngf, opt.channels, 4, 2, 1, bias=False),
                nn.Tanh(),
            )
        # elif opt.dataset == 'CIFAR':
        #     self.conv_blocks = nn.Sequential(
        #         *generator_block(opt.latent_dim, ngf * 16, 1, 1, 0, bias=False),
        #         *generator_block(ngf * 16, ngf * 8, 4, 1, 0, bias=False),
        #         *generator_block(ngf * 8, ngf * 4, 4, 2, 0, bias=False),
        #         *generator_block(ngf * 4, ngf * 2, 4, 1, 0, bias=False),
        #         *generator_block(ngf * 2, ngf * 1, 4, 2, 0, bias=False),
        #         nn.ConvTranspose2d(ngf, opt.channels, 5, 1, 0, bias=False),
        #         nn.Tanh(),
        #     )

    def forward(self, z):
        img = self.conv_blocks(z.view(*z.shape, 1, 1))
        return img


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        def encoder_block(in_filters, out_filters, kernel_size=4, stride=2, padding=1,
                relu_slope=0.1, bn_eps=1e-5):
            block = [
                SpectralNorm(
                    nn.Conv2d(in_filters, out_filters, kernel_size, stride, padding, bias=False)
                ), 
                # nn.BatchNorm2d(out_filters, bn_eps),
                nn.LeakyReLU(relu_slope, inplace=True), 
            ]
            return block

        nef = 32

                # *discriminator_block(opt.channels, ndf, 4, 2, 1),
                # *discriminator_block(ndf * 1, ndf * 2, 4, 2, 1),
                # *discriminator_block(ndf * 2, ndf * 4, 4, 2, 1),
                # Self_Attn(ndf * 2),
                # *discriminator_block(ndf * 4, ndf * 8, 4, 2, 1),
        if opt.dataset == 'MNIST':
            self.model = nn.Sequential(
                *encoder_block(opt.channels, nef, 4, 2, 1),
                *encoder_block(nef * 1, nef * 2, 4, 2, 1),
                *encoder_block(nef * 2, nef * 4, 3, 2, 1),
                Self_Attn(nef * 4),
                *encoder_block(nef * 4, nef * 8, 4, 1, 0),
                # 2 * latent = (mu, sigma)
                nn.Conv2d(nef * 8, 2 * opt.latent_dim, 1, 1, 0, bias=True),
                nn.Flatten()
            )
        # elif opt.dataset == 'CIFAR':
        #     self.model = nn.Sequential(
        #         *encoder_block(opt.channels, nef, 5, 1, 0),
        #         *encoder_block(nef * 1, nef * 2, 4, 2, 0),
        #         *encoder_block(nef * 2, nef * 4, 4, 1, 0),
        #         *encoder_block(nef * 4, nef * 8, 4, 2, 0),
        #         *encoder_block(nef * 8, nef * 16, 4, 1, 0),
        #         *encoder_block(nef * 16, nef * 16, 1, 1, 0),
        #         # 2 * latent = (mu, sigma)
        #         nn.Conv2d(nef * 16, 2 * opt.latent_dim, 1, 1, 0, bias=True),
        #         nn.Flatten()
        #     )

    def forward(self, x):
        out = self.model(x)
        return out


class DiscriminatorXZ(nn.Module):
    def __init__(self):
        super(DiscriminatorXZ, self).__init__()

        def discriminator_block(in_filters, out_filters, kernel_size=4, stride=2, padding=1, bias=False,
                bn=True, dropout=0.3, relu_slope=0.1, bn_eps=1e-5):
            block = [
                SpectralNorm(
                    nn.Conv2d(in_filters, out_filters, kernel_size, stride, padding, bias=bias)
                ), 
                nn.LeakyReLU(relu_slope, inplace=True), 
                # nn.Dropout2d(dropout)
            ]
            # if bn:
            #     block.insert(1, nn.BatchNorm2d(out_filters, bn_eps))
            return block

        ndf = 32

        if opt.dataset == 'MNIST':
            self.x_block = nn.Sequential(
                *discriminator_block(opt.channels, ndf, 4, 2, 1, bn=False),
                *discriminator_block(ndf * 1, ndf * 2, 4, 2, 1),
                *discriminator_block(ndf * 2, ndf * 4, 3, 2, 1),
                Self_Attn(ndf * 4),
                *discriminator_block(ndf * 4, ndf * 8, 4, 1, 0),
            )
            self.z_block = nn.Sequential(
                *discriminator_block(opt.latent_dim, ndf * 8, 1, 1, 0, dropout=0.2, bn=False),
                # *discriminator_block(ndf * 8, ndf * 8, 1, 1, 0, dropout=0.5, bn=False),
            )
            self.joint_block = nn.Sequential(
                # *discriminator_block(ndf * 16, ndf * 16, 1, 1, 0, dropout=0.5, bn=False, bias=True),
                # *discriminator_block(ndf * 16, ndf * 16, 1, 1, 0, dropout=0.5, bn=False, bias=True),
            )
            self.adv_layer = nn.Sequential(
                # nn.Linear(1024, 1),
                # nn.Conv2d(512, 1, 1, stride=1, bias=True), 
                *discriminator_block(ndf * 16, 1, 1, 1, 0, dropout=0.5, bn=False, bias=True),
                nn.Sigmoid(),
                nn.Flatten()
            )
        # elif opt.dataset == 'CIFAR':
        #     self.x_block = nn.Sequential(
        #         *discriminator_block(opt.channels, ndf, 5, 1, 0),
        #         *discriminator_block(ndf * 1, ndf * 2, 4, 2, 0),
        #         *discriminator_block(ndf * 2, ndf * 4, 4, 1, 0),
        #         *discriminator_block(ndf * 4, ndf * 8, 4, 2, 0),
        #         *discriminator_block(ndf * 8, ndf * 16, 4, 1, 0),
        #         *discriminator_block(ndf * 16, ndf * 16, 1, 1, 0),
        #     )
        #     self.z_block = nn.Sequential(
        #         *discriminator_block(opt.latent_dim, ndf * 16, 1, 1, 0, dropout=0.2, bn=False),
        #         *discriminator_block(ndf * 16, ndf * 16, 1, 1, 0, dropout=0.5, bn=False),
        #     )
        #     self.joint_block = nn.Sequential(
        #         # *discriminator_block(ndf * 32, ndf * 32, 1, 1, 0, dropout=0.5, bn=False, bias=True),
        #         # *discriminator_block(ndf * 32, ndf * 32, 1, 1, 0, dropout=0.5, bn=False, bias=True),
        #     )
        #     self.adv_layer = nn.Sequential(
        #         # nn.Linear(1024, 1),
        #         # nn.Conv2d(512, 1, 1, stride=1, bias=True), 
        #         *discriminator_block(ndf * 32, 1, 1, 1, 0, dropout=0.5, bn=False, bias=True),
        #         nn.Sigmoid(),
        #         nn.Flatten()
        #     )

    def forward(self, x, z):
        x_repr = self.x_block(x)
        z_repr = self.z_block(z.view(*z.shape, 1, 1))
        joint_repr = torch.cat([x_repr, z_repr], dim=1)
        out = self.joint_block(joint_repr)
        # out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


class MemoryModule(nn.Module):
    def __init__(self):
        super(MemoryModule, self).__init__()
        self.n_memory = opt.n_memory
        self.dim_memory = opt.dim_memory
        self.M = nn.Parameter(torch.randn(self.n_memory, self.dim_memory))

    # Perform projection onto memory
    def P(self, z):
        alpha = torch.softmax(z @ self.M.T, dim=1)
        return alpha @ self.M

    # Get projection coefficients
    def get_coefs(self, z_prime, apply_softmax=True):
        coef = z_prime @ self.M.T
        if not apply_softmax:
            return coef
        
        return torch.softmax(coef, axis=1)

    # Get convex combination of memory units
    def sample_memory(self, n):
        coef = torch.rand((n, self.n_memory)).to(self.M.device)
        coef = torch.softmax(coef, dim=1)
        return coef @ self.M, coef


# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
encoder = Encoder()
discriminator = DiscriminatorXZ()
M = MemoryModule()

generator.to(device)
encoder.to(device)
discriminator.to(device)
adversarial_loss.to(device)
M.to(device)

# Initialize weights
# generator.apply(weights_init_normal)
# encoder.apply(weights_init_normal)
# discriminator.apply(weights_init_normal)

# Configure data loader
if opt.dataset == 'MNIST':
    os.makedirs("../../data/mnist", exist_ok=True)
    dataset_train = datasets.MNIST(
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
    )
    dataset_test = datasets.MNIST(
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
    )

    # https://discuss.pytorch.org/t/how-to-use-one-class-of-number-in-mnist/26276/17
    is_inlier = dataset_train.targets == opt.training_class
    dataset_train = torch.utils.data.dataset.Subset(dataset_train, np.where(is_inlier)[0])
    # https://discuss.pytorch.org/t/change-labels-in-data-loader/36823/9
    dataset_test.targets = list(np.where(dataset_test.targets == opt.training_class, 1, 0))

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
    is_inlier = torch.tensor(dataset_train.targets) == opt.training_class
    dataset_train = torch.utils.data.dataset.Subset(dataset_train, np.where(is_inlier)[0])
    # https://discuss.pytorch.org/t/change-labels-in-data-loader/36823/9
    dataset_test.targets = list(np.where(torch.tensor(dataset_test.targets) == opt.training_class, 1, 0))

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
optimizer_E = torch.optim.Adam(encoder.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_M = torch.optim.Adam(M.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# scheduler_G = StepLR(optimizer_G, step_size=50, gamma=0.5)
# scheduler_E = StepLR(optimizer_E, step_size=50, gamma=0.5)
# scheduler_D = StepLR(optimizer_D, step_size=50, gamma=0.5)
# scheduler_M = StepLR(optimizer_M, step_size=50, gamma=0.5)

scheduler_G = StepLR(optimizer_G, step_size=50, gamma=1)
scheduler_E = StepLR(optimizer_E, step_size=50, gamma=1)
scheduler_D = StepLR(optimizer_D, step_size=50, gamma=1)
scheduler_M = StepLR(optimizer_M, step_size=50, gamma=1)

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
        # z_fake = Variable(torch.normal(0, 1, (batch_size, opt.latent_dim)).float().to(device))
        # z_fake = Variable(M.sample_memory(len(imgs)).to(device))
        sample, alpha_coef = M.sample_memory(len(imgs))
        z_fake = Variable(sample.to(device))

        # Generate a batch of images
        imgs_fake = generator(z_fake)

        # Encode real images
        z_real = reparametrization(encoder(imgs_real))

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
        #  MEMGAN Losses
        # -----------------


        # -----------------
        #  Train Encoder
        # -----------------

        optimizer_E.zero_grad()

        # Loss measures encoder's ability to fool the discriminator
        # Real images loss: should be classified as fake
        output_real = discriminator(imgs_real + noise_real, z_real)
        loss_real = adversarial_loss(output_real, label_fake)

        # Compute MEMGAN losses
        projection = M.P(z_real)

        # Cycle consistency loss
        imgs_recon = generator(projection)
        cc_loss = torch.mean(
            torch.linalg.vector_norm((imgs_recon - imgs_real), ord=2, dim=(1, 2, 3))
        )
        
        # Memory projection loss
        mp_loss = torch.mean(
            torch.linalg.vector_norm((projection - z_real), ord=2, dim=1)
        )

        # Mutual information loss
        z_fake_recon = reparametrization(encoder(imgs_fake.detach()))

        alpha_coef_recon = M.get_coefs(z_fake_recon, apply_softmax=False)
        # mi_loss = torch.mean(
        #     -torch.sum(alpha_coef * torch.log((torch.softmax(alpha_coef_recon, dim=1)), dim=1)
        # )
        mi_loss = F.cross_entropy(alpha_coef_recon, alpha_coef)

        loss_e = loss_real + cc_loss + mp_loss + mi_loss

        loss_e.backward()
        optimizer_E.step()

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Loss measures generator's ability to fool the discriminator
        # Generated images loss: should be classified as real
        output_fake = discriminator(imgs_fake + noise_fake, z_fake)
        loss_fake = adversarial_loss(output_fake, label_real)

        # Recompute MEMGAN losses

        # Cycle consistency loss
        imgs_recon = generator(projection.detach())
        cc_loss = torch.mean(
            torch.linalg.vector_norm((imgs_recon - imgs_real), ord=2, dim=(1, 2, 3))
        )

        # Mutual information loss
        z_fake_recon = reparametrization(encoder(imgs_fake))
        alpha_coef_recon = M.get_coefs(z_fake_recon, apply_softmax=False)
        # mi_loss = torch.mean(
        #     -torch.sum(alpha_coef * torch.log((torch.softmax(alpha_coef_recon, dim=1)), dim=1)
        # )
        mi_loss = F.cross_entropy(alpha_coef_recon, alpha_coef)

        loss_g = loss_fake + cc_loss + mi_loss

        loss_g.backward()
        optimizer_G.step()

        # -----------------
        #  Update Memory
        # -----------------

        # Recompute losses
        projection = M.P(z_real.detach())

        # Cycle consistency loss
        imgs_recon = generator(projection)
        cc_loss = torch.mean(
            torch.linalg.vector_norm((imgs_recon - imgs_real), ord=2, dim=(1, 2, 3))
        )
        
        # Memory projection loss
        mp_loss = torch.mean(
            torch.linalg.vector_norm((projection - z_real.detach()), ord=2, dim=1)
        )

        # Mutual information loss
        z_fake_recon = reparametrization(encoder(imgs_fake.detach()))
        alpha_coef_recon = M.get_coefs(z_fake_recon, apply_softmax=False)
        # mi_loss = torch.mean(
        #     -torch.sum(alpha_coef * torch.log((torch.softmax(alpha_coef_recon, dim=1)), dim=1)
        # )
        mi_loss = F.cross_entropy(alpha_coef_recon, alpha_coef)

        optimizer_M.zero_grad()

        # Loss measures memory quality
        loss_m = cc_loss + mp_loss + mi_loss

        loss_m.backward()
        optimizer_M.step()

        # -----------------
        #  Process results
        # -----------------

        kbar.update(i, values=[("D Loss", loss_d.item()), ("G loss", loss_g.item()), ("D(x)", output_real.mean().item()), ("D(G(x))", output_fake.mean().item()), ("M loss", loss_m.item()), ("E loss", loss_e.item())])

        if i == len(dataloader_train) - 1:
            with torch.no_grad():
                gen_examples = generator(vis_noise).detach().cpu()
                save_image(gen_examples, 
                    os.path.join(IMAGE_DIR, 'RANDOM_SAMPLES', f'{epoch+1}.png'),
                    nrow=vis_rows, normalize=True)
                
                z_mem_sample_vis, _ = M.sample_memory(vis_rows ** 2)
                gen_memory_examples = generator(z_mem_sample_vis).detach().cpu()
                save_image(gen_memory_examples, 
                    os.path.join(IMAGE_DIR, 'MEMORY_GEN_SAMPLES', f'{epoch+1}.png'),
                    nrow=vis_rows, normalize=True)
                
                z_memory_vis = M.M.data
                memory_vis = generator(z_memory_vis).detach().cpu()
                save_image(memory_vis, 
                    os.path.join(IMAGE_DIR, 'MEMORY_VIS', f'{epoch+1}.png'),
                    nrow=int(np.sqrt(opt.n_memory)), normalize=True)

    scheduler_G.step()
    scheduler_E.step()
    scheduler_D.step()
    scheduler_M.step()

    generator.eval()
    encoder.eval()
    discriminator.eval()
    M.eval()

    with torch.no_grad():
        pred_scores = []
        true_labels = []

        for i, (imgs, labels) in enumerate(dataloader_test):
            batch_size = len(imgs)

            # Calculate anomaly scores
            imgs = imgs.float().to(device)
            imgs_recon = generator(M.P(reparametrization(encoder(imgs))))
            scores = torch.norm((imgs - imgs_recon).reshape(batch_size, -1), p=2, dim=1)
            pred_scores.append(scores.detach())
            true_labels.append(labels)

            if i == 0:
                imgs_with_recons = torch.zeros(2 * vis_rows**2, imgs.shape[1], imgs.shape[2], imgs.shape[3])
                imgs_with_recons[::2] = imgs[:vis_rows**2]
                imgs_with_recons[1::2] = imgs_recon[:vis_rows**2]
                save_image(imgs_with_recons, 
                    os.path.join(IMAGE_DIR, 'TEST_RECONS', f'{epoch+1}.png'),
                    nrow=2 * vis_rows, normalize=True)

        pred_scores = torch.cat(pred_scores).cpu().numpy()
        true_labels = torch.cat(true_labels).cpu().numpy()

        # # THRESHOLDING IDEAS

        # threshold = np.quantile(-pred_scores, 1 - np.mean(true_labels))
        # pred_labels = np.where(-pred_scores < threshold, 0, 1)

        # pre = metrics.precision_score(true_labels, pred_labels)
        # rec = metrics.recall_score(true_labels, pred_labels)
        # f1 = metrics.f1_score(true_labels, pred_labels)

        # # from sklearn.cluster import SpectralClustering
        # # cluster = SpectralClustering(2)
        # # cluster_assignment = cluster.fit_predict(pred_scores.reshape(-1, 1))
        # # cluster_assignment
        # from sklearn.cluster import KMeans
        # cluster = KMeans(2, init=np.r_[cc_loss.detach().cpu().numpy(), np.mean(pred_scores)].reshape(-1, 1))
        # cluster_assignment = cluster.fit_predict(pred_scores.reshape(-1, 1))
        # cluster_assignment

        # cluster_0_mean = np.mean(pred_scores[cluster_assignment == 0])
        # cluster_1_mean = np.mean(pred_scores[cluster_assignment == 1])
        
        # outlier_cluster = 0
        # if cluster_1_mean > cluster_0_mean:
        #     outlier_cluster = 1
        
        # pred_labels = np.where(cluster_assignment == outlier_cluster, 0, 1)
        # pre = metrics.precision_score(true_labels, pred_labels)
        # rec = metrics.recall_score(true_labels, pred_labels)
        # f1 = metrics.f1_score(true_labels, pred_labels)

        # # THRESHOLDING IDEAS END

        auc = metrics.roc_auc_score(true_labels, -pred_scores)

        inliers = np.where(true_labels == 1)[0]
        inliers_mask = np.zeros_like(pred_scores, dtype=np.bool)
        inliers_mask[inliers] = True

        mean_inlier_score = np.mean(pred_scores[inliers_mask])
        mean_outlier_score = np.mean(pred_scores[~inliers_mask])

        kbar.add(1, values=[("AUC", auc), ("A(IN)", mean_inlier_score), ("A(OUT)", mean_outlier_score)])
        
        # PLOTTING

        import pandas as pd
        df = pd.DataFrame({
            'Score': pred_scores,
            'Placeholder': np.array('' * len(pred_scores)),
            'Class': np.where(true_labels == 0, 'Outlier', 'Inlier'),
        })

        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(8, 10))
        sns.histplot(df, x='Score', hue='Class', hue_order=['Outlier', 'Inlier'])
        plt.savefig(os.path.join(IMAGE_DIR, 'TEST_HISTOGRAMS', f'{epoch+1}.png'))
        plt.close()

    generator.train()
    encoder.train()
    discriminator.train()
    M.train()

    for metric,val_packed in kbar._values.items():
        value_sum, count = val_packed
        writer.add_scalar(metric, value_sum / count, epoch)
    
    writer.flush()

writer.close()

torch.save(generator, os.path.join(logdir, f'model_generator.torch'))
torch.save(encoder, os.path.join(logdir, f'model_encoder.torch'))
torch.save(discriminator, os.path.join(logdir, f'model_discriminator.torch'))
torch.save(M, os.path.join(logdir, f'model_memory.torch'))

# %%
