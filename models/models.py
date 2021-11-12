from kornia.losses import PSNRLoss, SSIM
from kornia import spatial_gradient

from torch import nn
import torch

from .networks import LiuGenerator, MultiscaleDiscriminator
from .base_model import BaseModel


class DecompModel(BaseModel):
    def __init__(self):
        super(DecompModel, self).__init__()

        ### GENERATOR
        norm_layer = nn.BatchNorm2d if self.opt.norm == 'batch' else nn.InstanceNorm2d
        self.netG = LiuGenerator(self.opt.input_nc, self.opt.output_nc, norm_layer=norm_layer,
                                     n_downsampling=self.opt.n_downsample_global, n_blocks=self.opt.n_blocks_global)
        params = list(self.netG.parameters())
        self.optimizerG = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))

        ### DISCRIMINATOR
        self.netD = MultiscaleDiscriminator(input_nc=self.opt.input_nc, num_D=self.opt.num_D, norm_layer=norm_layer)
        self.optimizerD = torch.optim.Adam(self.netD.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))

    def forward(self, imgA, imgB, ref):

        loss_dict = {}
        imagesA = self.netG.forward(imgA)
        imagesB = self.netG.forward(imgB)
        fake_imageA, fake_imageB = imagesA['output'], imagesB['output']

        # ----- CALCULATE GAN D LOSSES -----
        # for fakes
        fake_input_A, fake_input_B = fake_imageA.detach(), fake_imageB.detach()
        D_fake_A = self.netD.forward(fake_input_A)
        D_fake_B = self.netD.forward(fake_input_B)
        D_fake_loss_A = self.gan_loss(D_fake_A, False)
        D_fake_loss_B = self.gan_loss(D_fake_B, False)

        # for reals
        real_input_A = real_input_B = ref
        D_real_A = self.netD.forward(real_input_A)
        D_real_B = self.netD.forward(real_input_B)
        D_real_loss_A = self.gan_loss(D_real_A, True)
        D_real_loss_B = self.gan_loss(D_real_B, True)

        # sum up
        loss_dict['D_fake'] = D_fake_loss_A + D_fake_loss_B
        loss_dict['D_real'] = D_real_loss_A + D_real_loss_B

        # ----- CALCULATE GAN G LOSS -----
        G_fake_A = self.netD.forward(fake_imageA)
        G_fake_B = self.netD.forward(fake_imageB)
        G_fake_loss_A = self.gan_loss(G_fake_A, True)
        G_fake_loss_B = self.gan_loss(G_fake_B, True)
        loss_dict['G_fake'] = G_fake_loss_A + G_fake_loss_B

        # ----- VGG LOSS (G) -----
        if self.opt.use_vgg_loss:
            G_perc_loss_A = self.vgg_loss(fake_imageA, imgA)
            G_perc_loss_B = self.vgg_loss(fake_imageB, imgB)
            loss_dict['G_perc'] = G_perc_loss_A + G_perc_loss_B

        # ----- OUTPUT SIMILARITY (OS) LOSS (G) -----
        G_os_loss = 0
        if self.opt.use_os_loss:
            G_os_loss += self.l1_loss(fake_imageB, fake_imageA)

        if self.opt.use_gradient_loss:
            gradA = spatial_gradient(fake_imageA)
            gradB = spatial_gradient(fake_imageB)
            G_os_loss += self.l1_loss(gradB, gradA)

        loss_dict['G_os'] = G_os_loss

        # ----- DECOMP LOSSES (G) -----
        recon_loss_A = self.l1_loss(imgA, imagesA['r'] * imagesA['s'])
        recon_loss_B = self.l1_loss(imgB, imagesB['r'] * imagesB['s'])
        loss_dict['G_decomp'] = recon_loss_A + recon_loss_B
        if self.opt.use_reflectance_loss:
            loss_dict['G_refl'] = self.mse_loss(imagesA['r'], imagesB['r'])  # consistency between reflA & reflB

        image_dict = {'fake_A': fake_imageA, 'fake_B': fake_imageB}
        return image_dict, loss_dict

    def save_models(self, epoch):
        self.save_model(self.netG, self.optimizerG, 'G', epoch)
        self.save_model(self.netD, self.optimizerD, 'D', epoch)


class DecompModelInference(BaseModel):
    def __init__(self):
        super(DecompModelInference, self).__init__()

        ### GENERATOR
        norm_layer = nn.BatchNorm2d if self.opt.norm == 'batch' else nn.InstanceNorm2d
        self.netG = LiuGenerator(self.opt.input_nc, self.opt.output_nc, norm_layer=norm_layer,
                                     n_downsampling=self.opt.n_downsample_global, n_blocks=self.opt.n_blocks_global)
        self.optimizerG = torch.optim.Adam(self.netG.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))

    def forward(self, img, ref):
        loss_dict = {}
        out = self.netG.forward(img)
        fake_image = out['output']

        loss_dict['G_perc'] = self.vgg_loss(fake_image.detach(), ref)
        loss_dict['SSIM'] = 1 - 2 * SSIM(window_size=11, reduction='mean', max_val=1.0)(fake_image.detach(), ref).item()
        loss_dict['PSNR'] = PSNRLoss(max_val=1.0)(fake_image.detach(), ref).item()
        image_dict = {'fake': fake_image, 'r_map': out['r'], 's_map': out['s'], 'new_s_map': out['tgt_s']}

        return image_dict, loss_dict
