import torch
import os

from .losses import VGGLoss, GANLoss
from tools.args import BaseOptions
opt = BaseOptions().parse(save=False)


class BaseModel(torch.nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.opt = opt
        # init common losses
        self.gan_loss = GANLoss()
        self.l1_loss = torch.nn.L1Loss()
        self.mse_loss = torch.nn.MSELoss()
        if self.opt.use_vgg_loss:
            self.vgg_loss = VGGLoss(detach=True)

    def save_model(self, model, optim, name, epoch):
        save_filename = f'{epoch}_net_{name}.pth'
        save_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, save_filename)
        save_dict = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optim.state_dict()}
        torch.save(save_dict, save_path)

    def load_model(self, model, optim, name, epoch):
        load_filename = '%s_net_%s.pth' % (epoch, name)
        load_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, load_filename)
        checkpoint = torch.load(load_path)

        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])

