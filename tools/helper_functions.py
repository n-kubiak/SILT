import numpy as np
import os
import torch


class TrainTools:
    def __init__(self, opt):
        self.opt = opt
        self.ckpt_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
        self.my_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def manage_ckpts(self, model):
        try:
            model.load_model(model.netD, model.optimizerD, 'D', self.opt.which)
            print('got D')
            model.load_model(model.netG, model.optimizerG, 'G', self.opt.which)
            print('got G')

            if self.opt.which == 'latest':
                start_epoch, epoch_iter = np.loadtxt(self.ckpt_path, delimiter=',', dtype=int)
            else:
                start_epoch, epoch_iter = int(self.opt.which)+1, 0
            print(f'Resuming from epoch {start_epoch} at iteration {epoch_iter}')

        except:
            start_epoch, epoch_iter = 1, 0
            print('No checkpoints found; starting from scratch.')
        return start_epoch, epoch_iter

    def process_inputs(self, imgs, mode='train'):
        ref = imgs['reference']
        ref = ref.to(self.my_device)
        if mode == 'train':
            imgA, imgB = imgs['input_A'], imgs['input_B']
            imgA, imgB = imgA.to(self.my_device), imgB.to(self.my_device)
            return imgA, imgB, ref
        else:
            img = imgs['input']
            img = img.to(self.my_device)
            return img, ref


