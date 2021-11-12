import os
import numpy as np
import ntpath
import time
from PIL import Image
from torchvision.utils import save_image, make_grid
from torch import unsqueeze, cat, uint8, sqrt


class Visualizer():
    def __init__(self, opt):
        self.opt = opt
        self.tb_log = opt.tb_log
        self.name = opt.name
        self.save_path = os.path.join(opt.checkpoints_dir, opt.name)
        if self.tb_log:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.opt.tb_dir + self.name + '/')
            print('Tensorboard logging enabled.\n')

        self.log_name = os.path.join(self.opt.checkpoints_dir, self.name, 'losses.txt')
        self.test_name = os.path.join(self.opt.checkpoints_dir, 'test.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write(f'================ Training Loss {now} ================\n')

    # errors: dictionary of error labels and values
    def plot_current_errors(self, errors, step):
        if self.tb_log:
            for tag, value in errors.items():
                # summary = self.tf.Summary(value=[self.tf.Summary.Value(tag=tag, simple_value=value)])
                # self.writer.add_summary(summary, step)
                self.writer.add_scalar(tag, value, step)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors):
        message = f'(epoch: {epoch}, iters: {i}) '
        for k, v in errors.items():
            if v != 0:
                message += f'{k}: {v:.3f} '

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def print_test_errors(self, errors):
        message = f'(test scores for {self.name}) '
        for k, v in errors.items():
            if v != 0:
                message += f'{k}: {v:.3f} '

        # print(message)
        with open(self.test_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def quick_save(self, data, epoch, suffix, norm=True):
        return save_image(data, self.save_path + f'/Ep_{epoch}_{suffix}.jpg', normalize=norm)

    def better_save(self, data, epoch, suffix):
        output_img = self.to_img(data.cpu())
        grid = make_grid(output_img)
        ndarr = grid.permute(1, 2, 0).to('cpu', uint8).numpy()
        im = Image.fromarray(ndarr)
        file_path = self.save_path + f'/Ep_{epoch}_{suffix}.jpg'
        im.save(file_path)

    def to_img(self, x):
        std_dev = [0.5, 0.5, 0.5]
        mean = std_dev
        data_len = len(x.shape)
        # breakpoint()
        if data_len == 3:  # single image
            h, w = x.size(1), x.size(2)
            if not self.opt.no_norm:
                x_ch0 = unsqueeze(x[0], 0) * std_dev[0] + mean[0]
                x_ch1 = unsqueeze(x[1], 0) * std_dev[1] + mean[1]
                x_ch2 = unsqueeze(x[2], 0) * std_dev[2] + mean[2]
                x = cat((x_ch0, x_ch1, x_ch2), 1)
            x = x.mul(255).clamp(0, 255)
            x = x.view(3, h, w)
        else:  # batch of data
            h, w = x.size(2), x.size(3)
            if not self.opt.no_norm:
                x_ch0 = unsqueeze(x[:, 0], 1) * std_dev[0] + mean[0]
                x_ch1 = unsqueeze(x[:, 1], 1) * std_dev[1] + mean[1]
                x_ch2 = unsqueeze(x[:, 2], 1) * std_dev[2] + mean[2]
                x = cat((x_ch0, x_ch1, x_ch2), 1)
            x = x.mul(255).clamp(0, 255)
            x = x.view(x.size(0), 3, h, w)
        return x



