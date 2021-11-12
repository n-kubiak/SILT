import argparse
import torch
import os


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # meaningful params
        self.parser.add_argument('--single', action='store_true', help='flip to 1 to use only one folder (debug mode)')
        self.parser.add_argument('--num_imgs', type=int, default=10, help='how many imgs to use with MIT dataset')
        self.parser.add_argument('--no_norm', type=int, default=1, help='normalize images? (default: no)')
        self.parser.add_argument('--decomp_model', type=str, default='liu')

        # loss params
        self.parser.add_argument('--use_vgg_loss', type=int, default=1, help='use perceptual loss')
        self.parser.add_argument('--use_os_loss', type=int, default=1, help='use Output Similarity (OS) loss, ie make outA&B similar')
        self.parser.add_argument('--use_gradient_loss', type=int, default=1, help='spatial gradient loss, between outputs')
        self.parser.add_argument('--use_reflectance_loss', type=int, default=1, help='compare reflectance between branches')

        # training params
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2')
        self.parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        self.parser.add_argument('--niter', type=int, default=99, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--use_scheduler', action='store_true', help='use LR schedule for both optimizers')
        self.parser.add_argument('--scheduler_rate', type=float, default=0.999)

        # data-related options
        self.parser.add_argument('--dataset', type=str, default='mit', help='cityscapes or mit')
        self.parser.add_argument('--img_size', type=int, default=512)
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')

        # logging etc
        self.parser.add_argument('--name', type=str, default='default_name')
        self.parser.add_argument('--num_epochs', type=int, default=5)
        self.parser.add_argument('--from_scratch', action='store_true', help='train from scratch and not from latest')
        self.parser.add_argument('--which', type=str, default='latest', help='which epoch to load (for ckpts)')
        self.parser.add_argument('--save_models', action='store_true', help='use flag to save models during training')
        self.parser.add_argument('--checkpoints_dir', type=str, default='YOUR-PATH-TO-SILT/checkpoints/')
        self.parser.add_argument('--tb_log', action='store_true', help='if specified, use tensorboard logging')
        self.parser.add_argument('--tb_dir', type=str, default='YOUR-PATH-TO-SILT/tb_runs/')
        self.parser.add_argument('--display_freq', type=int, default=1000, help='frequency of showing training results')
        self.parser.add_argument('--save_latest_freq', type=int, default=1000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')

        # for generator
        self.parser.add_argument('--netG', type=str, default='global', help='selects model to use for netG')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--n_downsample_global', type=int, default=3, help='number of downsampling layers in netG - default: 4')
        self.parser.add_argument('--n_blocks_global', type=int, default=9, help='number of residual blocks in the global generator network')
        self.parser.add_argument('--n_blocks_local', type=int, default=3, help='number of residual blocks in the local enhancer network')
        self.parser.add_argument('--n_local_enhancers', type=int, default=1, help='number of local enhancers to use')
        self.parser.add_argument('--niter_fix_global', type=int, default=0, help='number of epochs that we only train the outmost local enhancer')

        # for discriminator
        self.parser.add_argument('--num_D', type=int, default=2, help='number of discriminators to use')
        self.parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        self.parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
        self.parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
        self.parser.add_argument('--pool_size', type=int, default=0, help='the size of image buffer that stores previously generated images')

        # other training parameters
        self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
        self.parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')

        self.initialized = True

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.save = save

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])
            
        args = vars(self.opt)

        # save to the disk
        if self.save:
            expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
            if not os.path.exists(expr_dir):
                os.mkdir(expr_dir)

            file_name = os.path.join(expr_dir, 'options.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')

        return self.opt
