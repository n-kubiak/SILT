import os
from tqdm import tqdm

import torch

from models.models import DecompModelInference
from tools.dataloader import get_loader
from tools.visualizer import Visualizer
from tools.args import BaseOptions
from tools.helper_functions import TrainTools

opt = BaseOptions().parse(save=False)
visualizer = Visualizer(opt)
tt = TrainTools(opt)

eval_data = get_loader(opt, 'test')
dataset_size = len(eval_data)
print(f'# evaluation images = {dataset_size}')
ckpt_path = os.path.join(opt.checkpoints_dir, opt.name)

# define D & G, losses and optim
my_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DecompModelInference().to(my_device)
model.load_model(model.netG, model.optimizerG, 'G', opt.which)
model.eval()

print('currently running: ', opt.name)
total_loss = dict.fromkeys(['SSIM', 'PSNR', 'Perc_loss'], 0)
with torch.no_grad():
    for data in tqdm(eval_data):
        out, loss_dict = model(*tt.process_inputs(data, mode='test'))
        total_loss['Perc_loss'] += loss_dict['G_perc']
        total_loss['SSIM'] += loss_dict['SSIM']
        total_loss['PSNR'] += loss_dict['PSNR']

for k, v in total_loss.items():
    total_loss[k] = v/dataset_size

# print out errors
errors = {k: round(v.data.item(),4) if isinstance(v, torch.Tensor) else round(v,4) for k, v in total_loss.items()}
visualizer.print_test_errors(errors)
print(errors)

# save images
visualizer.better_save(data['reference'], 'test', 'ref')
visualizer.better_save(data['input'], 'test', 'in')
visualizer.better_save(out['fake'], 'test', 'out')
#visualizer.better_save(out['r_map'], 'test', 'r')
#visualizer.better_save(out['s_map'], 'test', 's')
#visualizer.better_save(out['new_s_map'], 'test', 's_new')

