import os
import numpy as np
import torch

from models.models import DecompModel
from tools.args import BaseOptions
from tools.visualizer import Visualizer
from tools.dataloader import get_loader
from tools.helper_functions import TrainTools

opt = BaseOptions().parse()
visualizer = Visualizer(opt)
tt = TrainTools(opt)
train_data = get_loader(opt, 'train')
dataset_size = len(train_data)
print(f'# training pairs = {dataset_size}')

ckpt_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
my_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DecompModel().to(my_device)
start_epoch, epoch_iter = tt.manage_ckpts(model)
model.train()

total_steps = (start_epoch-1) * dataset_size + epoch_iter
display_delta = total_steps % opt.display_freq  # how often to show results
save_delta = total_steps % opt.save_latest_freq  # how often to save the model

for epoch in range(start_epoch, opt.num_epochs+1):

    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size

    for data in train_data:
        # ----- forward -----
        out, loss_dict = model(*tt.process_inputs(data))
        loss_D = (loss_dict.get('D_fake',0) + loss_dict.get('D_real',0)) * 0.5
        loss_G = sum(v if k[0] == 'G' else 0 for k, v in loss_dict.items())

        # ----- backward -----
        model.optimizerG.zero_grad()
        loss_G.backward()
        model.optimizerG.step()

        model.optimizerD.zero_grad()
        loss_D.backward()
        model.optimizerD.step()

        # ----- log & save -----
        total_steps += opt.batch_size
        epoch_iter += opt.batch_size

        if total_steps % opt.display_freq == display_delta:
            # print out errors
            errors = {k: v.data.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}
            errors['D_all'], errors['G_all'] = loss_D, loss_G  # add combined to see general trend
            visualizer.print_current_errors(epoch, epoch_iter, errors)
            visualizer.plot_current_errors(errors, total_steps)

            # save imgs
            visualizer.quick_save(data['input_A'][0], epoch, 'inA')
            visualizer.quick_save(out['fake_A'][0], epoch, 'outA')
            visualizer.quick_save(data['input_B'][0], epoch, 'inB')
            visualizer.quick_save(out['fake_B'][0], epoch, 'outB')

        # save latest model
        if total_steps % opt.save_latest_freq == save_delta:  # and opt.save_models:
            print(f'saving the latest model (epoch {epoch}, total_steps {total_steps}')
            model.save_models('latest')
            np.savetxt(ckpt_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

        if epoch_iter >= dataset_size:
            break

    # save model at the end of the epoch
    if epoch % opt.save_epoch_freq == 0:  # and opt.save_models:
        print(f'saving the model at the end of epoch {epoch}')
        model.save_models('latest')
        model.save_models(epoch)
        np.savetxt(ckpt_path, (epoch + 1, 0), delimiter=',', fmt='%d')
