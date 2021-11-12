from PIL import Image
import os
import random

from torch import log
from torchvision import transforms
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.reference = []
        self.image_folder = '/vol/vssp/SF_datasets/still/Multi_Illumination/'

        self.num_imgs = self.args.num_imgs  # default: 10
        self.ref_img = 0
        self.chosen_set = list(range(self.num_imgs))
        self.chosen_set.remove(self.ref_img)
        self.pairings = self.get_pairs(self.num_imgs-1)

        self.apply_transforms = transforms.Compose([
            transforms.Resize(self.args.img_size),
            transforms.ToTensor()])

    def __len__(self):
        return len(self.reference)

    def get_pairs(self, num_range):
        numbers = list(range(num_range))
        unique_pairs = [[numbers[num1], numbers[num2]] for num1 in range(len(numbers)) for num2 in range(num1 + 1, len(numbers))]
        return unique_pairs


class TrainDataset(BaseDataset):
    def __init__(self, args):
        super(TrainDataset, self).__init__(args)
        self.input_A = []
        self.input_B = []
        self.train_folder = os.path.join(self.image_folder, 'train/')

        dirlist = [item for item in os.listdir(self.train_folder) if os.path.isdir(os.path.join(self.train_folder, item))]
        for rooms in sorted(dirlist):  # get room names
            dir_input = os.path.join(self.train_folder, rooms)
            tmp_store = []
            valid_shots = [f'dir_{shot}_mip2.jpg' for shot in self.chosen_set]
            for vshot in valid_shots:
                org_path = f'{dir_input}/{vshot}'
                tmp_store.append(org_path)
            ref_name = f'{dir_input}/dir_{self.ref_img}_mip2.jpg'

            for j in range(0, len(self.pairings)):  # add all pairings to the list
                self.input_A.append(tmp_store[self.pairings[j][0] - 1])
                self.input_B.append(tmp_store[self.pairings[j][1] - 1])
                self.reference.append(ref_name)

    def __getitem__(self, index):

        ref_idx = index + len(self.pairings)
        if ref_idx >= len(self.reference):
            ref_idx -= len(self.reference)

        img_ref = Image.open(self.reference[ref_idx])  # was index
        ref_tensor = self.apply_transforms(img_ref)
        imgA = Image.open(self.input_A[index])
        imgA_tensor = self.apply_transforms(imgA)
        imgB = Image.open(self.input_B[index])
        imgB_tensor = self.apply_transforms(imgB)
        data_dict = {'input_A': imgA_tensor, 'input_B': imgB_tensor, 'reference': ref_tensor}
        return data_dict


class EvalDataset(BaseDataset):
    def __init__(self, args):
        super(EvalDataset, self).__init__(args)
        self.input_A = []
        self.eval_folder = os.path.join(self.image_folder, 'test/')

        dirlist = [item for item in os.listdir(self.eval_folder) if os.path.isdir(os.path.join(self.eval_folder, item))]
        if self.args.single:
            dirlist = ['everett_kitchen6']  # ['everett_kitchen14']
        for dirs in sorted(dirlist):  # get room names
            dir_input = os.path.join(self.eval_folder, dirs)
            tmp_store = []
            valid_shots = [f'dir_{shot}_mip2.jpg' for shot in self.chosen_set]
            for vshot in valid_shots:
                org_path = f'{dir_input}/{vshot}'
                tmp_store.append(org_path)
            ref_name = f'{dir_input}/dir_{self.ref_img}_mip2.jpg'

            for j in range(self.num_imgs - 1):
                self.input_A.append(tmp_store[j])
                self.reference.append(ref_name)

    def __getitem__(self, index):

        img_ref = Image.open(self.reference[index]).convert('RGB')
        ref_tensor = self.apply_transforms(img_ref)
        imgA = Image.open(self.input_A[index])
        imgA_tensor = self.apply_transforms(imgA)
        data_dict = {'input': imgA_tensor, 'reference': ref_tensor}
        return data_dict
