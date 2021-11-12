from PIL import Image
import os

from torchvision import transforms
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.reference = []

        self.apply_transforms = transforms.Compose([
            transforms.Resize(self.args.img_size),
            transforms.ToTensor()])

    def __len__(self):
        return len(self.reference)

    def get_pairs(self, num_range):
        numbers = list(range(0, num_range))
        unique_pairs = [[numbers[num1], numbers[num2]] for num1 in range(len(numbers)) for num2 in range(num1 + 1, len(numbers))]
        return unique_pairs


class TrainDataset(BaseDataset):
    def __init__(self, args):
        super(TrainDataset, self).__init__(args)
        self.input_A = []
        self.input_B = []
        self.train_folder = '/vol/vssp/SF_datasets/still/VIDIT/train/'
        self.ok_temp = [2500, 3500, 4500, 5500, 6500]
        self.ok_dir = ['NE', 'N', 'NW', 'W', 'SW', 'S', 'SE', 'E']
        self.num_imgs = len(self.ok_dir) * len(self.ok_temp) - 1
        self.pairings = self.get_pairs(self.num_imgs)

        valid_indices = list(range(0,300))
        for img in valid_indices:
            tmp_store = []  # only pair up per-scene
            for t in self.ok_temp:
                for d in self.ok_dir:
                    if d == 'E' and t == 4500:
                        break
                    img_path = os.path.join(self.train_folder, f'Image{img:03}_{t}_{d}.png')
                    tmp_store.append(img_path)
            ref_path = os.path.join(self.train_folder, f'Image{img:03}_4500_E.png')

            for j in range(0, len(self.pairings)):  # add all pairings to the list
                self.input_A.append(tmp_store[self.pairings[j][0]])
                self.input_B.append(tmp_store[self.pairings[j][1]])
                self.reference.append(ref_path)

    def __getitem__(self, index):

        ref_idx = index + len(self.pairings)
        if ref_idx >= len(self.reference):
            ref_idx -= len(self.reference)

        img_ref = Image.open(self.reference[ref_idx])  # was 'index'
        ref_tensor = self.apply_transforms(img_ref)[0:3]
        imgA = Image.open(self.input_A[index])
        imgA_tensor = self.apply_transforms(imgA)[0:3]
        imgB = Image.open(self.input_B[index])
        imgB_tensor = self.apply_transforms(imgB)[0:3]
        data_dict = {'input_A': imgA_tensor, 'input_B': imgB_tensor, 'reference': ref_tensor}
        return data_dict


class EvalDataset(BaseDataset):
    def __init__(self, args):
        super(EvalDataset, self).__init__(args)
        self.input = []
        self.eval_folder = '/vol/research/relighting/datasets/VIDIT/eval/'
        self.eval_gt_folder = '/vol/research/relighting/datasets/VIDIT/eval_gt/'

        for img in range(300,345):
            self.input.append(os.path.join(self.eval_folder, f'Image{img:03}.png'))
            self.reference.append(os.path.join(self.eval_gt_folder, f'Image{img:03}.png'))

    def __getitem__(self, index):
        img_ref = Image.open(self.reference[index]).convert('RGB')
        ref_tensor = self.apply_transforms(img_ref)[0:3]
        imgA = Image.open(self.input[index])
        imgA_tensor = self.apply_transforms(imgA)[0:3]
        data_dict = {'input': imgA_tensor, 'reference': ref_tensor}

        return data_dict

