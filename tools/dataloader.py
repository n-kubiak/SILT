import torch
from torch.utils.data import Dataset, Subset


def get_loader(args, mode):
    dataset = choose_dataset(args, mode)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                              drop_last=mode == 'train', shuffle=mode == 'train', pin_memory=True)
    return data_loader


def choose_dataset(args, mode):
    if args.dataset == 'vidit':
        from .datasets.vidit_dataset import TrainDataset, EvalDataset
    elif args.dataset == 'mit':
        from .datasets.mit_dataset import TrainDataset, EvalDataset
    else:
        print('Dataset not implemented.')
        raise NotImplementedError

    return TrainDataset(args) if mode == 'train' else EvalDataset(args)
