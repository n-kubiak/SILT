# SILT: Self-supervised Lighting Transfer Using Implicit Image Decomposition (BMVC 2021) - Kubiak _et al._
Project repo for the paper [SILT: Self-supervised Lighting Transfer Using Implicit Image Decomposition (BMVC 2021)](https://arxiv.org/abs/2110.12914)

## The basics
The SILT model is written in PyTorch and during the experiments PyTorch 1.8.1 was used with CUDA 11.1 (docker image: nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu18.04). The model was trained on a single GeForce RTX 3090 GPU.

We release the train and test code as well as the pretrained models. The checkpoints of SILT models trained on both datasets discussed in the paper can be downloaded from [here](http://personal.ee.surrey.ac.uk/Personal/S.Hadfield/SILT/SILT_ckpts.zip).

## Testing
To test the SILT model, create a ```checkpoints``` directory within SILT. Then, within ```checkpoints``` create a folder with the name of your experiment, e.g. ```checkpoints/EXP_NAME```. Place the downloaded checkpoint within your experiment folder and rename it to ```latest_net_G.pth```. 

To access the checkpoint folder, you have to update the ```--checkpoints_dir``` argument in ```tools/args.py``` to your path.

Then, to test the model, use the below command:
```
python test.py --name 'EXP_NAME'
```
The SILT project defaults to the Multi-Illumination dataset but you can create and place your own datasets in ```tools/datasets/NEW_DATASET.py``` and then add them to the dataloader ```tools/dataloader.py```.

## Training
To train the model, use the below command:
```
python train.py --name 'EXP_NAME'
```
You can also add new datasets as suggested in the Testing section. Please set your checkpointing directory ```--checkpoints_dir``` in ```tools/args.py```. To train in a self-supervised manner (with misaligned data), use the ```--misaligned``` flag.  If you wish to use tensorboard logging, use the ```--tb_log``` flag and set your logging directory using the ```--tb_dir``` argument.

## Citation
If you use or write about SILT, please use the below citation:
```
@inproceedings{kubiak_2021_silt,
  title={SILT: Self-supervised Lighting Transfer Using Implicit Image Decomposition},
  author={Nikolina Kubiak and Armin Mustafa and Graeme Phillipson and Stephen Jolly and Simon Hadfield},
  booktitle={Proceedings of the British Machine Vision Conference (BMVC)},
  year={2021}
}
```
