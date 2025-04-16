# MAXSyn
MAXSyn is the multi-view prohibited item X-ray security image synthetic dataset. This codebase is serve as MAXSyn generation.
# Description
The codebase provides training, inference, and training dataset construction code for Pix2Pix-XG. The Pix2Pix-XG is used to generate the X-ray prohitbit item with mask as the input condition. 

In addition, the codebase provides scripts that automate the operation of Blender to generate multi-angle 2D mapping masks.

A synthetic strategy code is also provided here which generates annotation files while synthesising X-ray security images.

![image](https://github.com/lz1054864168/MAXSyn/blob/main/code/IMG/github.png)

# Requirements
First clone this repository:
```
git clone https://github.com/lz1054864168/MAXSyn.git
cd MAXSyn/code
```

and install dependencies:

`conda env create -f Pix2Pix-XG.yml -n Pix2Pix-XG`

# Training
## Dataset
The combine_A_and_B.py in the folder Datasets can be used for training dataset construction by running the code as follows:

` python ./datasets/combine_A_and_B.py  --fold_A  .\datasets\datasetA  --fold_B .\datasets\datasetB  --fold_AB  .\datasets\combinAB  --no_multip
rocessing
`

In the B to A task, `.\datasets\datasetA` is the target image (X-ray image) path, `.\datasets\datasetB` is the condition image (mask) path.

## Weight
If you need download the weight from this link:

`https://zenodo.org/records/10065825/files/facades_BA.zip?download=1`

Put weight files into the `MAXsyn/code/models`.

## Run
For visualize intermediate results during the training process, you should run:

`python -m visdom.server -pory 8091`

Train your own model

`python train.py --name TASK --model atme --batch_size 8 --direction BtoA --dataroot ./datasets/yourdate --gpu_ids 0,1 --display_port 8091`

TASK is the project name.

For `--dataroot ./datasets/yourdate`, folder `yourdata` should contain the train folder and the val folder.

For `--display_port 8091`, `8091` is visualize host.


# Evaluation
















