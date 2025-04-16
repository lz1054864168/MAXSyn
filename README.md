# MAXSyn
MAXSyn is the multi-view prohibited item X-ray security image synthetic dataset. This codebase is serve as MAXSyn generation.
# Description
The codebase provides training, inference, and training dataset construction code for Pix2Pix-XG. The Pix2Pix-XG is used to generate the X-ray prohitbit item with mask as the input condition. 

In addition, the codebase provides scripts that automate the operation of Blender to generate multi-angle 2D mapping masks.

A synthetic strategy code is also provided here which generates annotation files while synthesising X-ray security images.
![image](https://github.com/lz1054864168/MAXSyn/blob/main/code/IMG/github.png)
# Environment Preparation
All the code packages that need to be installed are listed in the Pix2Pix-XG.yaml.

`conda env create -f Pix2Pix-XG.yml -n Pix2Pix-XG`
# Training
## Dataset
The combine_A_and_B.py in the folder Datasets can be used for training dataset construction by running the code as follows:

` python ./datasets/combine_A_and_B.py  --fold_A  .\datasets\datasetA  --fold_B .\datasets\datasetB  --fold_AB  .\datasets\combinAB  --no_multip
rocessing
`
.\datasets\datasetA is 
