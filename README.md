# MAXSyn
* [MAXSyn](#maxsyn)
* [Description](#description)
* [Pix2Pix\-XG Generative Model](#pix2pix-xg-generative-model)
  * [Requirements](#requirements)
  * [Training](#training)
    * [Dataset](#dataset)
    * [Weight](#weight)
    * [Run](#run)
  * [Evaluation](#evaluation)
* [Blender](#blender)
  * [Install](#install)
  * [Run](#run-1)
  * [Mask Reconstruction](#mask-reconstruction)
* [Synthesis](#synthesis)
  * [Run](#run-2)
* [Citation](#citation)
* [Acknowledgement](#acknowledgement)
MAXSyn is the multi-view prohibited item X-ray security image synthetic dataset. This codebase is serve as MAXSyn generation.

![image](https://github.com/lz1054864168/MAXSyn/blob/main/code/IMG/github.png)
# Description
The codebase provides training, inference, and training dataset construction code for Pix2Pix-XG. The Pix2Pix-XG is used to generate the X-ray prohitbit item with mask as the input condition. 

In addition, the codebase provides scripts that automate the operation of Blender to generate multi-angle 2D mapping masks.

A synthetic strategy code is also provided here which generates annotation files while synthesising X-ray security images.

# Pix2Pix-XG Generative Model
## Requirements
First clone this repository:
```
git clone https://github.com/lz1054864168/MAXSyn.git
cd MAXSyn/code
```

and install dependencies:

`conda env create -f Pix2Pix-XG.yml -n Pix2Pix-XG`

## Training
### Dataset
The combine_A_and_B.py in the folder Datasets can be used for training dataset construction by running the code as follows:

` python ./datasets/combine_A_and_B.py  --fold_A  .\datasets\datasetA  --fold_B .\datasets\datasetB  --fold_AB  .\datasets\combinAB  --no_multip
rocessing
`

In the B to A task, `.\datasets\datasetA` is the target image (X-ray image) path, `.\datasets\datasetB` is the condition image (mask) path.

### Weight
You need download the weight from this link:

`https://zenodo.org/records/10065825/files/facades_BA.zip?download=1`

Put weight files into the `MAXsyn/code/models`.

### Run
For visualize intermediate results during the training process, you should run:

`python -m visdom.server -pory 8091`

Train your own model

`python train.py --name TASK --model atme --batch_size 8 --direction BtoA --dataroot ./datasets/yourdata --gpu_ids 0,1 --display_port 8091`

TASK is the project name.

For `--dataroot ./datasets/yourdate`, folder `yourdata` should contain the train folder and the val folder.

For `--display_port 8091`, `8091` is visualize host.


## Evaluation
Test your model, just run
```
 python test.py --name fine3  --model atme --direction BtoA --dataroot ./datasets/yourdata
```
The results are saved in `./results/TASK`

# Blender
The Blender platform is used to generate a multi-view mapping of the 3D model.
## Install
Download the installation package from `https://www.blender.org/download/`. We use the Blender v2.79b.
## Run
windows:
```
"C:\Program Files\Blender Foundation\Blender\blender.exe" phong.blend --background --python phong.py -- .\\single_off_samples\\hammer.off .\\single_samples_MV
```
ubuntu:
```
blender phong.blend --background --python phong.py -- ./single_off_samples/hammer.off ./single_samples_MV
```
## Mask Reconstruction
Rebuild the mask based on the mapping constructed by Blender.
```
cd MAXSyn/Blender
python mask.py
```
As shown in the table, the semantic segmentation follows the RGB pixel encoding standard.

| Categories  | Pixel Encoding | 
|-------------|----------------|
| Gun         | (255, 0, 0)    | 
| Knife       | (0, 0, 255)    |
| Wrench      | (200, 100, 200)| 
| Pliers      | (0, 255, 0)    |
| Scissors    | (255, 0, 255)  | 
| Hammer      | (0, 200, 255)  | 
| Fork        | (255, 255, 0)  |
| Exploder    | (255, 200, 200)| 
| Firecracker | (100, 255, 100)| 
| Dynamite    | (255, 200, 100)| 

# Synthesis
We provide two strategies for image synthesis.

## Run
Signal image synthesis
```
cd MAXSyn/Synthesis
python Syn1.py
```
Multiple image synthesis
```
cd MAXSyn/Synthesis
python Syn2.py
```
The synthesis process generates annotation files simultaneously.

# Citation
```
@misc{zhuo_2025_15221190,
	author= {Zhuo Li and Caixia Liu},
	note = {\url{https://doi.org/10.5281/zenodo.15221190}},
	title = {Multi-view Prohibited Item X-ray Security Image Synthetic Dataset},
	month = apr,
	year  = 2025,
	publisher = {Zenodo},
	doi = {10.5281/zenodo.15221190},
}
```
# Acknowledgement
In this project, we reimplemented Pix2Pix-XG on PyTorch based on [ATME.pytorch](https://github.com/DLR-MI/atm) and [Feature Frequency Loss](https://github.com/EndlessSora/focal-frequency-loss) .

Blender Scripts based on [ModelNet_Blender_OFF2Multiview](https://github.com/zeaggler/ModelNet_Blender_OFF2Multiview).












