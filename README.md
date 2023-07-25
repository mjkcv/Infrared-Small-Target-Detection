# Infrared-Small-Target-Detection
Official implementation for article "Infrared Dim and Small Target Detection Based on Background Prediction".
The framework contains a coarse detection module and a fine detection module.

And if the implementation of this repo is helpful to you, just star it, thanks!

# Requirement
**Packages:**
* Python 3.6
* Pytorch 1.10
* opencv-python
* numpy
* tqdm
* pandas
* yaml


# Coarse Detection Module
The coarse detection module utilizes the Region Proposal Network (RPN) to detect the 
location of the candidate targets. The original image on these locations is segmented by simple threshold to obtain the 
candidate targets, which are labeled with the $mask$ so that It is convenient to detect it in the fine detection stage.


## Training RPN
The training process are conducted using [PyTorch](https://github.com/pytorch/pytorch) with a single GeForce RTX 3090 GPU of 24 GB Memory.

Run following command to train model using a data set (MFIRST) [**[Dataset]**](https://github.com/wanghuanphd/MDvsFA_cGAN)
[**[Paper]**](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Miss_Detection_vs._False_Alarm_Adversarial_Learning_for_Small_Object_ICCV_2019_paper.pdf).
The MFIRST data set includes 9,956 training images and 100 test images. These images come from realistic infrared sequences and synthetic infrared images.
The size of the original image is set to $256\times256$. To ensure that the Region Proposal Network acquires target candidate areas, the size of the anchor box is fixed as $10\times10$.
RPN is trained separately by the similar loss function of [**[Paper]**](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9745054).
```
python train.py --batch_size 16 --epochs 100 --save_path ./outputs/demo/
```

## Generating $mask$
Run our pretrained model for generating $mask$ masks the candidate targets.
```
python mask_generate.py --weights ./outputs/demo/last.pt --save_path mask/
```

# Fine Detection Module
In Fine Detection Module, a inpainting with a mask-aware dynamic filtering module (MADF) is adapted to predict the background at candidate tagrets.

## Inpainting
### Training Inpainting for Infrared Images
 A dataset of pure infrared cloud images is proposed for the training phase of image inpainting. This dataset is composed of various cloud images. 
 Images of the dataset are acquired by cropping and segmenting original images in MATLAB 2019b. These images were made into a dataset containing 
 43,500 infrared background images including cirrus,  stratus, cirrocumulus, etc. The part of the dataset is displayed as follows.
 ![image](figures/dataset-part.png)
 ```
 python train.py --train_root train_root --mask_root mask_root --test_root test_root --use_incremental_supervision
 ```
 Train datasets are these cloud images. Mask datasets can be download [here](https://nv-adlr.github.io/publication/partialconv-inpainting). We train all the datasets with the same mask datasets. Notice that mask images must be 0-255 images.
In the training process of inapinting, the model for Places2 (You can download this dataset [here](http://places2.csail.mit.edu/download.html).)
is employed as the starting point of our model. The detailed parameters of the model are similar to the training process in [**[Paper]**](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9423556). 
The learning rate is set to 0.0002. We train our model for about 300K iterations. Our model was trained on a GeForce RTX 2080 GPU (8G) with a batch size of 8.
The pretrained model is are provided in [here](Link：https://pan.baidu.com/s/1iGr35YXGX9E0Af1OhG3KgQ code：edim).
### Testing Inpainting
```
python test.py --list_file name_list --snapshot output/snapshots/default/ckpt/1000000.pth
```

## Target Detection

The infrared image and the repaired image are fed in target_detection_not_weighting.m. The tagret is obtained bu the difference between the infrared image and the repaired image.

