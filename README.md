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

Run following command to train model from scratch
```
python train.py --batch_size 8 --epochs 10 --save_path ./outputs/demo/
```
Start from pre-traind RPN
```
python train.py --rpn_pretrained ./pretrained/rpn.pt --save_path ./outputs/demo/
```

## Testing RPN
Run our pretrained model for testing
```
python test.py --weights ./pretrained/iaanet.pt
```
Run in fast version:
```
python test.py --weights ./pretrained/iaanet.pt --fast
```
Run `python test.py --help` for more configurations

# Fine Detection Module
