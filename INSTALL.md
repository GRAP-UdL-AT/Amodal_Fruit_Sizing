## Installation

## 1.) Make sure CUDA 11.3 is installed
Tested on: Ubuntu 18.04, CUDA: 11.3<br/> 
(will probably work with CUDA 10.1 or later)</br>

**1.1) Check if CUDA has been installed properly:**
```
nvcc --version *(should the CUDA details)*<br/> <br/>
```

## 2.) Install Amodal_Fruit_Sizing in a virtual environment (using virtualenv)
Tested with: python 3.0, Pytorch 1.12 and torchvision 0.13<br/>
(will probably work with python 3.6, pytorch 1.4 and torchvision 0.5)<br/>

**2.1) Create a virtual environment (called __amodal__) using the terminal and activate it:**
```
mkdir -p ~/venv
virtualenv --python=python3.9 ~/venv/amodal
source ~/venv/amodal/bin/activate
```

**2.2) Download the code repository:**
```
git clone https://github.com/GRAP-UdL-AT/Amodal_Fruit_Sizing
cd Amodal_Fruit_Sizing <br/> <br/>
```

**2.3) Install the required software libraries (in the __amodal__ virtual environment, using the terminal):**
```
pip install numpy==1.23.4
pip install cython pyyaml==6.0
pip install -U 'git+https://github.com/cocodataset/cocoapi.gitsubdirectory=PythonAPI'
pip install jupyter==1.0.0
pip install opencv-python==4.6.0.66
pip install nbformat==5.7.0
pip install scikit-image==0.19.3 matplotlib==3.6.0 imageio==2.22.2
pip install black==22.10.0 isort==5.10.1 flake8==5.0.4 flake8-bugbear==22.9.23 flake8-comprehensions==3.10.0
pip install tifffile==2022.10.10
pip install tqdm==4.64.1 
pip install pandas==1.5.1
pip install seaborn==0.12.1 
pip install circle-fit==0.1.3
pip install fvcore==0.1.5.post20220512
pip install pycocotools==2.0
pip install sklearn==1.1.2
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -e . 
```
Always make sure to install a pytorch version matching your cuda version.<br/>

**2.4) Check if Pytorch links with CUDA (in the __amodal__ virtual environment, using the terminal):**
- python
- import torch
- torch.version.cuda *(should print 10.1)*
- torch.cuda.is_available() *(should True)*
- torch.cuda.get_device_name(0) *(should print the name of the first GPU)*
- quit() <br/> <br/>

**2.5) Check if detectron2 is found in python (in the amodal virtual environment, using the terminal):**
- python
- import detectron2 *(should not print an error)*
- from detectron2 import model_zoo *(should not print an error)*
- quit() <br/>
