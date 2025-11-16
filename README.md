{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # CIFAR-10 Image Classification with PyTorch \
\
A work-in-progress deep learning project to build a Convolutional Neural Network (CNN) for classifying images from the CIFAR-10 dataset.\
\
##  Current Status\
\uc0\u9989  Completed:\
- Loaded CIFAR-10 dataset using `torchvision.datasets`\
- Applied transforms (ToTensor)\
- Visualized sample images\
- Set up train/validation splits\
\
#In Progress:\
- Designing CNN architecture\
- Training loop implementation\
- Model evaluation and hyperparameter tuning\
\
##  Project Structure\
```txt\
src/\
\uc0\u9492 \u9472 \u9472  main.py     # Data loading and visualization\
data/           # Downloaded automatically by torchvision\
README.md       # This file\
requirements.txt}