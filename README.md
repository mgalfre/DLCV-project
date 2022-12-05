## 20600 - Deep Learning for Computer Vision
# Final Group Project 
#### _Group 5_: Bagnara Arturo Mario 3188256, Galfrè Michela 3076847, Marchetti Simone 3185524, Marcianò Tancredi 3070110

This repository contains the notebooks, python files and data folders created for the Final Group Project of the Deep Learning for Computer Vision course. They are organized as follows:
1. Data:
    - `Aves`: folder containing examples of the original photos divided in our 10 selected species
    - `data`: folder containing original original colored images and labels, both divided in Train, Val, Test folders
    - `data_bw`: folder containing black and white images and respective labels, divided in Train, Val, Test folders
    - `colorized`: folder containing examples of colorized images obtained with DeOldify GAN
    - `pickles`: folder containing pickles of the final dataframes for Train, Dev and Test, obtained in the _data_processing.ipynb_ notebook
2. SATM:
    - `preprocess.py`: set of functions used to handle and preprocess data from iNaturalis dataset
    - `dataset.py`: set of methods and classes used in the datasets creation for training Faster R-CNN model
    - `inference.py`: set of methods and classes broadly used in the inference phase of the project
3. Plots:
    - `plots`: folder containing plots obtained in the inference phase for our models
4. Data preprocessing and visualization:
    - `data_processing.ipynb`: jupyter notebook of initial preprocessing steps to obtained the final dataframes we will use for our modles
    - `data_visualization.ipynb`: jupyter notebook for exploratory analysis and visualization
5. Faster R-CNN:
    - `Faster_training.ipynb`: jupyter notebook containing the training code for Faster R-CNN 
    - `Faster_inference.ipynb`: jupyter notebook containing the inference code for Faster R-CNN
6. YOLOv7:
    - `YOLO_training_inference.ipynb`: jupyter notebook containing the training and inference code for YOLOv7
    - `yolov7`: cloned official YOLOv7 repository from [Wong Kin Yiu](https://github.com/WongKinYiu/yolov7.git)

## Our Task
- Object Detection

## Dataset
The dataset was retrieved on [Kaggle](https://www.kaggle.com/c/inaturalist-challenge-at-fgvc-2017) and we decided to focus on the Aves species. We divided the dataset in the following way:
- Training: 22000 images and 25000 boxes (check!)
- Validation: 22000 images and 3000 boxes (check!)
- Test: 22000 images and 3000 boxes (check!)

Here below a sample image for each of the 10 aves species we selected.
![image](/plots/pictures_example.pdf?raw=true)