import pandas as pd
import json
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
#from bs4 import BeautifulSoup
import numpy as np
from PIL import Image
import torchvision
import torch
from torchvision import transforms, datasets, models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import time
from tqdm.notebook import tqdm
import random
import pickle
import shutil



def json2df(path = 'train_2017_bboxes.json'):
    """
    this function takes as input the path of a json file
    it returns a pandas dataframe with the features we are interesed in 
    """
    # Python program to read
    # json file
    # Opening JSON file
    f = open(path)
    # returns JSON object as 
    # a dictionary
    df = json.load(f)

    # Iterating through the json
    # list
    #for i in data['emp_details']:
        #print(i)
    f.close()
    
    annotations = pd.DataFrame(df["annotations"])
    images = pd.DataFrame(df["images"])
    merged = pd.merge(images, annotations, left_on="id", right_on= "image_id").rename({"id_x":"id"}, axis = 1)
    
    merged["identifier"] = merged.apply(lambda x: x.file_name.split("/")[-1], axis = 1)
    merged["category_name"] = merged.file_name.apply(lambda x: x.split("/")[2])
    merged["super_category_name"] = merged.file_name.apply(lambda x: x.split("/")[1])
    
    
    return merged.sample(frac = 1, random_state = 810)


def merge_aves_df(df1 = "train_2017_bboxes.json", df2 = "val_2017_bboxes.json"):
    return pd.concat([json2df(df1), json2df(df2)])




category_list = os.listdir("Aves")
def clean_aves(path = "Aves", element = ".ipynb_checkpoints", cat_list = category_list):
    for cat in cat_list:
        try:
            shutil.rmtree(path+"/"+cat+"/"+element)
            print("checkpoint removed")
        except:
            continue

def clean_data(path = "data", element = ".ipynb_checkpoints"):
    for folder in ["images", "labels"]:
        for i in ["Train", "Val", "Test"]:
            try:
                shutil.rmtree(path+"/"+folder+"/"+i+"/"+element)
                print("checkpoint removed")
            except:
                continue
                


def encode_df(df, var_name = "category_id"):
    # encoding categories
    # One of the models we use, accepts only encoded categories
    encode_cat_dic = {'Ardea alba': 1,
     'Melospiza melodia': 2,
     'Buteo jamaicensis': 3,
     'Pandion haliaetus': 4,
     'Junco hyemalis': 5,
     'Zenaida macroura': 6,
     'Cardinalis cardinalis': 7,
     'Picoides pubescens': 8,
     'Agelaius phoeniceus': 9,
     'Ardea herodias': 10}
    df[var_name] = df.category_name.apply(lambda x: encode_cat_dic[x])
    return df