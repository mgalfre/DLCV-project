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
from sklearn.decomposition import PCA



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
    
    annotations = pd.DataFrame(df["annotations"]) # building a pandas df with the annotations
    images = pd.DataFrame(df["images"]) # building a pandas df with with the images
    # merging the annotations and images dataframe, basing on the images' keys (unique id)
    merged = pd.merge(images, annotations, left_on="id", right_on= "image_id").rename({"id_x":"id"}, axis = 1)
    
    # identifier is a string with the name of the .jpg image
    merged["identifier"] = merged.apply(lambda x: x.file_name.split("/")[-1], axis = 1)
    
    # retrieving the category (what type of bird)
    merged["category_name"] = merged.file_name.apply(lambda x: x.split("/")[2])
    
    # retrieving the super category (what species)
    merged["super_category_name"] = merged.file_name.apply(lambda x: x.split("/")[1])
    
    
    return merged.sample(frac = 1, random_state = 810)


def merge_aves_df(df1 = "train_2017_bboxes.json", df2 = "val_2017_bboxes.json"):
    """
    this function merges the information contained in the two json file
    that we use as source
    """
    return pd.concat([json2df(df1), json2df(df2)])




def clean_aves(cat_list, path = "Aves", element = ".ipynb_checkpoints"):
    """
    going through all the categories to delete unnecessary files (.ipynb_checkpoints files)
    """
    for cat in cat_list:
        try:
            shutil.rmtree(path+"/"+cat+"/"+element)
            print("checkpoint removed")
        except:
            continue

def clean_data(path = "data", element = ".ipynb_checkpoints"):
    """
    going through all the train, val and test folders
    to delete unnecessary files (.ipynb_checkpoints files)
    """
    for folder in ["images", "labels"]:
        for i in ["Train", "Val", "Test"]:
            try:
                shutil.rmtree(path+"/"+folder+"/"+i+"/"+element)
                print("checkpoint removed")
            except:
                continue
                


def encode_df(df, var_name = "category_id"):
    """
    the model we use need an encoded version of the category
    this function maps the category into a integer ranging from 1 to 10
    """
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



def convert_to_yolov5(which, df):
    """
    yolo needs images to be rescaled in 1:1 format
    a coherent transformation needs to be perfomed also in the boxes coordinates
    this function also writes the labels in .txt file as required by yolo
    """
    print_buffer = []
    input_path = os.listdir("data/images/"+which)
    for path in tqdm(input_path):
        
        temp_df = df[df.identifier == path][["category_id","bbox","width", "height"]]
        #print(path)
        with open("data/labels/"+which+"/"+path[:-3]+"txt", "w") as fp:
            for i in range(len(temp_df)):
                class_id = temp_df.iloc[i].category_id - 1 # retrieving the class id
                # note that yolo requires class ids starting from 0
                # whereas our initial class id ranged from 1 to 10
                # we just subtract -1 from each id, to have yolo-compatible ids
                
                # in the json files we are provided with the left upper corner, height and width
                # instead yolo requires the coordinates of the center of the box, computed below
                b_x, b_y, b_width, b_height  = temp_df.iloc[i].bbox
                b_center_x = b_x + b_width/2
                b_center_y = b_y + b_height/2
                
                
                image_w, image_h = temp_df[["width", "height"]].iloc[0]
                b_center_x /= image_w 
                b_center_y /= image_h 
                b_width    /= image_w 
                b_height   /= image_h 
                
                # preparing the line to be written in the .txt file corresponding to each image
                line = "{} {:.3f} {:.3f} {:.3f} {:.3f}".format(class_id, b_center_x, b_center_y, b_width, b_height)
                #print(line)
                fp.writelines(line+"\n") # writing the line
                
                
                
                
def generate_txt(identifier, transformed, v):
    """
    prepares .txt files containing box and label information
    """
    boxes = transformed["bboxes"]
    labels = [i.item() -1  for i in transformed["labels"]]
    with open("data_over/labels/Train/"+identifier[:-4]+v+".txt", "w") as fp:
        for i in range(len(boxes)):
            class_id = labels[i]


            b_x, b_y, b_width, b_height  = boxes[i]
            b_center_x = b_x + b_width/2
            b_center_y = b_y + b_height/2

            image_w, image_h = train_df[train_df.identifier == identifier][["width", "height"]].iloc[0]

            b_center_x /= image_w 
            b_center_y /= image_h 
            b_width    /= image_w 
            b_height   /= image_h 

            line = "{} {:.3f} {:.3f} {:.3f} {:.3f}".format(class_id, b_center_x, b_center_y, b_width, b_height)
            #print(line)
            fp.writelines(line+"\n")
            
            

            
def generate_box(l):
    xmin = l[0]
    xmax = xmin +l[2]
    ymin = l[1]
    ymax = ymin + l[3]
    return [xmin, ymin, xmax, ymax]
    
def generate_target(df):
    boxes = []
    labels = []
    for i in range(df.shape[0]):
        boxes.append(generate_box(df.iloc[i].bbox))
        labels.append(df.iloc[i].category_id)
    boxes = torch.as_tensor(boxes, dtype=torch.float32) 
    labels = torch.as_tensor(labels, dtype=torch.int64) 
    target = {}
    target["boxes"] = boxes
    target["labels"] = labels
    return target


def plot_image_from_output(img, annotation):
    
    img = img.cpu().permute(1,2,0)
    
    fig,ax = plt.subplots(1)
    ax.imshow(img)
    
    for idx in range(len(annotation["boxes"])):
        
        xmin, ymin, xmax, ymax = annotation["boxes"][idx]

        rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='red',facecolor='none')

        ax.add_patch(rect)

    plt.show()            
            
class AvesDataset(object):
    def __init__(self, transforms, path,  df): #maybe use explicit variable for train and test
        '''
        path: path to train folder or test folder
        df: dataframe to take info of box from
        train: if train or not
        '''
        # define the path to the images and what transform will be used
        self.transforms = transforms
        self.path = path
        self.imgs = list(sorted(os.listdir(self.path)))
        self.df = df
        #self.train = train ----> should be true or false


    def __getitem__(self, idx): #special method
        # load images ad masks
        file_image = self.imgs[idx]
        
        #print(file_image)
        #print(file_image)
        #file_label = self.imgs[idx][:-3] + 'xml' #\here use "json" instead of "xml"
        img_path = os.path.join(self.path, file_image)
        #print(file_image)
        file_df = self.df[self.df.identifier == file_image]
        #print(file_df.head())
        #print(file)

        #if 'test' in self.path: #if self.train == False

        #print(img_path)
        img = Image.open(img_path).convert("RGB")
        #Generate Label
        target = generate_target(file_df)
        
        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self): 
        return len(self.imgs)
    
    
    
    
    
def plot_images(images):
    if isinstance(images, list) and len(images) > 1:
        fig, axes = plt.subplots(nrows=1, ncols=len(images), figsize=(12,8))
        for ax, im in zip(axes, images):
            ax.imshow(im)
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
    else:
        fig, ax = plt.subplots(figsize=(12,8))
        try:
            ax.imshow(images)
        except:
            ax.imshow(images[0])
        finally:
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])

            

def scatter_with_pca(texts, images):
    if not isinstance(images, list):
        images = [images]

    text_embeds = embed_texts(texts, processor, model)
    image_embed = embed_images(images, processor, model)

    pca = PCA(n_components=2)
    X = torch.cat([text_embeds, image_embed], axis=0)
    X_2d = pca.fit_transform(X)

    X_texts = X_2d[:len(texts), :]
    X_images = X_2d[len(texts):, :]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(X_texts[:, 0], X_texts[:, 1], c="darkorange", label="Text")
    
    for idx, image in tqdm(enumerate(images)):
        imagebox = OffsetImage(image, zoom=0.1)
        ab = AnnotationBbox(imagebox, (X_images[idx, 0] - .6, X_images[idx, 1] - .5))
        ax.add_artist(ab)

    ax.scatter(X_images[:, 0], X_images[:, 1], c="royalblue", label="Image")

    plt.legend()

    for index, word in enumerate(texts):
        # annotate
        plt.annotate(word, xy=(X_texts[index, 0], X_texts[index, 1]))


def get_image_from_url(url):
    return Image.open(url).convert("RGB")
    # return Image.open(requests.get(url, stream=True).raw).convert("RGB")

def embed_texts(texts, processor, model):
    inputs = processor(text=texts, padding="longest")
    input_ids = torch.tensor(inputs["input_ids"])
    attention_mask = torch.tensor(inputs["attention_mask"])
    
    with torch.no_grad():    
        embeddings = model.get_text_features(
            input_ids=input_ids, attention_mask=attention_mask
        )
    return embeddings

def embed_images(images, processor, model):
    inputs = processor(images=images)
    pixel_values = torch.tensor(np.array(inputs["pixel_values"]))

    with torch.no_grad():
        embeddings = model.get_image_features(pixel_values=pixel_values)
    return embeddings


def legend_without_duplicate_labels(figure, loc = 'lower right'):
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    figure.legend(by_label.values(), by_label.keys(), loc=loc,  frameon=False)
    
    
def scatter_pca(image_embed, c_scatter, cat_scatter):
    from sklearn.decomposition import PCA
    """
    this function receives as input a list of embedded images
    the embeddings are reduced to 2 dimensions using PCA
    and plotted in a 2D scatter plot
    """

    pca = PCA(n_components=2)
    X = torch.cat([image_embed], axis=0)
    X_2d = pca.fit_transform(X)

    #X_texts = X_2d[:len(texts), :]
    X_images = X_2d[:, :]

    plt.figure(figsize=(10, 8))

    for i in tqdm(range(len(image_embed))):
        plt.scatter(X_images[i, 0], X_images[i, 1], c=c_scatter[i], label = cat_scatter[i])
    legend_without_duplicate_labels(plt, loc = "upper right")
    plt.title("CLIP embedding and PCA", weight = "bold", fontsize = 20)
    plt.savefig("plots/CLIP.pdf", transparent = True)