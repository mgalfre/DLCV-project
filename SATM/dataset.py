import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset
import albumentations.pytorch
import albumentations
import torchvision
from torch.utils.data.sampler import Sampler
from torchvision import transforms, datasets, models
import cv2
import random

##################################### DATASET CLASSES #########################################################################################################

class AvesAugmentationDataset(Dataset):
    def __init__(self, path, transform, df, num_aug=1): #maybe use explicit variable for train and test
        '''
        path: path to train folder or test folder
        df: dataframe to take info of box from
        train: if train or not
        '''
        # define the path to the images and what transform will be used
        
        self.path = path
        self.imgs = list(sorted(os.listdir(self.path)))
        #self.imgs = list(os.listdir(self.path))
        #self.imgs = list((df.identifier))
        self.transform = transform
        self.df = df
        self.num_aug = num_aug

    def __len__(self): 
        return len(self.imgs)*self.num_aug 
    
    def check_bbox(self,bbox,height,width):
        """Check if bbox boundaries are in range 0, 1 and minimums are lesser then maximums"""
        xmin, ymin, xmax, ymax = bbox
        if int(xmax) > width:
            print("Error in box width")
            xmax = width
        elif ymax > height:
            print("Error in box height")
            ymax = height
        
        bbox = [xmin, ymin, xmax, ymax]
        bbox = torch.Tensor(bbox).to(dtype=torch.float32)
        
        return bbox
        
    def aug_idx(self,idx):
        return idx//self.num_aug
    
        
    def __getitem__(self, idx): #special method
        # load images ad masks
        
        curr_idx = self.aug_idx(idx)
        
        file_image = self.imgs[curr_idx]
        
        img_path = os.path.join(self.path, file_image)
        #print(img_path)

        file_df = self.df[self.df.identifier == file_image]
        
        #self.width = file_df.width.iloc[0]
        #self.height = file_df.height.iloc[0]
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #print(image.shape)
        
        """if "bw" in self.path:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            print(image.shape)"""
        
        target = generate_target(file_df)
            
        if idx%self.num_aug==0:
            bbox_transform = albumentations.Compose([albumentations.pytorch.transforms.ToTensorV2(p=1)],
                                                    bbox_params = albumentations.BboxParams(format='pascal_voc',
                                                                                            label_fields=['labels']))
        else:
            bbox_transform = albumentations.Compose(self.transform+[albumentations.pytorch.transforms.ToTensorV2(p=1)],
                                                    bbox_params = albumentations.BboxParams(format='pascal_voc',
                                                                                            label_fields=['labels']))

        while True:

                
            result_image = image
            result_target = target
            
            """if "bw" in self.path:
                height, width = image.shape
            else:"""
            
            height, width, _ = image.shape
            
            for i,box in enumerate(result_target['boxes']):
                result_target['boxes'][i] = self.check_bbox(box,height, width)
            
            transformed = bbox_transform(image = result_image,
                                         bboxes = result_target['boxes'],
                                         labels = result_target['labels'])
                
            result_image = transformed['image']/255
        
            result_target = {'boxes':torch.Tensor(transformed['bboxes']).to(dtype=torch.float32),
                      'labels':torch.Tensor(list(transformed['labels'])).to(dtype=torch.int64)}
            
            if result_target['boxes'].size() != torch.Size([0]) and result_target['labels'].size() != torch.Size([0]):
                break
            
        return result_image, result_target, file_image


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
        
        img_path = os.path.join(self.path, file_image)

        file_df = self.df[self.df.identifier == file_image]

        img = Image.open(img_path).convert("RGB")
        #Generate Label
        target = generate_target(file_df)
        
        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self): 
        return len(self.imgs)   
    

class AvesStochAugDataset(Dataset):
    def __init__(self, path, transform, df, labels, prop_aug = 0.3, balanced = True): #maybe use explicit variable for train and test
        '''
        path: path to train folder or test folder
        df: dataframe to take info of box from
        train: if train or not
        '''
        # define the path to the images and what transform will be used
        
        self.path = path
        self.imgs = list(os.listdir(self.path))
        self.transform = transform
        self.df = df
        self.prop_aug = prop_aug

    def __len__(self): 
        return len(self.imgs)
    
    def check_bbox(self,bbox,height,width):
        """Check if bbox boundaries are in range 0, 1 and minimums are lesser then maximums"""
        xmin, ymin, xmax, ymax = bbox
        if int(xmax) > width:
            print("Error in box width")
            xmax = width
        elif ymax > height:
            print("Error in box height")
            ymax = height
        
        bbox = [xmin, ymin, xmax, ymax]
        bbox=torch.Tensor(bbox).to(dtype=torch.float32)
        
        return bbox
    
        
    def __getitem__(self, idx): #special method
        # load images ad masks
        
        file_image = self.imgs[idx]
        
        img_path = os.path.join(self.path, file_image)

        file_df = self.df[self.df.identifier == file_image]
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        target = generate_target(file_df)
            
        if random.random() >= self.prop_aug: # prop_aug = prob of augmenting
            bbox_transform = albumentations.Compose([albumentations.pytorch.transforms.ToTensorV2(p=1)],
                                                    bbox_params = albumentations.BboxParams(format='pascal_voc',
                                                                                            label_fields=['labels']))
        else:
            bbox_transform = albumentations.Compose(self.transform+[albumentations.pytorch.transforms.ToTensorV2(p=1)],
                                                    bbox_params = albumentations.BboxParams(format='pascal_voc',
                                                                                            label_fields=['labels']))
        while True:

                
            result_image = image
            result_target = target
            
            height, width, _ = image.shape
            
            for i,box in enumerate(result_target['boxes']):
                result_target['boxes'][i] = self.check_bbox(box,height, width)
            
            transformed = bbox_transform(image = result_image,
                                         bboxes = result_target['boxes'],
                                         labels = result_target['labels'])
                
            result_image = transformed['image']/255
        
            result_target = {'boxes':torch.Tensor(transformed['bboxes']).to(dtype=torch.float32),
                      'labels':torch.Tensor(list(transformed['labels'])).to(dtype=torch.int64)}
            
            if result_target['boxes'].size() != torch.Size([0]) and result_target['labels'].size() != torch.Size([0]):
                break
            
        return result_image, result_target
     

##################################### SAMPLER CLASSES #########################################################################################################

        
class SATMsampler(Sampler):
    """
    Create an equally balanced sampler, which allows for both undersampling and oversampling.
    """
    
    def __init__(self, data_source, size, labels):
        self.data_source = data_source
        self.size = size
        if self.size > len(data_source):
            self.oversampling = True
        else:
            self.oversampling = False
        self.labels = labels
        self.classes_dict = {i:0 for i in range(len(set(self.labels)))}
        self.per_classes_samples = {l: list(self.labels[self.labels==l].index) for l in set(self.labels)}
        self.augmenting = {l: list(self.labels[self.labels==l].index) for l in set(self.labels)}
        self.random_sample = [self.sample_idx() for idx in range(self.size)]
        #self.curr_iter = None
                    
    def sample_idx(self):
        random_class = random.randint(0,9)
        self.classes_dict[random_class] += 1
        idx = random.choice(self.per_classes_samples[random_class])
        if idx in set(self.augmenting[random_class]):
            self.augmenting[random_class].remove(idx)
        if not self.oversampling:
            self.per_classes_samples[random_class].remove(idx)
        return idx    
            
    def __iter__(self):
        #if not self.curr_iter:
        #    self.curr_iter = 0   
        #self.curr_iter += 1
        return iter(self.random_sample)

    def __len__(self):
        return self.size
    
    def class_prop(self):
        return {i:j/sum(self.classes_dict.values()) for i,j in self.classes_dict.items()}
    
    def reset(self):
        self.classes_dict = {i:0 for i in range(len(set(self.labels)))}
        self.random_sample = [self.sample_idx() for idx in range(self.size)]
        self.per_classes_samples = {l: list(labels[labels==l].index) for l in set(self.labels)}
        

############################## FUNCTIONS ##################################################################################################################### 
    

def collate_fn(batch):
    return tuple(zip(*batch))

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