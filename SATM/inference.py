import numpy as np
import pandas as pd
import torch
import cv2 
import pickle
import os
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import seaborn as sn

if torch.cuda.is_available():    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")  
    
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 

enc2cat = {1: 'Ardea alba',
 2: 'Melospiza melodia',
 3: 'Buteo jamaicensis',
 4: 'Pandion haliaetus',
 5: 'Junco hyemalis',
 6: 'Zenaida macroura',
 7: 'Cardinalis cardinalis',
 8: 'Picoides pubescens',
 9: 'Agelaius phoeniceus',
 10: 'Ardea herodias',
 11: 'Background'}

def make_prediction(model, img, threshold):
    # evaluation phase
    model.eval()
    preds = model(img)
    for id in range(len(preds)) :
        idx_list = []
        for idx, score in enumerate(preds[id]['scores']) :
            if score > threshold : 
                idx_list.append(idx)

        preds[id]['boxes'] = preds[id]['boxes'][idx_list]
        preds[id]['labels'] = preds[id]['labels'][idx_list]
        preds[id]['scores'] = preds[id]['scores'][idx_list]

    return preds

def get_batch_statistics(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i] # predict
        # pred_boxes = output['boxes']
        # pred_scores = output['scores']
        # pred_labels = output['labels']

        true_positives = torch.zeros(output['boxes'].shape[0])   # 예측 객체 개수 (number of predicted objects)
 
        annotations = targets[sample_i]  # actual
        target_labels = annotations['labels'] if len(annotations) else []
        if len(annotations):    # len(annotations) = 3
            detected_boxes = []
            target_boxes = annotations['boxes']

            for pred_i, (pred_box, pred_label) in enumerate(zip(output['boxes'], output['labels'])):

                # If targets are found break
                if len(detected_boxes) == len(target_labels): # annotations -> target_labels
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)  
                if iou >= iou_threshold and box_index not in detected_boxes: 
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]  
        batch_metrics.append([true_positives, output['scores'], output['labels']])
    return batch_metrics

def bbox_iou_cm(box1, box2):
    """
    Returns the IoU of two bounding boxes
    """

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    
    if type(b2_x1) != torch.Tensor:
        b2_x1 = torch.Tensor(np.array(b2_x1))
    if type(b2_y1) != torch.Tensor:
        b2_y1 = torch.Tensor(np.array(b2_y1))
    if type(b2_x2) != torch.Tensor:
        b2_x2 = torch.Tensor(np.array(b2_x2))
    if type(b2_y2) != torch.Tensor:
        b2_y2 = torch.Tensor(np.array(b2_y2))
    
    # get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes
    """

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    
    # get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = torch.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = torch.unique(target_cls)   

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in unique_classes:
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = torch.cumsum(1 - tp[i],-1)
            tpc = torch.cumsum(tp[i],-1)

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = torch.tensor(np.array(p)), torch.tensor(np.array(r)), torch.tensor(np.array(ap))
    f1 = 2 * p * r / (p + r + 1e-16) # + 1e-16 to avoid division by zero

    return p, r, ap, f1, unique_classes, precision_curve, recall_curve

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def pred_cm(model,data_loader):
    
    # storing results
    dict_ground_truth = dict()
    dict_result = dict()

    batch_size = data_loader.batch_size

    for im, annot, identifier in tqdm(data_loader, position = 0, leave = True):
        im = list(img.to(device) for img in im)

        for i in range(batch_size):
            id_ = identifier[i]
            annotation_image = annot[i]
            dict_ground_truth[id_] = {}
            dict_ground_truth[id_]["labels"] = annot[0]["labels"]
            dict_ground_truth[id_]["boxes"] = annot[0]["boxes"]

            with torch.no_grad():
                preds_adj = make_prediction(model, [im[i]], 0.5)
                preds_adj = [{k: v.to(torch.device('cpu')) for k, v in t.items()} for t in preds_adj]
                dict_result[id_] = preds_adj[0]
    
    return dict_ground_truth, dict_result


def create_confusion_matrix(dict_ground_truth, dict_result, nclasses):
    
    confusion_matrix = pd.DataFrame(np.zeros((nclasses+1,nclasses+1)), columns = np.arange(1,nclasses+2))
    confusion_matrix.set_index(np.arange(1,nclasses+2), inplace = True)

    for image in tqdm(dict_ground_truth.keys()):
        
        if len(dict_result[image]["boxes"]) > 0:
            box_to_compare_with = dict_result[image]["boxes"].tolist()
        else:
            box_to_compare_with = dict_result[image]["boxes"].tolist()
        if len(dict_result[image]["boxes"]) > 0:
            label_to_compare_with = dict_result[image]["labels"].tolist()
        else:
            label_to_compare_with = dict_result[image]["labels"].tolist()
        
        if len(box_to_compare_with)>0:
            for k in range(len(dict_ground_truth[image]["boxes"])): #take box in ground_truth
                box = dict_ground_truth[image]["boxes"][k]
                label = dict_ground_truth[image]["labels"][k]
                iou_dict = {}
                for i in range(len(box_to_compare_with)):
                    iou_dict[i] = bbox_iou_cm(box,box_to_compare_with[i])
                if len(iou_dict) == 0:
                    continue
                max_ = max(iou_dict, key = iou_dict.get)
                if iou_dict[max_] < 0.25:
                    confusion_matrix.loc[nclasses+1,label.item()] += 1 
                    continue
                box_to_compare_with.pop(max_)
            
                confusion_matrix.loc[label_to_compare_with[max_],label.item()] += 1
                label_to_compare_with.pop(max_)
                
            if len(box_to_compare_with) > 0:
                for k in range(len(box_to_compare_with)):
                    lab_ = label_to_compare_with[k]
                    confusion_matrix.loc[label.item(),nclasses+1] += 1
                    
    confusion_matrix.columns = enc2cat.values()
    confusion_matrix.index = enc2cat.values()
    
    confusion_matrix.loc[nclasses+2,:] = np.sum(confusion_matrix, axis = 0)
    
    final_confusion=confusion_matrix.apply(lambda x: x/confusion_matrix.iloc[nclasses+1,:], axis = 1)
    final_confusion = final_confusion.iloc[0:nclasses+1,0:nclasses+1]
    
    plt.figure(figsize = (10,8))
    plt.title("Confusion Matrix")
    sn.heatmap(round(final_confusion,2), annot=True,  cmap="Blues")
    plt.ylabel('Predicted')
    plt.xlabel('True')
    plt.xticks(rotation=90)
    plt.tight_layout()
            
    return final_confusion


def epoch_eval(model_dir,data_loader):
    
    mPrecisions = []
    mRecalls = []
    mAPs = []
    mF1s = []
    
    for epoch in tqdm(range(1,len(os.listdir(model_dir))+1)): 
    
        model_directory = f"{model_dir}/model_epoch{epoch}.pkl"

        file = open(model_directory, "rb")
        model_epoch = pickle.load(file)
        file.close()

        labels = []
        preds_adj_all = []
        annot_all = []

        for im, annot, _ in tqdm(data_loader, position = 0, leave = True):
            im = list(img.to(device) for img in im)

            for t in annot:
                labels += t['labels']

            with torch.no_grad():
                preds_adj = make_prediction(model_epoch, im, 0.5)
                preds_adj = [{k: v.to(torch.device('cpu')) for k, v in t.items()} for t in preds_adj]
                preds_adj_all.append(preds_adj)
                annot_all.append(annot)

        sample_metrics = []
        for batch_i in range(len(preds_adj_all)):
            sample_metrics += get_batch_statistics(preds_adj_all[batch_i], annot_all[batch_i], iou_threshold=0.5)

        true_positives, pred_scores, pred_labels = [torch.cat(x, 0) for x in list(zip(*sample_metrics))]

        precision, recall, AP, f1, ap_class, precision_curve, recall_curve = ap_per_class(true_positives, pred_scores,                                                                                               pred_labels, torch.tensor(labels))

        mPrecisions.append(torch.mean(precision))
        mRecalls.append(torch.mean(recall))
        mAPs.append(torch.mean(AP))
        mF1s.append(torch.mean(f1))
    
    return mPrecisions, mRecalls, mAPs, mF1s


def plot_epoch_eval(model_dir, mPrecisions, mRecalls, mAPs, mF1s):
    plt.figure(figsize=(8,6))
    epochs = range(1,len(os.listdir(model_dir))+1)
    plt.title("Test Performances")
    plt.plot(epochs,mPrecisions,c="mediumturquoise",label="mPrecision")
    plt.plot(epochs,mRecalls,c="forestgreen",label="mRecall")
    plt.plot(epochs,mAPs,c="firebrick",label="mAP")
    plt.plot(epochs,mF1s,c="royalblue",label="mF1")
    plt.xlabel("Epochs")
    plt.xticks([1,2,3,4])
    plt.legend()
    plt.tight_layout()