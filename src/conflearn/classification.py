from typing import Literal

import numpy as np

class Classification:
    def __init__(self, labels, preds, num_classes):
        """
        Initialize the Classification instance for finding low-quality labels by assessing classification model predictions.
        
        Args:
            labels (np.array): Ground truth labels, shape (N,).
            preds (np.array): Predicted probabilities for each class, shape (N, num_classes).
            num_classes (int): Number of unique classes.
        """
        self.labels = labels
        self.preds = preds
        self.num_classes = num_classes

    def per_class_thresholds(self):
        """
        Calculate the mean predicted confidence for each class where the ground truth is that class.
        
        Returns:
            np.array: An array of thresholds for each class.
        """
        thresholds = np.zeros(self.num_classes)
        for i in range(self.num_classes):
            # self-confidences of predictions whose GT is class i
            p_i = self.preds[self.labels == i, i]
            # Mean self-confidence for class i as the threshold
            thresholds[i] = np.mean(p_i)

        if np.any(thresholds == 0):
            raise ValueError("Some classes have no predictions, resulting in zero thresholds.")
        
        return thresholds

    def confident_joint(self):
        joint = [[] * self.num_classes for _ in range(self.num_classes)]
        thresholds = self.per_class_thresholds()
        return joint

    def get_result(self, method: Literal["pl", "cl"] = "cl", score_method: Literal["self_confidence", "normalized_margin"] = "self_confidence"):
        """
        Compute error masks and label quality scores based on specified methods.

        Args:
            method (str): The method to compute errors ('pl' for pseudo-labeling, 'cl' for confident learning).
            score_method (str): The method to compute label quality scores ('self_confidence' or 'normalized_margin').

        Returns:
            tuple: A boolean mask of errors and an array of label quality scores.
        """
        match method:
            case "pl":
                pseudo_labels = self.preds.argmax(axis=1)
                error_mask = pseudo_labels != self.labels

            case "cl":
                thresholds = self.per_class_thresholds()
                above_thresholds = self.preds >= thresholds
            
                error_mask = []
                for i in range(len(self.preds)):
                    indices = above_thresholds[i].nonzero()[0]
                    if len(indices) == 0:
                        # ignore pred if none of the classes are above threshold
                        error_mask.append(False)
                        continue
                    
                    most_confident_class = indices[self.preds[i, indices].argmax()]
                    error = self.labels[i] != most_confident_class and self.preds[i, most_confident_class] >= thresholds[most_confident_class]
                    error_mask.append(error)

                error_mask = np.array(error_mask)

            case "PBC":
                raise NotImplementedError
        
            case "PBNR":
                raise NotImplementedError

            case "C+NR":
                raise NotImplementedError

            case _:
                raise ValueError(f"Unknown method: {method}")
        
        match score_method:
            case "self_confidence":
                label_quality_scores = self.preds[np.arange(len(self.preds)), self.labels]
            
            case "normalized_margin":
                # p_y-p_{y'}, y'=argmax_k!=y p_k 
                self_confidence = self.preds[np.arange(len(self.preds)), self.labels]
                preds = self.preds.copy()
                preds[np.arange(len(self.preds)), self.labels] = 0
                max_confidence = preds.max(axis=1)
                # normalize [-1, 1] to [0, 1]
                label_quality_scores = (self_confidence - max_confidence + 1) / 2

            case _:
                raise ValueError(f"Unknown score_method: {score_method}")
            
        return error_mask, label_quality_scores
    
if __name__ == "__main__":
    # from torchvision import models
    # weights = models.get_weight("ResNet18_Weights.IMAGENET1K_V1")
    # resnet18 = models.resnet18(weights=weights)
    # transform = weights.transforms(antialias=True)

    # from torchvision.datasets import ImageNet
    # imagenet = ImageNet(root="/datasets/imagenet/val", split="val", transform=transform)

    # from torch.utils.data import DataLoader
    # dataloader = DataLoader(imagenet, batch_size=128, num_workers=8)
    # from tqdm import tqdm
    # resnet18 = resnet18.cuda().eval()

    # p_ij = []
    # y_i = []

    # for images, labels in tqdm(dataloader):
    #     images = images.cuda()
    #     preds = resnet18(images).softmax(dim=1).detach().cpu().numpy()
    #     labels = labels.numpy()

    #     p_ij.append(preds)
    #     y_i.append(labels)
    # cat_p_ij = np.concatenate(p_ij, axis=0)
    # cat_y_i = np.concatenate(y_i, axis=0)
    import torch
    from cleanlab.filter import find_label_issues, get_label_quality_scores
    # torch.save(cat_p_ij, "cla_p.pt")
    # torch.save(cat_y_i, "cla_y.pt")
    cat_p_ij = torch.load("cla_p.pt")
    cat_y_i = torch.load("cla_y.pt")
    cla=Classification(cat_y_i, cat_p_ij, num_classes=1000)

    for method, cl_method in zip(("cl", "pl"), ("confident_learning", "predicted_neq_given")):
        for score_method in ("self_confidence", "normalized_margin"):
            error_indices, scores = cla.get_result(method, score_method)

            cl_indices = find_label_issues(cat_y_i, cat_p_ij, filter_by=cl_method).nonzero()[0]
            cl_sc = get_label_quality_scores(cat_y_i, cat_p_ij, method=score_method)

            print(f"Method: {method}, CL Method: {cl_method}, Score Method: {score_method}")
            print(f"Error scores: {np.all(scores == cl_sc)}")