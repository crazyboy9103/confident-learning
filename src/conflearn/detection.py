
from tqdm import tqdm
import numpy as np

from .utils import batch_iou, similarity, softmin1d_pooling

class Detection:
    def __init__(self, labels, preds):
        """
        Initialize the Detection instance for assessing detection model predictions.
        Expects (x1, y1, x2, y2) format for bounding boxes.

        Args:
            labels (list): List of dictionaries with keys 'boxes' (ground truth bounding boxes),
                           'labels' (ground truth labels).
            preds (list): List of dictionaries with keys 'boxes' (predicted bounding boxes),
                          'scores' (confidence scores), 'labels' (predicted labels).
        """
        self.labels = labels
        self.preds = preds
    
    def badloc_scores(self, pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, min_confidence, alpha):
        """
        Calculates scores indicating the quality of localization based on overlap and class accuracy above a confidence threshold.
        
        Args:
            pred_boxes (np.array): Array of predicted bounding boxes.
            pred_scores (np.array): Array of prediction confidence scores.
            pred_labels (np.array): Array of predicted labels.
            gt_boxes (np.array): Array of ground truth bounding boxes.
            gt_labels (np.array): Array of ground truth labels.
            min_confidence (float): Minimum confidence score to consider a prediction valid.
            alpha (float): Hyperparameter for adjusting the impact of similarity in scoring.
        
        Returns:
            List[float]: Localization error scores for each ground truth box.
        """
        scores = []
        for gt_box, gt_label in zip(gt_boxes, gt_labels):
            overlapping = (batch_iou(gt_box, pred_boxes) > 0).squeeze(0)
            confident = pred_scores >= min_confidence
            matching = pred_labels == gt_label

            # Compute and use the max similarity as the score for the prediction boxes that
            #   1. are of the same class as the GT
            #   2. have high confidence
            #   3. overlap with the GT
            if any(matching & confident & overlapping):
                score = similarity(pred_boxes[matching & confident & overlapping], gt_box, alpha).max()

            # If there is no prediction box that satisfies the above conditions, the score is 1 (correct)
            else:
                score = 1.0
            
            scores.append(score)
        return scores
    
    def overlooked_scores(self, pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, min_similarity, min_confidence, alpha):
        """
        Calculates scores for potentially overlooked predictions that are not justified by the ground truth.
        
        Returns:
            List[float]: Scores for each prediction box, indicating 1-likelihood of being an overlooked false negative.
                         (lower score means higher likelihood of being overlooked)
        """
        # If there is no GT, ignore all prediction boxes that have low confidence (nan)
        # and score min_similarity * (1 - confidence) (overlooked) for the rest
        if len(gt_boxes) == 0:
            scores = np.zeros_like(pred_scores)
            high_confidence = pred_scores >= min_confidence
            scores[high_confidence] = min_similarity * (1-pred_scores[high_confidence])
            scores[~high_confidence] = np.nan
            return scores
        
        scores = []
        for predicted_box, predicted_confidence, predicted_label in zip(pred_boxes, pred_scores, pred_labels):
            overlapping = (batch_iou(predicted_box, gt_boxes) > 0).squeeze(0)
            confident = predicted_confidence >= min_confidence
            matching = gt_labels == predicted_label

            # If the predicted box has low confidence or overlaps with at least one GT, ignore it
            # (in the latter case, the box will be scored in badloc_scores)
            if not confident or any(overlapping):
                score = np.nan

            else:
                # If there is no GT with same class as the predicted box, compute the score as
                # min_similarity * (1 - confidence)
                # Higher confidence or lower minimum similarity will result in lower score (more likely to be overlooked)
                if not any(matching):
                    score = min_similarity * (1-predicted_confidence)

                else:
                    # If the predicted box  
                    #   1. does not overlap with any GT
                    #   2. has high confidence
                    #   3. is of same class as at least one GT
                    # Compute the max similarity as the score
                    score = similarity(predicted_box, gt_boxes[matching], alpha).max()

            scores.append(score)

        return scores

    def swapped_scores(self, pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, min_confidence, alpha):
        """
        Computes scores for class swap errors where the predicted class does not match the ground truth class but overlaps spatially.
        
        Returns:
            List[float]: Scores indicating likelihood of class swap errors for each ground truth box.
        """
        scores = []
        for gt_box, gt_label in zip(gt_boxes, gt_labels):
            # Find the predicted boxes 
            #   1. that have high confidence 
            #   2. are of different class from the GT
            confident = pred_scores > min_confidence
            not_matching = pred_labels != gt_label
            overlapping = (batch_iou(gt_box, pred_boxes) > 0).squeeze(0)
            # If there is no such prediction box, the score is 1 (correct)
            if not any(confident & not_matching & overlapping):
                score = 1.0  

            else:
                # Compute the max similarity for the prediction boxes that satisfy the above conditions
                # The score is 1 - max similarity (higher similarity means lower score)
                # (the image contains prediction boxes that are of different class from the GT, 
                # so if the prediction boxes are close enough to the GT, it is likely a swapped case)
                score = 1 - similarity(gt_box, pred_boxes[confident & not_matching & overlapping], alpha).max()
            scores.append(score)

        return scores
    
    def min_similarity(self, alpha):
        """
        Compute the minimum similarity score across all predictions and ground truths in all images. 

        Args:
            alpha (float): Hyperparameter for the similarity computation.

        Returns:
            float: Minimum similarity score observed.
        """
        preds = self.preds
        labels = self.labels
        
        min_similarity = 1.0
        for pred, label in zip(preds, labels):
            pred_boxes = pred["boxes"]
            gt_boxes = label["boxes"]
            
            # Ignore if there are neither prediction nor GT boxes
            if len(pred_boxes) == 0 or len(gt_boxes) == 0:
                continue
            
            similarity_matrix = similarity(pred_boxes, gt_boxes, alpha=alpha)
            
            # Take only positive similarities
            positive_similarity = similarity_matrix[similarity_matrix > 0]

            if positive_similarity.size == 0:
                continue
            
            min_similarity = min(min_similarity, positive_similarity.min())

        return min_similarity        
    
    def get_result(self, alpha=0.1, badloc_min_confidence=0.5, min_confidence = 0.95, pooling=True, softmin_temperature=0.1):
        """
        Aggregates scores across various metrics for all predictions.
        
        Args:
            alpha (float): Hyperparameter for the similarity computation.
            badloc_min_confidence (float): Minimum confidence score to consider a prediction valid for bad localization.
            min_confidence (float): Minimum confidence score to consider a prediction valid for overlooked and swapped cases.
            pooling (bool): Whether to apply softmin pooling to the per-box scores to get per-image scores.
            softmin_temperature (float): Temperature parameter for softmin pooling, only used if pooling is True.
        Returns:
            tuple: Collections of scores for bad localization, overlooked predictions, and swapped classes.
        """
        preds = self.preds
        labels = self.labels

        badloc_scores = []
        overlooked_scores = []
        swapped_scores = []
        
        min_similarity = self.min_similarity(alpha)

        for pred, label in tqdm(zip(preds, labels), total=len(preds), desc="Calculating scores"):
            pred_boxes = pred["boxes"]
            pred_scores = pred["scores"]
            pred_labels = pred["labels"]

            gt_boxes = label["boxes"]
            gt_labels = label["labels"]

            badloc_scores_per_box = self.badloc_scores(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, badloc_min_confidence, alpha)
            overlooked_scores_per_box = self.overlooked_scores(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, min_similarity, min_confidence, alpha)
            swapped_scores_per_box = self.swapped_scores(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, min_confidence, alpha)

            assert len(badloc_scores_per_box) == len(gt_boxes)
            assert len(overlooked_scores_per_box) == len(pred_boxes)
            assert len(swapped_scores_per_box) == len(gt_boxes)

            # Per-box scores to per-image score
            if pooling:
                badloc_scores_per_box = softmin1d_pooling(badloc_scores_per_box, temperature=softmin_temperature)
                overlooked_scores_per_box = softmin1d_pooling(overlooked_scores_per_box, temperature=softmin_temperature)
                swapped_scores_per_box = softmin1d_pooling(swapped_scores_per_box, temperature=softmin_temperature)

            badloc_scores.append(badloc_scores_per_box)
            overlooked_scores.append(overlooked_scores_per_box)
            swapped_scores.append(swapped_scores_per_box)

        return badloc_scores, overlooked_scores, swapped_scores        
    
if __name__ == "__main__":
    # from torchvision import models
    # weights = models.get_weight("FasterRCNN_ResNet50_FPN_Weights.COCO_V1")
    # model = models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    # transform = weights.transforms()

    # from torchvision.datasets import CocoDetection
    # coco = CocoDetection(
    #     root="/datasets/coco/val2017", 
    #     annFile="/datasets/coco/annotations_trainval2017/annotations/instances_val2017.json", 
    #     transform=transform,
    #     target_transform=None
    # )

    # import numpy as np
    # from tqdm import tqdm
    # model = model.cuda().eval()

    # p_ij = []
    # y_i = []

    # for images, labels in tqdm(coco):
    #     images = images.cuda().unsqueeze(0)
    #     preds = model(images)

    #     temp_boxes = []
    #     temp_labels = []
    #     for label in labels:
    #         bbox = label["bbox"]
    #         # x1, y1, w, h to x1, y1, x2, y2
    #         bbox[2] = bbox[0] + bbox[2]
    #         bbox[3] = bbox[1] + bbox[3]
    #         temp_boxes.append(bbox)
    #         temp_labels.append(label["category_id"])

    #     temp_boxes = np.array(temp_boxes)
    #     temp_labels = np.array(temp_labels)

    #     for k, v in preds[0].items():
    #         preds[0][k] = v.detach().cpu().numpy()


    #     p_ij.append(preds[0])
    #     y_i.append({"boxes": temp_boxes, "labels": temp_labels})

    import torch
    # torch.save(p_ij, "obd_p.pt")
    # torch.save(y_i, "obd_y.pt")

    # import numpy as np
    # from torchvision.transforms import functional as F
    # pil=F.to_pil_image(coco[0][0])

    # from PIL import ImageDraw
    # draw = ImageDraw.Draw(pil)

    # for pred in p_ij[0]['boxes']:
    #     draw.rectangle(pred, outline="red")

    # for gt in y_i[0]['boxes']:
    #     draw.rectangle(gt.astype(np.float32), outline="blue")
    p_ij = torch.load("obd_p.pt")
    y_i = torch.load("obd_y.pt")