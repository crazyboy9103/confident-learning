
from tqdm import tqdm
import numpy as np

from .utils import batch_iou, similarity, softmin1d_pooling

class Detection:
    def __init__(self, labels, preds):
        self.labels = labels
        self.preds = preds
    
    def badloc_scores(self, pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, min_confidence, alpha):
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
        scores = []
        for gt_box, gt_label in zip(gt_boxes, gt_labels):
            # Find the predicted boxes 
            #   1. that have high confidence 
            #   2. are of different class from the GT
            high_confidence_wrong_class = (pred_labels != gt_label) & (pred_scores > min_confidence)
            
            # If there is no such prediction box, the score is 1 (correct)
            if not any(high_confidence_wrong_class):
                score = 1.0  

            else:
                wrong_class_boxes = pred_boxes[high_confidence_wrong_class]
                # Compute the max similarity for the prediction boxes that satisfy the above conditions
                # The score is 1 - max similarity (higher similarity means lower score)
                # (the image contains prediction boxes that are of different class from the GT, 
                # so if the prediction boxes are close enough to the GT, it is likely to be a swapped case)
                score = 1 - similarity(gt_box, wrong_class_boxes, alpha).max()
            scores.append(score)

        return scores
    
    def min_similarity(self, alpha):
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
        preds = self.preds
        labels = self.labels

        min_similarity = self.min_similarity(alpha=alpha)

        badloc_scores = []
        overlooked_scores = []
        swapped_scores = []

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