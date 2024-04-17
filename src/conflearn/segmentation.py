from tqdm import tqdm

from utils import softmin

class Segmentation:
    def __init__(self, labels, pred_masks, pred_max_probs, pred_selfconf_probs, num_classes):
        self.labels = labels
        self.pred_masks = pred_masks
        self.pred_max_probs = pred_max_probs
        self.pred_selfconf_probs = pred_selfconf_probs
        self.num_classes = num_classes
    
    def get_result(self, pooling=False, softmin_temperature=0.1):
        scores = []
        for pred_selfconf, gt_mask in tqdm(zip(self.pred_selfconf_probs, self.labels), total=len(self.labels), desc="Calculating scores"):
            score = pred_selfconf
            if pooling:
                weights = softmin(pred_selfconf, temperature=softmin_temperature)
                score = weights * pred_selfconf
                score = score.sum()
            
            scores.append(score)
        return scores
    
if __name__ == "__main__":
    # Below is an example of how to use the Segmentation class and compare the results with cleanlab
    # with some visualization code at the end
    # # Adapted from torchvision/references/segmentation/coco_utils.py
    # import os
    # import copy 

    # from PIL import Image
    # from pycocotools import mask as coco_mask
    # import numpy as np
    # import torch
    # from torchvision.datasets import CocoDetection

    # CAT_LIST = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4, 1, 64, 20, 63, 7, 72]

    # class FilterAndRemapCocoCategories:
    #     def __init__(self, categories, remap=True):
    #         self.categories = categories
    #         self.remap = remap

    #     def __call__(self, image, anno):
    #         anno = [obj for obj in anno if obj["category_id"] in self.categories]
    #         if not self.remap:
    #             return image, anno
    #         anno = copy.deepcopy(anno)
    #         for obj in anno:
    #             obj["category_id"] = self.categories.index(obj["category_id"])
    #         return image, anno

    # def convert_coco_poly_to_mask(segmentations, height, width):
    #     masks = []
    #     for polygons in segmentations:
    #         rles = coco_mask.frPyObjects(polygons, height, width)
    #         mask = coco_mask.decode(rles)
    #         if len(mask.shape) < 3:
    #             mask = mask[..., None]
    #         mask = torch.as_tensor(mask, dtype=torch.uint8)
    #         mask = mask.any(dim=2)
    #         masks.append(mask)
    #     if masks:
    #         masks = torch.stack(masks, dim=0)
    #     else:
    #         masks = torch.zeros((0, height, width), dtype=torch.uint8)
    #     return masks

    # class ConvertCocoPolysToMask:
    #     def __call__(self, image, anno):
    #         w, h = image.size
    #         segmentations = [obj["segmentation"] for obj in anno]
    #         cats = [obj["category_id"] for obj in anno]
    #         if segmentations:
    #             masks = convert_coco_poly_to_mask(segmentations, h, w)
    #             cats = torch.as_tensor(cats, dtype=masks.dtype)
    #             # merge all instance masks into a single segmentation map
    #             # with its corresponding categories
    #             target, _ = (masks * cats[:, None, None]).max(dim=0)
    #             # discard overlapping instances
    #             # MODIFIED: in original code, overlapping instances are set to 255, then ignored in training by ignore_index=255
    #             # but, in this code, overlapping instances are set to 0, i.e. background
    #             target[masks.sum(0) > 1] = 0
    #         else:
    #             target = torch.zeros((h, w), dtype=torch.uint8)
    #         target = Image.fromarray(target.numpy())
    #         return image, target

    # class Compose:
    #     def __init__(self, transforms):
    #         self.transforms = transforms

    #     def __call__(self, image, target):
    #         for t in self.transforms:
    #             image, target = t(image, target)
    #         return image, target
        
    # def get_valid_coco(root, transforms):
    #     img_folder = os.path.join(root, "val2017")
    #     ann_file = os.path.join(root, "annotations_trainval2017/annotations/instances_val2017.json")

    #     transform = [FilterAndRemapCocoCategories(CAT_LIST, remap=True), ConvertCocoPolysToMask()]
    #     if transforms:
    #         transform.append(transforms)

    #     dataset = CocoDetection(img_folder, ann_file, transforms=Compose(transform))
    #     return dataset
    
    # # Adapted from torchvision/references/segmentation/train.py
    # from torchvision import models
    # from torchvision.transforms import functional as F
    # from torchvision.transforms import InterpolationMode

    # def get_pretrained_model_and_transform(model, weight_name):
    #     weight = models.get_weight(weight_name)
    #     model = models.get_model(model, weights=weight)
    #     transform = weight.transforms()

    #     def preprocessing(img, target):
    #         img = transform(img)
    #         size = F.get_dimensions(img)[1:]
    #         target = F.resize(target, size, interpolation=InterpolationMode.NEAREST)
    #         return img, F.pil_to_tensor(target)
        
    #     return model, preprocessing 
    
    # model, transform = get_pretrained_model_and_transform("fcn_resnet50", "FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1")
    # coco = get_valid_coco("/datasets/coco", transform)
    # model = model.eval().cuda()

    # from tqdm import tqdm 
    # y = []

    # c_hat = []
    # p_max = []
    # p_selfconf = []

    # for i, (image, target) in enumerate(tqdm(coco, total=len(coco))):
    #     image_gpu = image.to('cuda')

    #     with torch.no_grad():
    #         output = model(image_gpu[None])

    #     pred_mask = output['out'].cpu().softmax(dim=1).squeeze(0)

    #     pred_classes = pred_mask.argmax(dim=0)

    #     max_prob = pred_mask.gather(0, pred_classes.unsqueeze(0)).squeeze(0)

    #     target = target.long().squeeze(0)

    #     selfconf_prob = pred_mask.gather(0, target.unsqueeze(0)).squeeze(0)

    #     y.append(target)
    #     c_hat.append(pred_classes)
    #     p_max.append(max_prob)
    #     p_selfconf.append(selfconf_prob)

    # torch.save(y, "seg_y.pt")
    # torch.save(c_hat, "seg_pred_c.pt")
    # torch.save(p_max, "seg_p_max.pt")
    # torch.save(p_selfconf, "seg_p_selfconf.pt")

    # # Load the saved tensors, and use them 
    # import torch
    # y, c_hat, p_max, p_selfconf = torch.load("seg_y.pt"), torch.load("seg_pred_c.pt"), torch.load("seg_p_max.pt"), torch.load("seg_p_selfconf.pt")

    # to_numpy = lambda x: x.numpy()
    # y = list(map(to_numpy, y))
    # c_hat = list(map(to_numpy, c_hat))
    # p_max = list(map(to_numpy, p_max))
    # p_selfconf = list(map(to_numpy, p_selfconf))

    # seg = Segmentation(y, c_hat, p_max, p_selfconf, num_classes=len(CAT_LIST))

    # # Compare the results with cleanlab
    # from cleanlab.segmentation.rank import get_label_quality_scores
    # for i, (image, target) in enumerate(tqdm(coco, total=len(coco))):
    #     image_gpu = image.to('cuda')

    #     with torch.no_grad():
    #         output = model(image_gpu[None])

    #     pred_mask = output['out'].cpu().softmax(dim=1).squeeze(0)
    #     break

    # target = target.numpy()
    # pred_mask = pred_mask.unsqueeze(0).numpy()

    # image_score, pixel_scores = get_label_quality_scores(target, pred_mask)

    # scores = seg.get_result(pooling=True)
    # print(np.allclose(scores[0], image_score))

    # scores = seg.get_result(pooling=False)
    # print(np.allclose(scores[0], pixel_scores))

    # # for visualization, need to invert normalization
    # import torchvision.transforms as T
    # inv_normalize = T.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])

    # rank = 0 # 0 for the worst -1 for the best
    # ind = np.argsort(scores)[rank]

    # score = scores[ind]
    # image = inv_normalize(coco[ind][0])
    # target = coco[ind][1]
    # pred_mask = c_hat[ind]
    
    # catIds = coco.coco.getCatIds()
    # cats = coco.coco.loadCats(catIds)
    # id_to_cat = {cat['id']: cat['name'] for cat in cats}
    # id_to_cat[0] = 'background'

    # import numpy as np
    # import matplotlib.pyplot as plt
    # from matplotlib.patches import Patch

    # numpy_image = image.permute(1, 2, 0).cpu().numpy()
    # numpy_mask = target.cpu().squeeze().numpy()

    # num_labels = len(CAT_LIST)  
    # colors = plt.cm.get_cmap('tab20', num_labels) 

    # colored_mask = np.zeros((*numpy_mask.shape, 3), dtype=np.float32)

    # existing_indices = set()

    # for label in range(1, num_labels + 1):  
    #     mask = numpy_mask == label
    #     if np.any(mask):
    #         existing_indices.add(label)

    #     colored_mask[mask] = colors(label)[:3]  # Ignore alpha channel

    # alpha = 0.9
    # overlay_image = (numpy_image * (1 - alpha) + colored_mask * alpha).clip(0, 1)

    # # For prediction
    # colored_pred_mask = np.zeros((*numpy_mask.shape, 3), dtype=np.float32)

    # for label in range(1, num_labels + 1):  # Start from 1 to skip the background
    #     mask = pred_mask == label
    #     if np.any(mask):
    #         existing_indices.add(label)

    #     colored_pred_mask[mask] = colors(label)[:3]

    # alpha = 0.9
    # pred_overlay_image = (numpy_image * (1 - alpha) + colored_pred_mask * alpha).clip(0, 1)

    # # Create a list of patches for the legend
    # legend_patches = [Patch(color=colors(label)[:3], label=id_to_cat[CAT_LIST[label]]) for label in existing_indices]

    # # Plot
    # plt.figure(figsize=(10, 10))
    # plt.subplot(1, 3, 1)
    # plt.imshow(numpy_image)
    # plt.title('Orig Image')
    # plt.axis('off')

    # plt.subplot(1, 3, 2)
    # plt.imshow(overlay_image)
    # plt.title('GT')
    # plt.axis('off')

    # plt.subplot(1, 3, 3)
    # plt.imshow(pred_overlay_image)
    # plt.title('Pred')
    # plt.axis('off')

    # plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left')

    # plt.show()
    pass