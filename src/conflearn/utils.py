import numpy as np

def batch_iou(boxes1, boxes2):
    # Ensure boxes1 and boxes2 are 2D (N, 4)
    boxes1 = np.atleast_2d(boxes1)
    boxes2 = np.atleast_2d(boxes2)
    
    # Calculate intersection coordinates
    inter_x1 = np.maximum(boxes1[:, None, 0], boxes2[:, 0])
    inter_y1 = np.maximum(boxes1[:, None, 1], boxes2[:, 1])
    inter_x2 = np.minimum(boxes1[:, None, 2], boxes2[:, 2])
    inter_y2 = np.minimum(boxes1[:, None, 3], boxes2[:, 3])

    # Intersection area
    inter_area = np.maximum(inter_x2 - inter_x1, 0) * np.maximum(inter_y2 - inter_y1, 0)
    
    # Areas of both sets of boxes
    area_boxes1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area_boxes2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # Union area
    union_area = area_boxes1[:, None] + area_boxes2 - inter_area
    
    # Compute IoU
    iou = inter_area / union_area
    assert iou.shape == (boxes1.shape[0], boxes2.shape[0])    
    assert np.all(iou >= 0) and np.all(iou <= 1)    
    return iou

def gaussian_kernel_similarity(b1, b2, sigma=10):
    # Ensure inputs are 2D for broadcasting
    b1 = np.atleast_2d(b1)
    b2 = np.atleast_2d(b2)

    # Calculate the square of the L2 norm between two boxes in 2D space
    x1 = (b1[:, 0] + b1[:, 2]) / 2
    y1 = (b1[:, 1] + b1[:, 3]) / 2
    x2 = (b2[:, 0] + b2[:, 2]) / 2
    y2 = (b2[:, 1] + b2[:, 3]) / 2
    
    # [2, b1] -> [b1, 2], [2, b2] -> [b2, 2]
    c1, c2 = np.array([x1, y1]).T, np.array([x2, y2]).T
    # [b1, 2, 1] - [b2, 2] -> [b1, b2, 2] -> [b1, b2]
    dist = np.linalg.norm(c1[:, None] - c2, axis=2)
    # dist = np.sqrt((x1[:, None] - x2) ** 2 + (y1[:, None] - y2) ** 2)
    assert dist.shape == (b1.shape[0], b2.shape[0])
    return np.exp(-dist / sigma)
    
def similarity(boxes1, boxes2, alpha=0.1):
    iou_matrix = batch_iou(boxes1, boxes2)
    kernel_matrix = gaussian_kernel_similarity(boxes1, boxes2)
    similarity_matrix = alpha * kernel_matrix + (1 - alpha) * iou_matrix
    return similarity_matrix

def softmax(x, temperature=1.0):
    """
    Compute softmax values for each sets of scores in x.
    softmax(x/temperature)
    """
    # Use the maximum value to stabilize the computation (subtracting the maximum value does not change the result)
    # to avoid overflow in exp
    x = x / temperature
    e_x = np.exp(x - np.max(x, keepdims=True))
    return e_x / np.sum(e_x, keepdims=True)

def softmin(x, temperature=1.0):
    """
    Compute softmin values
    softmin(x/temperature)
    """
    if isinstance(x, list):
        x = np.array(x)
    return softmax(1-x, temperature)

def softmin1d_pooling(x, temperature=1.0):
    nan_mask = np.isnan(x)
    # If the input only contains np.nan, return 1.0
    if np.all(nan_mask):
        return 1.0
    
    def_values = np.array(x)[~nan_mask]
    return np.dot(def_values, softmax(1-def_values, temperature))