import cv2
import numpy as np
import torch
from config import system_configs
from img_utils import crop_image

def _rescale_points(dets, ratios, borders, sizes):
    """
    Rescale detection points based on given ratios, borders, and sizes.

    Parameters:
        dets (ndarray): Detection boxes, typically a 3D array with position information.
        ratios (ndarray): Rescaling ratios to adjust the size of the detection boxes.
        borders (ndarray): Border offsets to adjust the position of the detection boxes.
        sizes (ndarray): New size limits for the detection boxes.
    """
    xs, ys = dets[:, :, 2], dets[:, :, 3]
    xs /= ratios[0, 1]
    ys /= ratios[0, 0]
    xs -= borders[0, 2]
    ys -= borders[0, 0]
    np.clip(xs, 0, sizes[0, 1], out=xs)
    np.clip(ys, 0, sizes[0, 0], out=ys)

def kp_decode_detection(nnet, images):
    """
    Decode the detection from the neural network.

    Parameters:
        nnet: The neural network model used for testing.
        images: Input images, can be a batch of images.

    Returns:
        detections_tl (ndarray): Top-left corner detections.
        detections_br (ndarray): Bottom-right corner detections.
    """
    detections_tl_detection_br, *_ = nnet.test([images])
    detections_tl = detections_tl_detection_br[0].data.cpu().numpy().transpose((2, 1, 0))
    detections_br = detections_tl_detection_br[1].data.cpu().numpy().transpose((2, 1, 0))
    return detections_tl, detections_br

def test_kp_detection(image, db, nnet, decode_func=kp_decode_detection, cuda_id=0):
    """
    Test keypoint detection on a given image.

    Parameters:
        image (ndarray): Input image for detection.
        db: Database containing configurations.
        nnet: Neural network model for detection.
        decode_func: Function to decode the detection results.
        cuda_id (int): GPU ID for CUDA operations.

    Returns:
        top_points_tl (dict): Top-left keypoint detections categorized by class.
        top_points_br (dict): Bottom-right keypoint detections categorized by class.
    """
    categories = db.configs["categories"]
    max_per_image = db.configs["max_per_image"]
    height, width = image.shape[0:2]

    center = np.array([height // 2, width // 2])
    inp_height, inp_width = height | 127, width | 127
    images = np.zeros((1, 3, inp_height, inp_width), dtype=np.float32)
    ratios = np.zeros((1, 2), dtype=np.float32)
    borders = np.zeros((1, 4), dtype=np.float32)
    sizes = np.zeros((1, 2), dtype=np.float32)

    out_height, out_width = (inp_height + 1) // 4, (inp_width + 1) // 4
    height_ratio, width_ratio = out_height / inp_height, out_width / inp_width

    resized_image = cv2.resize(image, (width, height))
    resized_image, border, _ = crop_image(resized_image, center, [inp_height, inp_width])
    resized_image /= 255.0

    images[0] = resized_image.transpose((2, 0, 1))
    borders[0] = border
    sizes[0] = [height, width]
    ratios[0] = [height_ratio, width_ratio]

    images = torch.from_numpy(images).cuda(cuda_id) if torch.cuda.is_available() else torch.from_numpy(images)

    dets_tl, dets_br = decode_func(nnet, images)
    _rescale_points(dets_tl, ratios, borders, sizes)
    _rescale_points(dets_br, ratios, borders, sizes)

    detections_point_tl = np.concatenate([dets_tl], axis=1)
    detections_point_br = np.concatenate([dets_br], axis=1)

    classes_p_tl = detections_point_tl[:, 0, 1]
    classes_p_br = detections_point_br[:, 0, 1]

    keep_inds_p = (detections_point_tl[:, 0, 0] > 0)
    detections_point_tl = detections_point_tl[keep_inds_p, 0]
    classes_p_tl = classes_p_tl[keep_inds_p]

    keep_inds_p = (detections_point_br[:, 0, 0] > 0)
    detections_point_br = detections_point_br[keep_inds_p, 0]
    classes_p_br = classes_p_br[keep_inds_p]

    top_points_tl = {j: detections_point_tl[classes_p_tl == j].astype(np.float32) for j in range(categories)}
    top_points_br = {j: detections_point_br[classes_p_br == j].astype(np.float32) for j in range(categories)}

    # Filter top_points_tl based on scores
    scores = np.hstack([top_points_tl[j][:, 0] for j in range(categories)])
    if len(scores) > max_per_image:
        thresh = np.partition(scores, len(scores) - max_per_image)[len(scores) - max_per_image]
        for j in range(categories):
            top_points_tl[j] = top_points_tl[j][top_points_tl[j][:, 0] >= thresh]

    # Filter top_points_br based on scores
    scores = np.hstack([top_points_br[j][:, 0] for j in range(categories)])
    if len(scores) > max_per_image:
        thresh = np.partition(scores, len(scores) - max_per_image)[len(scores) - max_per_image]
        for j in range(categories):
            top_points_br[j] = top_points_br[j][top_points_br[j][:, 0] >= thresh]

    return top_points_tl, top_points_br

def kp_decode_grouping(nnet, images):
    """
    Decode group detections from the neural network.

    Parameters:
        nnet: The neural network model used for testing.
        images: Input images, can be a batch of images.

    Returns:
        detections_tl (ndarray): Top-left corner detections.
        detections_br (ndarray): Bottom-right corner detections.
        group_scores (ndarray): Group scores.
    """
    detections_tl, detections_br, group_scores = nnet.test([images])
    detections_tl = detections_tl.data.cpu().numpy().transpose((2, 1, 0))
    detections_br = detections_br.data.cpu().numpy().transpose((2, 1, 0))
    return detections_tl, detections_br, group_scores

def test_kp_grouping(image, db, nnet, decode_func=kp_decode_grouping, cuda_id=0):
    """
    Test keypoint grouping on a given image.

    Parameters:
        image (ndarray): Input image for grouping.
        db: Database containing configurations.
        nnet: Neural network model for grouping.
        decode_func: Function to decode the grouping results.
        cuda_id (int): GPU ID for CUDA operations.

    Returns:
        top_points_tl (dict): Top-left keypoint groupings categorized by class.
        top_points_br (dict): Bottom-right keypoint groupings categorized by class.
    """
    categories = db.configs["categories"]
    max_per_image = db.configs["max_per_image"]
    height, width = image.shape[0:2]

    center = np.array([height // 2, width // 2])
    inp_height, inp_width = height | 127, width | 127
    images = np.zeros((1, 3, inp_height, inp_width), dtype=np.float32)
    ratios = np.zeros((1, 2), dtype=np.float32)
    borders = np.zeros((1, 4), dtype=np.float32)
    sizes = np.zeros((1, 2), dtype=np.float32)

    out_height, out_width = (inp_height + 1) // 4, (inp_width + 1) // 4
    height_ratio, width_ratio = out_height / inp_height, out_width / inp_width

    resized_image = cv2.resize(image, (width, height))
    resized_image, border, _ = crop_image(resized_image, center, [inp_height, inp_width])
    resized_image /= 255.0

    images[0] = resized_image.transpose((2, 0, 1))
    borders[0] = border
    sizes[0] = [height, width]
    ratios[0] = [height_ratio, width_ratio]

    images = torch.from_numpy(images).cuda(cuda_id) if torch.cuda.is_available() else torch.from_numpy(images)

    dets_tl, dets_br, group_scores = decode_func(nnet, images)
    _rescale_points(dets_tl, ratios, borders, sizes)
    _rescale_points(dets_br, ratios, borders, sizes)

    detections_point_tl = np.concatenate([dets_tl], axis=1)
    detections_point_br = np.concatenate([dets_br], axis=1)

    classes_p_tl = detections_point_tl[:, 0, 1]
    classes_p_br = detections_point_br[:, 0, 1]

    keep_inds_p = (detections_point_tl[:, 0, 0] > 0)
    detections_point_tl = detections_point_tl[keep_inds_p, 0]
    classes_p_tl = classes_p_tl[keep_inds_p]

    keep_inds_p = (detections_point_br[:, 0, 0] > 0)
    detections_point_br = detections_point_br[keep_inds_p, 0]
    classes_p_br = classes_p_br[keep_inds_p]

    top_points_tl = {j: detections_point_tl[classes_p_tl == j].astype(np.float32) for j in range(categories)}
    top_points_br = {j: detections_point_br[classes_p_br == j].astype(np.float32) for j in range(categories)}

    # Filter top_points_tl based on scores
    scores = np.hstack([top_points_tl[j][:, 0] for j in range(categories)])
    if len(scores) > max_per_image:
        thresh = np.partition(scores, len(scores) - max_per_image)[len(scores) - max_per_image]
        for j in range(categories):
            top_points_tl[j] = top_points_tl[j][top_points_tl[j][:, 0] >= thresh]

    # Filter top_points_br based on scores
    scores = np.hstack([top_points_br[j][:, 0] for j in range(categories)])
    if len(scores) > max_per_image:
        thresh = np.partition(scores, len(scores) - max_per_image)[len(scores) - max_per_image]
        for j in range(categories):
            top_points_br[j] = top_points_br[j][top_points_br[j][:, 0] >= thresh]

    return top_points_tl, top_points_br
