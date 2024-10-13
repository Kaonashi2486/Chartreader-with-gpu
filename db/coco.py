import os
import json
import numpy as np
import pickle
import copy
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from detection import DETECTION
# from config import system_configs
import sys
# Add the path to your config.py to sys.path
sys.path.append(r'C:\Users\saksh\OneDrive\Desktop\stuffs\Chartreader-with-gpu')
# Now you can import system_configs from config.py
from config import system_configs

# Use system_configs as needed
# # Import (deserialize) the instance from the file
# with open('config_instance.pkl', 'rb') as file:
#     system_configs = pickle.load(file)
# # Now you can use the loaded instance
# print(system_configs)


class Chart(DETECTION):
    def __init__(self, db_config, split):
        super(Chart, self).__init__(db_config)
        data_dir = system_configs.data_dir
        cache_dir = system_configs.cache_dir

        self._split = split
        self._dataset = {
            "trainchart": "train",
            "valchart": "val",
            "testchart": "test"
        }[self._split]

        is_inference = False
        self._coco_dir = data_dir

        # Windows-compatible path
        self._label_dir = os.path.join(self._coco_dir, "annotations")
        self._label_file = os.path.join(self._label_dir, f"{self._dataset}.json")

        if not os.path.exists(self._label_file):
            self._label_file = None
            is_inference = True
        print(f"Label file: {self._label_file}")

        self._image_dir = os.path.join(self._coco_dir, "images", self._dataset)
        self._image_file = os.path.join(self._image_dir, "{}")

        # Normalized image values
        self._mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32)
        self._std = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)
        
        self._cat_ids = [0, 1, 2]
        self._classes = {ind: cat_id for ind, cat_id in enumerate(self._cat_ids)}
        self._coco_to_class_map = {value: key for key, value in self._classes.items()}

        # Windows-compatible cache file path
        self._cache_file = os.path.join(cache_dir, f"{self._dataset}_cache.pkl")
        
        if not is_inference:
            self._load_data()
            self._db_inds = np.arange(len(self._image_ids))
            self._load_coco_data()

    def _load_data(self):
        if not os.path.exists("./cache"):
            os.makedirs("./cache")

        print(f"Loading from cache file: {self._cache_file}")
        if not os.path.exists(self._cache_file):
            print("No cache file found...")
            self._extract_data()
            with open(self._cache_file, "wb") as f:
                pickle.dump([self._detections, self._image_ids], f)
        else:
            with open(self._cache_file, "rb") as f:
                self._detections, self._image_ids = pickle.load(f)

    def _load_coco_data(self):
        self._coco = COCO(self._label_file)
        with open(self._label_file, "r") as f:
            data = json.load(f)

        coco_ids = self._coco.getImgIds()
        eval_ids = {
            self._coco.loadImgs(coco_id)[0]["file_name"]: coco_id
            for coco_id in coco_ids
        }

        self._coco_categories = data["categories"]
        self._coco_eval_ids = eval_ids

    def class_name(self, cid):
        cat_id = self._classes[cid]
        cat = self._coco.loadCats([cat_id])[0]
        return cat["name"]

    def _extract_data(self):
        self._coco = COCO(self._label_file)
        self._cat_ids = self._coco.getCatIds()
        coco_image_ids = self._coco.getImgIds()
        self._image_ids = [
            self._coco.loadImgs(img_id)[0]["file_name"]
            for img_id in coco_image_ids
        ]
        self._detections = {}
        for ind, (coco_image_id, image_id) in enumerate(tqdm(zip(coco_image_ids, self._image_ids))):
            image = self._coco.loadImgs(coco_image_id)[0]
            bboxes = []
            categories = []
            max_len = 0
            for cat_id in self._cat_ids:
                category = self._coco_to_class_map[cat_id]
                annotation_ids = self._coco.getAnnIds(imgIds=image["id"], catIds=cat_id)
                annotations = self._coco.loadAnns(annotation_ids)
                if len(annotations) == 0:
                    continue
                if category == 0:
                    # Bar chart
                    for annotation in annotations:
                        bbox = np.array(annotation["bbox"])
                        bbox[[2, 3]] += bbox[[0, 1]]
                        bboxes.append(bbox)
                        categories.append(category)
                elif category == 1:
                    # Line chart
                    for annotation in annotations:
                        bbox = np.array(annotation["bbox"])
                        bboxes.append(bbox)
                        categories.append(category)
                        max_len = max(max_len, len(bbox))
                else:
                    # Pie chart
                    for annotation in annotations:
                        bbox = np.array(annotation["bbox"])
                        bboxes.append(bbox)
                        categories.append(category)
            if max_len:
                for ind_bbox in range(len(bboxes)):
                    if len(bboxes[ind_bbox]) < max_len:
                        bboxes[ind_bbox] = np.pad(bboxes[ind_bbox], (0, max_len - len(bboxes[ind_bbox])), 'constant', constant_values=(0, 0))
            categories = np.array(categories, dtype=float)
            bboxes = np.array(bboxes, dtype=float)
            if bboxes.size == 0 or categories.size == 0:
                self._detections[image_id] = np.zeros((0, 5), dtype=np.float32)
            else:
                self._detections[image_id] = (np.hstack((bboxes, categories[:, None])), categories)

    def detections(self, ind):
        image_id = self._image_ids[ind]
        return copy.deepcopy(self._detections[image_id])

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_to_coco(self, all_bboxes):
        detections = []
        for image_id in all_bboxes:
            coco_id = self._coco_eval_ids[image_id]
            for cls_ind in all_bboxes[image_id]:
                category_id = self._classes[cls_ind]
                for bbox in all_bboxes[image_id][cls_ind]:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                    score = bbox[4]
                    bbox = list(map(self._to_float, bbox[0:4]))
                    detection = {
                        "image_id": coco_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "score": float("{:.2f}".format(score))
                    }
                    detections.append(detection)
        return detections

    def evaluate(self, result_json, cls_ids, image_ids, gt_json=None):
        if self._split == "testdev":
            return None
        coco = self._coco if gt_json is None else COCO(gt_json)
        eval_ids = [self._coco_eval_ids[image_id] for image_id in image_ids]
        cat_ids = [self._classes[cls_id] for cls_id in cls_ids]
        coco_dets = coco.loadRes(result_json)
        coco_eval = COCOeval(coco, coco_dets, "bbox")
        coco_eval.params.imgIds = eval_ids
        coco_eval.params.catIds = cat_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return coco_eval.stats[0], coco_eval.stats[12:]
