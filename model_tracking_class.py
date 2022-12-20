import copy
import os
import re
from heapq import nlargest

import numpy as np
import tensorflow as tf
import tifffile as tiff
from object_detection.builders import model_builder
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from tifffile import TiffFileError
import cv2 as cv

from model_template_class import ModelABC
from opencv_tracking import *


class TrackingModel(ModelABC):

    def __init__(self, model_path: str):
        super().__init__()
        self.category_index = None
        self.load(path=model_path)
        self.out = None

    def load(self, path: str) -> None:
        configs = config_util.get_configs_from_pipeline_file(os.path.join(path, 'pipeline.config'))
        model_config = configs['model']
        detection_model = model_builder.build(model_config=model_config, is_training=False)
        ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
        ckpt.restore(os.path.join(path, 'checkpoint', 'ckpt-0')).expect_partial()
        self._model = detection_model
        self.category_index = label_map_util.create_category_index_from_labelmap(os.path.join(path, 'label_map.pbtxt'),
                                                                                 use_display_name=False)

    @staticmethod
    def convert(img):
        tmp = []
        try:
            temp = tiff.imread(img)
            for img in temp:
                size = np.shape(img)
                if len(size) == 2:
                    img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
                tmp.append(img)
            return tmp
        except TiffFileError:
            print(img)

    @staticmethod
    def convertToTiff(array, out_image, postprocess='wp_', tracker_type='custom_'):
        array = np.array(array)
        tiff.imwrite(postprocess + tracker_type + out_image, array, photometric='rgb')
        return array

    @staticmethod
    def convertToPngJpg(array, out_image, postprocess='wp_', tracker_type='custom_'):
        array = np.array(array)
        cv.imwrite(postprocess + tracker_type + out_image, array, photometric='rgb')
        return array

    def preprocessing(self, path: str) -> list:
        data = []
        if len(re.findall('.jpg', path)) > 0:
            data = cv.imread(path)
        if len(re.findall('.png', path)) > 0:
            data = cv.imread(path)
        if len(re.findall('.jpeg', path)) > 0:
            data = cv.imread(path)
        if len(re.findall('.tif', path)) > 0:
            data = self.convert(path)
        return data

    @staticmethod
    def detect_fn(image, detection_model):
        """Detect objects in image."""
        image, shapes = detection_model.preprocess(image)
        prediction_dict = detection_model.predict(image, shapes)
        detections = detection_model.postprocess(prediction_dict, shapes)
        return detections

    @staticmethod
    def IoU(box1, box2):
        x1, y1, x2, y2 = box1[0]
        x3, y3, x4, y4 = box2
        if x1 > x4:
            return 0
        if x2 < x3:
            return 0
        if y1 > y4:
            return 0
        if y2 < y3:
            return 0
        x_inter1 = max(x1, x3)
        y_inter1 = max(y1, y3)
        x_inter2 = min(x2, x4)
        y_inter2 = min(y2, y4)
        width_inter = abs(x_inter2 - x_inter1)
        height_inter = abs(y_inter2 - y_inter1)
        area_inter = width_inter * height_inter
        width_box1 = abs(x2 - x1)
        height_box1 = abs(y2 - y1)
        width_box2 = abs(x4 - x3)
        height_box2 = abs(y4 - y3)
        area_box1 = width_box1 * height_box1
        area_box2 = width_box2 * height_box2
        area_union = area_box1 + area_box2 - area_inter
        value = area_inter / area_union
        if value > 1:
            value = 0
        return value

    def predict(self, path: str) -> object:
        result = []
        data = self.preprocessing(path)
        for image in data:
            input_tensor = tf.convert_to_tensor(np.expand_dims(image, 0), dtype=tf.float32)

            detections = self.detect_fn(input_tensor, self._model)

            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                          for key, value in detections.items()}
            detections['num_detections'] = num_detections
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
            image_np_with_detections = copy.deepcopy(image)
            viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes'],
                detections['detection_scores'],
                self.category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=200,
                min_score_thresh=.20,
                agnostic_mode=True)
            result.append([image_np_with_detections, detections])
        result1 = self.postProcess(result)
        postResult = []
        for stat, image in enumerate(data):
            image_np_with_detections1 = copy.deepcopy(image)
            try:
                viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_np_with_detections1,
                    result1[stat][1]['detection_boxes'],
                    result1[stat][1]['detection_classes'],
                    result1[stat][1]['detection_scores'],
                    self.category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=200,
                    min_score_thresh=.20,
                    agnostic_mode=True)
            except TypeError:
                continue
            postResult.append(image_np_with_detections1)
        if len(data) == 1:
            self.convertToPngJpg(postResult[0], out_image='result.png')
            self.out = postResult
            return self.out
        self.out = self.convertToTiff(postResult, out_image='result.tif')
        return self.out

    def postProcess(self, result):
        res = copy.deepcopy(result)
        box = 0, 0, 0, 0
        number = None
        # Поиск первого обнаружения узла
        for stat, image in enumerate(result):
            scores = image[1]['detection_scores']
            boxes = image[1]['detection_boxes']
            score = max(scores)
            if score >= 0.3:
                index = np.where(scores == score)
                box = boxes[index]
                number = stat
                res[stat][1]['detection_scores'] = np.array(score).reshape(1, )
                res[stat][1]['detection_boxes'] = box
                break
        stop_score = 0.1
        tracker_score = 0.4
        # Отсеивание лишних боксов
        if number is None:
            return None
        for i in range(number + 1, len(result)):
            iou = []
            boxes1 = result[i][1]['detection_boxes']
            scores1 = result[i][1]['detection_scores']
            max_score = max(scores1)
            if max_score < 0.2:
                flagScore = False
            else:
                flagScore = True
            box_max_score = boxes1[np.where(scores1 == max_score)]
            if not flagScore:
                res[i][1]['detection_boxes'] = box_max_score
                res[i][1]['detection_scores'] = np.array(stop_score).reshape(1, )
                continue
            for j in range(len(boxes1)):
                iou.append(self.IoU(box, boxes1[j]))
            iou_max = nlargest(5, iou)
            index_iou = []
            for k in range(len(iou_max)):
                if iou_max[k] < 0.5:
                    continue
                index_iou.append(np.where(iou == iou_max[k]))
            if len(index_iou) == 0:
                flagIOU = False
            else:
                flagIOU = True
            box1 = []
            max_score_iou = self.IoU(box, box_max_score[0])
            for l in index_iou:
                box1.append(boxes1[l])
            if len(boxes1) == 1:
                box1.append(boxes1)
            if max_score_iou >= 0.5:
                box1.append(box_max_score)
            box1 = np.array(box1)
            x1 = np.mean([a[0][0] for a in box1])
            y1 = np.mean([a[0][1] for a in box1])
            x2 = np.mean([a[0][2] for a in box1])
            y2 = np.mean([a[0][3] for a in box1])
            box = np.array([x1, y1, x2, y2]).reshape(1, 4)
            if max_score_iou < 0.5:
                res[i][1]['detection_boxes'] = box
            else:
                res[i][1]['detection_boxes'] = box_max_score
            score = []
            for u in index_iou:
                score.append(res[i][1]['detection_scores'][u])
            if max_score_iou >= 0.5:
                score.append(np.array(max_score).reshape(1, ))
            if max_score_iou < 0.5:
                res[i][1]['detection_scores'] = np.array(tracker_score).reshape(1, )
            else:
                res[i][1]['detection_scores'] = np.array(max_score).reshape(1, )
        return res


if __name__ == '__main__':
    tif_path = '25_25_TIRADS3_cross_16.jpg'
    tracking_model = TrackingModel(model_path='exported/full')
    tracking_model.predict(path=tif_path)
