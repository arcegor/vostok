import argparse
import copy
import os
import re
import pandas as pd
from numpy import maximum

from load import *
import numpy as np
import warnings
import cv2 as cv
import tifftojpg
import tifffile as tiff
from heapq import nlargest
from opencv_tracking import *

warnings.filterwarnings('ignore')  # Suppress Matplotlib warnings

parser = argparse.ArgumentParser(description='Great Description To Be Here')

parser.add_argument("-i",
                    "--images",
                    type=str)
parser.add_argument("-m",
                    "--model",
                    type=str)


def get_data(src):
    data = []
    if len(re.findall('.jpg', src)) > 0:
        data = cv.imread(src)
    if len(re.findall('.jpeg', src)) > 0:
        data = cv.imread(src)
    if len(re.findall('.tif', src)) > 0:
        data = tifftojpg.convert(src)
    return data


def convertToTiff(array, out_image, postprocess='wp_', tracker_type='custom_'):
    # arr = []
    # for stat, image in enumerate(array):
    #     arr.append(image[0])
    array = np.array(array)
    tiff.imwrite(postprocess + tracker_type + out_image, array, photometric='rgb')
    return array


def get_boxes(test_image_path="50_TIRADS5_cross.tif", model_path="exported/full"):
    name = test_image_path[5:]
    out_image = 'result_' + name
    detection_model = load(model_path)
    category_index = label_map_util.create_category_index_from_labelmap(model_path + '/' + "label_map.pbtxt",
                                                                        use_display_name=False)
    result = []
    if os.path.isdir(test_image_path):
        print('Укажите путь к файлу')
        return
    data = get_data(test_image_path)
    tracker_types = ['BOOSTING_', 'MIL_', 'KCF_', 'TLD_', 'MEDIANFLOW_', 'MOSSE_', 'CSRT_']
    for image in data:
        input_tensor = tf.convert_to_tensor(np.expand_dims(image, 0), dtype=tf.float32)

        detections = detect_fn(input_tensor, detection_model)

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
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.20,
            agnostic_mode=True)
        result.append([image_np_with_detections, detections])
    name = re.split('/', test_image_path)[-1]
    result2 = postProcess(result, name)
    result1 = result2[0]
    result2 = result2[1]
    postResult = []
    trackResult = []
    for stat, image in enumerate(data):
        image_np_with_detections1 = copy.deepcopy(image)
        try:
            viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections1,
                result1[stat][1]['detection_boxes'],
                result1[stat][1]['detection_classes'],
                result1[stat][1]['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=200,
                min_score_thresh=.20,
                agnostic_mode=True)
        except TypeError:
            continue
        postResult.append(image_np_with_detections1)

    # trackResults
    for num, tracker_type in enumerate(tracker_types):
        trackResult1 = []
        for stat, image in enumerate(data):
            image_np_with_detections1 = image.copy()
            try:
                viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_np_with_detections1,
                    result2[num][stat][1]['detection_boxes'],
                    result2[num][stat][1]['detection_classes'],
                    result2[num][stat][1]['detection_scores'],
                    category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=200,
                    min_score_thresh=.20,
                    agnostic_mode=True)
            except TypeError:
                continue
            trackResult1.append(image_np_with_detections1)
        trackResult.append(trackResult1)

    if len(data) == 1:
        cv.imwrite('result.jpg', postResult[0])
        return postResult
    postResult = convertToTiff(postResult, out_image=out_image)
    result = [obj[0] for obj in result]
    for count, item in enumerate(trackResult):
        trackResult2 = convertToTiff(item, out_image=out_image, tracker_type=tracker_types[count])
    result = convertToTiff(result, out_image=out_image, postprocess='np_')
    return postResult


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


def postProcess(result, name):
    res = copy.deepcopy(result)

    iou_test = {
        'BOOSTING': [],
        'MIL': [],
        'KCF': [],
        'TLD': [],
        'MEDIANFLOW': [],
        'MOSSE': [],
        'CSRT': [],
        'custom': []
    }
    trackers = {
        'BOOSTING': cv.legacy.TrackerBoosting_create(),
        'MIL': cv.legacy.TrackerMIL_create(),
        'KCF': cv.legacy.TrackerKCF_create(),
        'TLD': cv.legacy.TrackerTLD_create(),
        'MEDIANFLOW': cv.legacy.TrackerMedianFlow_create(),
        'MOSSE': cv.legacy.TrackerMOSSE_create(),
        'CSRT': cv.TrackerCSRT_create()
    }
    end = len(result)
    masks = get_masks(name, end)

    results = [copy.deepcopy(result) for a in range(len(trackers))]

    box = 0, 0, 0, 0
    number = None
    # Поиск первого обнаружения узла
    for stat, image in enumerate(result):
        scores = image[1]['detection_scores']
        boxes = image[1]['detection_boxes']
        score = max(scores)
        index = np.where(scores == score)
        box_all = boxes[index]
        for test in iou_test.keys():
            iou_test[test].append(IoU(box_all, masks[stat]))
        if score >= 0.3:
            index = np.where(scores == score)
            box = boxes[index]
            number = stat
            res[stat][1]['detection_scores'] = np.array(score).reshape(1, )
            res[stat][1]['detection_boxes'] = box

            x1, y1, x2, y2 = unnormilize(box, image[0])
            for count, tracker in enumerate(trackers.values()):
                ok = tracker.init(image[0], (x1, y1, x2 - x1, y2 - y1))
                results[count][stat][1]['detection_scores'] = np.array(score).reshape(1, )
                results[count][stat][1]['detection_boxes'] = box
            break
        res[stat][1]['detection_scores'] = np.zeros([len(boxes), ])
    stop_score = 0.1
    tracker_score = 0.4
    # Отсеивание лишних боксов
    if number is None:
        return None
    for i in range(number + 1, len(result)):
        iou = []
        bboxes = []

        # tracker
        for tracker in trackers.keys():
            ok, bbox = trackers[tracker].update(res[i][0])
            bbox = normilize(bbox, res[i][0])
            bboxes.append(bbox)
            iou_test[tracker].append(IoU(bbox, masks[i]))

        for num, track_res in enumerate(results):
            track_res[i][1]['detection_boxes'] = bboxes[num]
            track_res[i][1]['detection_scores'] = np.array(tracker_score).reshape(1, )

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

        # if max_score < 0.3:
        #     # res[i][1]['detection_boxes'] = bbox
        #     # res[i][1]['detection_scores'] = np.array(stop_score).reshape(1, )
        #     # box = bbox
        #     continue

        for j in range(len(boxes1)):
            iou.append(IoU(box, boxes1[j]))

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
        max_score_iou = IoU(box, box_max_score[0])

        for l in index_iou:
            box1.append(boxes1[l])
        if len(boxes1) == 1:
            box1.append(boxes1)
        if max_score_iou >= 0.5:
            box1.append(box_max_score)
        box1 = np.array(box1)
        try:
            x1 = np.mean([a[0][0] for a in box1])
            y1 = np.mean([a[0][1] for a in box1])
            x2 = np.mean([a[0][2] for a in box1])
            y2 = np.mean([a[0][3] for a in box1])
            box = np.array([x1, y1, x2, y2]).reshape(1, 4)
        except BaseException:
            print(name)

        if max_score_iou < 0.5:
            res[i][1]['detection_boxes'] = box
            iou_test['custom'].append(IoU(box, masks[i]))
        else:
            res[i][1]['detection_boxes'] = box_max_score
            iou_test['custom'].append(IoU(box_max_score, masks[i]))
        score = []
        for u in index_iou:
            score.append(res[i][1]['detection_scores'][u])
        if max_score_iou >= 0.5:
            score.append(np.array(max_score).reshape(1, ))
        if max_score_iou < 0.5:
            res[i][1]['detection_scores'] = np.array(tracker_score).reshape(1, )
        else:
            res[i][1]['detection_scores'] = np.array(max_score).reshape(1, )
    print(iou_test)
    for key, value in iou_test.items():
        length = len(value)
        tmp = 0
        for item in value:
            if item >= 0.4:
                tmp += 1
        map = tmp / length
        print(key, ': ', np.mean(value), 'map = ', map)

    return res, results


def get_masks(img, end):
    masks = []
    mask = re.sub('cross', 'cross_mask', img)
    mask = re.sub('long', 'long_mask', mask)[:-3]
    mask = mask[5:]
    df = pd.read_csv('full.csv')
    arr = df.values.tolist()
    mn = 1000
    mx = -1
    rr = []
    for item in arr:
        try:
            rr = re.findall(mask, item[1])
        except BaseException:
            print(mask, item)
        if len(rr) > 0:
            temp = int(re.sub('.jpg', '', re.split('_', re.split('/', item[0])[-1])[-1]))
            if temp >= mx:
                mx = temp
            if temp <= mn:
                mn = temp
            x1, y1, x2, y2 = item[2], item[4], item[3], item[5]
            x1, y1, x2, y2 = normilize1(x1, y1, x2, y2, item[7], item[6])
            masks.append((x1, y1, x2, y2))
    for i in range(mn):
        masks.insert(0, (0, 0, 0, 0))
    for j in range(mx, end):
        masks.insert(0, (0, 0, 0, 0))
    return masks


def unnormilize(box, image):
    x_real = np.shape(image)[0]
    y_real = np.shape(image)[1]
    x1, y1, x2, y2 = box[0][0] * x_real, box[0][1] * y_real, box[0][2] * x_real, box[0][3] * y_real
    x1, y1, x2, y2 = np.round(x1).astype(int), np.round(y1).astype(int), np.round(x2).astype(int), np.round(y2).astype(
        int)
    return x1, y1, x2, y2


def normilize(box, image):
    x_real = np.shape(image)[0]
    y_real = np.shape(image)[1]
    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = x1 / x_real, y1 / y_real, (x1 + x2) / x_real, (y1 + y2) / y_real
    box = np.array([x1, y1, x2, y2]).reshape(1, 4)
    return box


def normilize1(x1, y1, x2, y2, x_real, y_real):
    x1, y1, x2, y2 = x1 / x_real, y1 / y_real, x2 / x_real, y2 / y_real
    return x1, y1, x2, y2


# def mAp(dict1):
#     for key, value in dict1.items():
#         length = len(value)
#         tmp = np.arange(0.3, 0.95, 0.05)
#         for item in value:
#             for param in tmp:
#
#
#             if item >= 0.4:
#                tmp += 1
#         map = tmp / length
#         print(key, ': ', np.mean(value), 'map = ', map)


def main(_):
    return get_boxes()


def process_group(path):
    test = os.listdir(path)
    for img in test:
        get_boxes(os.path.join(path, img))


if __name__ == '__main__':
    args = ('results/58_TIRADS2_long.tif', 'exported/full')
    # main(args)
    process_group('test')
