import logging
from typing import Dict, Tuple, Any

import cv2
import numpy as np
from shapely.geometry import Point, Polygon

from modules.base import BaseModelTF, InputFormatException
from modules.face import face_align

LOGGER = logging.getLogger("model")
LOGGER.setLevel(logging.INFO)


class FaceDetection(BaseModelTF):
    def __init__(self, service_host: str, service_port: int, model_name: str, version: int = None,
                 image_size: int = 320, conf_thresh: float = 0.92,
                 options: list = None, timeout: int = 5, debug: bool = False):
        super(FaceDetection, self).__init__(service_host, service_port, model_name, version, options, timeout, debug)
        self.image_size = image_size
        self.conf_thresh = conf_thresh
        self.down_scale_factor = 1.0

    def _pre_process(self, x: np.ndarray) -> Tuple[Dict[Any, np.ndarray], dict]:
        if x is None or not isinstance(x, np.ndarray):
            raise InputFormatException("{}: {}".format(self.__class__.__name__, "input must be not None."))
        img_height_raw, img_width_raw, _ = x.shape
        img = np.float32(x.copy())

        width_scale_factor = self.image_size / img_width_raw
        height_scale_factor = self.image_size / img_height_raw
        down_scale_factor = min(width_scale_factor, height_scale_factor, 1)
        # LOGGER.info(f"Scale factor: width={width_scale_factor}  height={height_scale_factor}  down={down_scale_factor}")
        img = cv2.resize(img, (0, 0), fx=down_scale_factor, fy=down_scale_factor)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img, pad_params = FaceDetection.pad_input_image(img, max_steps=32)
        img = np.expand_dims(img, axis=0)
        return {
                   self._input_signature_key[0][0]: img
               }, {
                   'image': x,
                   'pad_params': pad_params,
                   'img_width_raw': img_width_raw,
                   'img_height_raw': img_height_raw,
                   'width_scale_factor': width_scale_factor,
                   'height_scale_factor': height_scale_factor
               }

    def _post_process(self, predict_response: dict, params_process: dict) -> Dict:
        pad_params = params_process['pad_params']
        img_width = params_process['img_width_raw']
        img_height = params_process['img_height_raw']
        size_threshold = 0.0
        if 'size_threshold' in params_process:
            size_threshold = params_process['size_threshold']
        max_faces = 10
        if 'max_faces' in params_process:
            max_faces = params_process['max_faces']
        receive_mode = 'meta'
        if 'receive_mode' in params_process:
            receive_mode = params_process['receive_mode']
        img = params_process['image']

        outputs = predict_response["tf_op_layer_GatherV2"]
        outputs = np.reshape(outputs, (-1, 16))
        outputs = FaceDetection.recover_pad_output(outputs, pad_params)
        outputs[:, 0] = outputs[:, 0] * img_width  # x1
        outputs[:, 1] = outputs[:, 1] * img_height  # y1
        outputs[:, 2] = outputs[:, 2] * img_width  # x2
        outputs[:, 3] = outputs[:, 3] * img_height  # y2

        outputs[:, 4] = outputs[:, 4] * img_width  # x2
        outputs[:, 5] = outputs[:, 5] * img_height  # y2

        outputs[:, 6] = outputs[:, 6] * img_width  # x2
        outputs[:, 7] = outputs[:, 7] * img_height  # y2

        outputs[:, 8] = outputs[:, 8] * img_width  # x2
        outputs[:, 9] = outputs[:, 9] * img_height  # y2

        outputs[:, 10] = outputs[:, 10] * img_width  # x2
        outputs[:, 11] = outputs[:, 11] * img_height  # y2

        outputs[:, 12] = outputs[:, 12] * img_width  # x2
        outputs[:, 13] = outputs[:, 13] * img_height  # y2

        bboxes = outputs[:, 0:4].astype(int)
        landmarks = outputs[:, 4:14].astype(int)
        valid_landmarks = outputs[:, 14]
        confidences = outputs[:, 15]

        # filter result
        if bboxes.shape[0] == 0:
            return {
                'bboxes': None,
                'landmarks': None,
                'confidences': None,
                'faces': None
            }  # bboxes, landmarks, confidences, img_face

        # filter with confidence threshold
        if 0 < self.conf_thresh < 1:
            mask = np.where(confidences >= self.conf_thresh)
            bboxes = bboxes[mask]
            landmarks = landmarks[mask]

        # filter with size threshold
        if 0 < size_threshold < 1:
            area_threshold = size_threshold * (img.shape[0] * img.shape[1])
            # Calculate area of faces
            area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])

            mask = np.where(area >= area_threshold)
            bboxes = bboxes[mask]
            landmarks = landmarks[mask]

        # Get top biggest and near center of image
        if 0 < max_faces < bboxes.shape[0]:
            # Calculate area of faces
            area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
            img_center = img.shape[0] // 2, img.shape[1] // 2

            # Get offset of center face with center of image
            offsets = np.vstack(
                [(bboxes[:, 0] + bboxes[:, 2]) / 2 - img_center[1], (bboxes[:, 1] + bboxes[:, 3]) / 2 - img_center[0]]
            )

            # Calculate distance square of offset: x^2 + y^2
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            values = area - offset_dist_squared * 2.0  # some extra weight on the centering
            bindex = np.argsort(values)[::-1]
            bindex = bindex[0:max_faces]
            bboxes = bboxes[bindex, :]
            landmarks = landmarks[bindex, :]

        mask = []
        for i, landmark in enumerate(landmarks):
            point = Point(landmark[4], landmark[5])
            polygon = Polygon([landmark[0:2], landmark[2:4], landmark[8:10], landmark[6:8]])
            if polygon.contains(point):
                mask.append(i)
            # if self._debug:
                # for i in range(5):
                #     img = cv2.circle(img, (landmark[2 * i], landmark[2 * i + 1]), 1, (255, 255, 0), 1)
                #     img = cv2.putText(img, str(i), (landmark[2 * i], landmark[2 * i + 1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
                # cv2.imshow("test", img)
            # cv2.waitKey(0)
        landmarks = landmarks[mask]
        bboxes = bboxes[mask]
        confidences = confidences[mask]

        # return via receive mode
        if receive_mode == 'meta':
            return {
                'bboxes': bboxes.tolist(),
                'landmarks': landmarks.tolist(),
                'confidences': confidences.tolist(),
                'faces': [None] * len(bboxes)
            }
        elif receive_mode == 'image':
            # aligned face
            faces = []
            landmarks_ = landmarks.reshape((-1, 5, 2))
            for landmark in landmarks_:
                image_size = 112
                crop_mode = 'arcface'
                if 'crop_mode' in params_process:
                    crop_mode = params_process['crop_mode']
                if crop_mode != 'arcface':
                    image_size = params_process['face_image_size']
                face = face_align.norm_crop(img, landmark, image_size=image_size, mode=crop_mode)
                faces.append(face)
            return {
                'bboxes': bboxes.tolist(),
                'landmarks': landmarks.tolist(),
                'confidences': confidences.tolist(),
                'faces': faces
            }

    def _normalize_output(self, post_processed: dict, params: dict):
        return post_processed['bboxes'], post_processed['landmarks'], post_processed['confidences'], post_processed[
            'faces']

    @staticmethod
    def pad_input_image(img, max_steps=32):
        """pad image to suitable shape"""
        img_h, img_w, _ = img.shape

        img_pad_h = 0
        if img_h % max_steps > 0:
            img_pad_h = max_steps - img_h % max_steps

        img_pad_w = 0
        if img_w % max_steps > 0:
            img_pad_w = max_steps - img_w % max_steps

        padd_val = np.mean(img, axis=(0, 1)).astype(np.uint8)
        img = cv2.copyMakeBorder(img, 0, img_pad_h, 0, img_pad_w,
                                 cv2.BORDER_CONSTANT, value=padd_val.tolist())
        pad_params = (img_h, img_w, img_pad_h, img_pad_w)

        return img, pad_params

    @staticmethod
    def recover_pad_output(outputs, pad_params):
        """recover the padded output effect"""
        img_h, img_w, img_pad_h, img_pad_w = pad_params
        recover_xy = np.reshape(outputs[:, :14], [-1, 7, 2]) * \
                     [(img_pad_w + img_w) / img_w, (img_pad_h + img_h) / img_h]
        outputs[:, :14] = np.reshape(recover_xy, [-1, 14])

        return outputs


class AccuracyFaceDetection(FaceDetection):
    def __init__(self, service_host: str, service_port: int, version: int = None,
                 image_size: int = 320, conf_thresh: float = 0.92,
                 options: list = None, timeout: int = 5, debug: bool = False):
        super(AccuracyFaceDetection, self).__init__(service_host, service_port, 'retinaface_res50', version, image_size,
                                                    conf_thresh, options, timeout, debug)


class FastFaceDetection(FaceDetection):
    def __init__(self, service_host: str, service_port: int, version: int = None,
                 image_size: int = 320, conf_thresh: float = 0.92,
                 options: list = None, timeout: int = 5, debug: bool = False):
        super(FastFaceDetection, self).__init__(service_host, service_port, 'retinaface_mbnet', version, image_size,
                                                conf_thresh, options, timeout, debug)
