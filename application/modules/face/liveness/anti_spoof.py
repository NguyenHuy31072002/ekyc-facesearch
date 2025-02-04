import logging
from typing import Dict, Tuple, Any

import cv2
import numpy as np

from modules.base import BaseModelTF, InputFormatException


class FaceAntiSpoof(BaseModelTF):
    def __init__(self, service_host: str, service_port: int, model_name: str, version: int = None,
                 image_size: int = 260,
                 threshold: float = 0.63, options: list = [], timeout: int = 5, debug: bool = False):
        super(FaceAntiSpoof, self).__init__(service_host, service_port, model_name, version=version, options=options,
                                            timeout=timeout, debug=debug)
        self.image_size = image_size
        self.threshold = threshold

    def _pre_process(self, x: np.ndarray) -> Tuple[Dict[Any, np.ndarray], dict]:
        if x is None or not isinstance(x, np.ndarray):
            raise InputFormatException("{}: {}".format(self.__class__.__name__, "input must be not None."))
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = cv2.resize(x, (self.image_size, self.image_size))
        x = x / 255.0
        x = x.astype(np.float32)
        x = np.expand_dims(x, axis=0)

        return {
                   self._input_signature_key[0][0]: x
               }, {}

    def _post_process(self, predict_response: dict, params_process: dict) -> Dict:
        score = predict_response['predict']

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        print("fraud/real", sigmoid(score))
        fake_prob = float(score[0][0])
        real_prob = float(score[0][1])
        return {
            'fake_prob': fake_prob,
            'real_prob': real_prob,
            'is_fake': fake_prob > self.threshold
        }

    def _normalize_output(self, post_processed: dict, params: dict):
        return post_processed


class FaceAntiSpoofV2(BaseModelTF):
    def __init__(self, service_host: str, service_port: int, model_name: str, version: int = None,
                 image_size: int = 224,
                 threshold: float = 0.7, options: list = [], timeout: int = 5, debug: bool = False):
        super(FaceAntiSpoofV2, self).__init__(service_host, service_port, model_name, version=version, options=options,
                                            timeout=timeout, debug=debug)
        self.image_size = image_size
        self.threshold = threshold

    def _pre_process(self, x: np.ndarray) -> Tuple[Dict[Any, np.ndarray], dict]:
        if x is None or not isinstance(x, np.ndarray):
            raise InputFormatException("{}: {}".format(self.__class__.__name__, "input must be not None."))
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = cv2.resize(x, (self.image_size, self.image_size))
        x = np.expand_dims(x, axis=0)
        x = x.astype(np.float32)
        x = x / 255.0

        return {
                   self._input_signature_key[0][0]: x
               }, {}

    def _post_process(self, predict_response: dict, params_process: dict) -> Dict:
        score = predict_response['output']

        real_prob = float(score[0][0])
        fake_prob = float(score[0][1])
        return {
            'fake_prob': fake_prob,
            'real_prob': real_prob,
            'is_fake': fake_prob > self.threshold
        }

    def _normalize_output(self, post_processed: dict, params: dict):
        return post_processed


if __name__ == '__main__':
    model = FaceAntiSpoofV2('172.16.1.36', 8500, 'anti_spoof', threshold=0.7)

    import glob, os
    count_true = 0
    count_false = 0
    output_path = '/media/thiennt/projects/remote_lvt/ekyc-lvt/application/test/data/anti_spoof/result_wrong_5_5.csv'
    output_f = open(output_path, 'a')
    if not os.path.exists(output_path):
        output_f.write("filename,label,score\n")
    label = 'fake'
    for f in glob.glob('/media/thiennt/projects/remote_lvt/ekyc-lvt/application/test/data/anti_spoof/wrong_2_cropped_224/*'):
        filename = os.path.basename(f)
        img = cv2.imread(f)
        output = model.predict(img)
        if output['is_fake'] == True:
            count_true += 1
        else:
            count_false += 1
        line = f'{filename},{label},{output["fake_prob"]}'
        output_f.write(line+ "\n")
        print(line)

    print(f'fraud: {count_true}')
    print(f'real: {count_false}')
    output_f.close()
