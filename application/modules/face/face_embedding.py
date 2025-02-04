from typing import Dict, Tuple, Any, List

import cv2
import numpy as np

from modules.base import BaseModelTF, InputFormatException


class FaceEmbedding(BaseModelTF):
    def __init__(self, service_host: str, service_port: int, version: int = None,
                 image_size: tuple = (112, 112),
                 # keep_prob=0.4,
                 options: list = [], timeout: int = 5, debug: bool = False):
        super(FaceEmbedding, self).__init__(service_host, service_port, model_name='arcface', version=version,
                                            options=options, timeout=timeout, debug=debug)

        self.image_size = image_size
        self._debug = debug

    def _pre_process(self, x: List[np.ndarray]) -> Tuple[Dict[Any, np.ndarray], dict]:
        if x is None or not isinstance(x, List):
            raise InputFormatException("{}: {}".format(self.__class__.__name__, "input must be not None."))
        imgs = []
        for img in x:
            img = np.float32(img.copy())

            if img.shape[0] != self.image_size or img.shape[1] != self.image_size:
                img = cv2.resize(img, self.image_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(img)
        imgs = np.array(imgs)
        return {self._input_signature_key[0][0]: imgs}, {}

    def _post_process(self, predict_response: dict, params_process: dict) -> Dict:
        embeddings = predict_response["output"]
        embeddings = np.reshape(embeddings, (-1, 512))
        return {'embeddings': embeddings}

    def _normalize_output(self, post_processed: dict, params: dict):
        return post_processed['embeddings']
