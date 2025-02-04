import base64
import io
import re
import ssl
from typing import List, Dict

import cv2
import numpy as np
import requests
import requests.adapters
from PIL import Image
from cerberus import Validator
from numpy.linalg import norm
from urllib3 import poolmanager

from modules.face.face_analysis import Face
from modules.utils import dfloat32

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff',
                      'png"', 'jpg"', 'jpeg"', 'tiff"'}


def is_support_type(filename):
    """Check type support"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def url_to_image(url):
    """Download image from url"""
    session = requests.session()
    session.mount('https://', TLSAdapter())
    res = session.get(url, timeout=5)
    session.close()

    image = np.asarray(bytearray(res.content), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    return image


def base64_to_image(img_string):
    """Parser image from base64"""
    img_string = re.sub('^data:image/[a-z]+;base64,', '', img_string)
    imgdata = base64.b64decode(img_string)

    image = Image.open(io.BytesIO(imgdata))
    img = np.array(image)

    if img.ndim == 3 and img.shape[2] > 3:
        img = img[:, :, :3]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


class ImageException(Exception):
    pass


class TLSAdapter(requests.adapters.HTTPAdapter):

    def init_poolmanager(self, connections, maxsize, block=False):
        """Create and initialize the urllib3 PoolManager."""
        ctx = ssl.create_default_context()
        ctx.set_ciphers('DEFAULT@SECLEVEL=1')
        self.poolmanager = poolmanager.PoolManager(
            num_pools=connections,
            maxsize=maxsize,
            block=block,
            ssl_version=ssl.PROTOCOL_TLS,
            ssl_context=ctx)


def validate_input(schema, data):
    v = Validator(schema)
    if v.validate(data):
        return (True, None)
    else:
        return (False, v.errors)


def get_embedding_from_es(es_res: List[Dict]):
    if es_res and len(es_res) > 0:
        face = es_res[0]
        bytes = base64.b64decode(face['_source']['face'])
        return np.frombuffer(bytes, dtype=dfloat32)


def get_embedding_from_model(faces: List[Face]):
    if faces and len(faces) > 0:
        face = faces[0]
        embedding_norm = norm(face.embedding)
        normed_embedding = face.embedding / embedding_norm
        return normed_embedding


REGISTER_SCHEMA = {
    "image": {
        "type": "string",
        "required": True
    },
    "people_id": {
        "type": "string",
        "required": True
    },
    "created_at": {
        "type": "string",
        "required": True
    },
    "is_live_check": {
        "oneof": [
            {
                "allowed": [False],
            },

            {
                "allowed": [True],
                "dependencies": "liveness"
            },
        ],
        "type": "boolean",
        "required": False
    },
    "liveness": {
        "dependencies": {"is_live_check": True},
        "type": "list",
        "schema": {
            "type": "string",
            "required": False
        }}}
FACE_OPT_SCHEMA = {
    "people_1": {
        "type": "dict",
        "required": True,
        "schema": {
            "id": {
                "type": "string",
                "excludes": "image",
                "required": True
            },

            "image": {
                "type": "string",
                "excludes": "id",
                "required": True
            },
        }
    },
    "people_2": {
        "type": "string",
        "required": True
    },
    "is_live_check": {
        "oneof": [
            {
                "allowed": [False],
            },

            {
                "allowed": [True],
                "dependencies": "people_2_liveness"
            },
        ],
        "type": "boolean",
        "required": False
    },
    "people_2_liveness": {
        "dependencies": {"is_live_check": True},
        "type": "list",
        "schema": {
            "type": "string",
            "required": False
        }}}

CHECK_2_FACE_SCHEMA = {
    "people_1": {
        "type": "string",
        "required": True
    },
    "people_2": {
        "type": "string",
        "required": True
    },
    "is_live_check": {
        "oneof": [
            {
                "allowed": [False],
            },

            {
                "allowed": [True],
                "dependencies": "people_2_liveness"
            },
        ],
        "type": "boolean",
        "required": False
    },
    "people_2_liveness": {
        "dependencies": {"is_live_check": True},
        "type": "list",
        "schema": {
            "type": "string",
            "required": False
        }}}
