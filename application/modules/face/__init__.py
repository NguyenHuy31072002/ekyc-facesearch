import logging
from typing import List

import numpy as np

from config_run import Config
from modules import utils
from modules.face.face_analysis import FaceAnalysis
from modules.face.liveness import LivenessChecking, Command
from modules.face.liveness.anti_spoof import FaceAntiSpoofV2
from modules.utils import compute_sim

LOGGER = logging.getLogger("model")


class NotEnoughFaceCompare(Exception):
    pass


class NoFaceDetection(Exception):
    pass


face_analysis: FaceAnalysis = None

face_liveness: LivenessChecking = None

face_anti_spoof: FaceAntiSpoofV2 = None


def init_module(config):
    global face_analysis, face_liveness, face_anti_spoof
    if face_analysis is None:
        face_analysis = FaceAnalysis(service_host=config.service_host,
                                     service_port=config.service_port,
                                     timeout=config.service_timeout,
                                     debug=config.DEBUG)
    if face_liveness is None:
        face_liveness = LivenessChecking(service_host=config.service_host,
                                         service_port=config.service_port,
                                         thresh_config=config,
                                         timeout=config.service_timeout,
                                         debug=config.DEBUG)
    if face_anti_spoof is None:
        face_anti_spoof = FaceAntiSpoofV2(service_host=config.service_host,
                                          service_port=config.service_port,
                                          model_name='anti_spoof',
                                          version=None,
                                          threshold=config.face_anti_spoof_threshold,
                                          timeout=config.service_timeout,
                                          debug=config.DEBUG)


def detect_face(img, size_threshold=0.0, max_faces=10, receive_mode="meta", fast=True):
    """
    Detect face in image input. User size threshold to filter small face.
    :param img: numpy array image
    :param size_threshold: area threshold has range from 0 - 1. 0 is not filter and 1 is filter all.
    :param max_faces: Maximum of faces has response
    :param receive_mode: type of response 'meta': bounding boxes and landmarks;     'images': image of faces
    :param fast: boolean - mode inference
    :return:
    """

    return face_analysis.get(img,
                             size_threshold=size_threshold,
                             max_faces=max_faces,
                             receive_mode=receive_mode,
                             fast=fast,
                             mode="detect")


def embed_face(img, size_threshold=0.0, max_faces=10, fast=False):
    """
    Detect face in image input. User size threshold to filter small face.
    :param img: numpy array image
    :param size_threshold: area threshold has range from 0 - 1. 0 is not filter and 1 is filter all.
    :param max_faces: Maximum of faces has response
    :param fast: boolean - mode inference
    :return:
    """
    faces = face_analysis.get(img,
                              size_threshold=size_threshold,
                              max_faces=max_faces,
                              receive_mode="image",
                              fast=fast,
                              mode="embed")
    if len(faces) == 0:
        raise NoFaceDetection('Can\'t detect face. Please update another picture')
    return faces


def embed_face_only(images: List[np.ndarray], size_threshold=0.0, max_faces=10, fast=False) -> np.ndarray:
    """
        Get embedding of list face
        :param images: list numpy array image
        :param size_threshold: area threshold has range from 0 - 1. 0 is not filter and 1 is filter all.
        :param max_faces: Maximum of faces has response
        :param fast: boolean - mode inference
        :return: List embedding
        """
    embeddings = face_analysis.get(images, mode="embed_only")
    return embeddings


def compare_face(img1: np.ndarray, img2s: List[np.ndarray], threshold: float = Config.similar_thresh_compare):
    """
    Detect and embed face in each image and compare them.
    :raise NotEnoughFaceCompare if image don't have faces
    :param img1: Input 1st image
    :param img2: Input 2nd image
    :return:
    """
    face1 = face_analysis.get(img1,
                              size_threshold=0.0,
                              max_faces=1,
                              receive_mode="image",
                              fast=True,
                              mode="embed")
    num_face1 = len(face1)
    if num_face1 < 1:
        raise NoFaceDetection('Can\'t detect face. Please update another picture')
    face1 = face1[0]

    face2s = []
    for img2 in img2s:
        face2 = face_analysis.get(img2,
                                  size_threshold=0.0,
                                  max_faces=1,
                                  receive_mode="image",
                                  fast=True,
                                  mode="embed")
        num_face2 = len(face2)
        if num_face2 < 1:
            raise NoFaceDetection('Can\'t detect face. Please update another picture')
        face2s.append(face2[0])

    matches = []
    similars = []
    for j in range(len(face2s)):
        embedding1 = face1.embedding
        embedding2 = face2s[j].embedding
        sim = compute_sim(embedding1, embedding2, new_range=True)
        match = sim >= threshold
        similars.append(sim)
        matches.append(match)

    return matches, similars


def check_liveness(matches: List[bool], sims: List[float]) -> bool:
    if Config.min_liveness > 0:
        return sum(matches) >= Config.min_liveness
    else:
        return sum(matches) / len(matches) >= Config.thresh_liveness


def get_similars(emb1, emb2s, threshold: float = Config.similar_thresh_liveness):
    matches = []
    similars = []
    for j in range(len(emb2s)):
        sim = compute_sim(emb1, emb2s[j], new_range=True)
        match = sim >= threshold
        similars.append(sim)
        matches.append(match)

    return matches, similars


def check_action(imgs: List[np.ndarray], cmd: Command) -> bool:
    return face_liveness.check_action(imgs, cmd)


def is_fake(img: np.ndarray) -> bool:
    faces = face_analysis.get(img,
                              size_threshold=0.0,
                              max_faces=1,
                              receive_mode='image',
                              fast=True,
                              mode="detect",
                              face_image_size=Config.face_anti_spoof_image_size,
                              crop_mode='crop')
    if faces is not None and len(faces) > 0:
        return face_anti_spoof.predict(faces[0].image)
    else:
        raise NoFaceDetection("Not found face to check")
