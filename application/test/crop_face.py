import glob
import os.path

import cv2

from modules.face.face_detection import FastFaceDetection

face_detector = FastFaceDetection(service_host="172.16.1.36",
                                  service_port=8500,
                                  image_size=320,
                                  options=[],
                                  timeout=5,
                                  debug=False)


def crop_face(img):
    bboxes, landmarks, confidences, images = face_detector.predict(img,
                                                                   {"size_threshold": 0.0,
                                                                    "max_faces": 1,
                                                                    "receive_mode": 'image',
                                                                    "fast": True,
                                                                    "face_image_size": 224,
                                                                    'crop_mode': 'crop'})
    if bboxes is None or len(bboxes) == 0:
        return None

    return images[0]


def crop_and_save(input_path, output_path):
    img = cv2.imread(input_path)
    output_face = crop_face(img)
    if output_face is not None:
        cv2.imwrite(output_path, output_face)


def run(input_dir, output_dir):
    for file in glob.glob(input_dir + '/*'):
        filename = os.path.basename(file)

        output_path = os.path.join(output_dir, filename)
        crop_and_save(file, output_path)


if __name__ == '__main__':
    run('/media/thiennt/projects/remote_lvt/ekyc-lvt/application/test/data/anti_spoof/wrong_2', '/media/thiennt/projects/remote_lvt/ekyc-lvt/application/test/data/anti_spoof/wrong_2_cropped_224')