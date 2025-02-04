from datetime import datetime
import os
import cv2
import base64
import json
import numpy as np

import sys

sys.path.insert(0, "../modules/inference")

from app.utils import get_embedding_from_model
from config_run import Config, init_config
from modules.driver import mongo_driver, elasticsearch_driver
from modules.face import init_module, embed_face, NoFaceDetection
from modules.utils import encode_array
dfloat32 = np.dtype('>f4')
# init_config('production.yml')
# init_module(Config)
mongo_driver.initialize_driver(Config.mongo_host, Config.mongo_port, Config.mongo_username, Config.mongo_password)

log_file_path = 'embedding_log.jl'
file_output_log = open(log_file_path, 'a')
file_data_insert_open_distro = 'data/data_insert_open_distro.jl'


def run(vn_celeb_dir: str):
    for people_id in sorted(os.listdir(vn_celeb_dir)):
        for file in sorted(os.listdir(os.path.join(vn_celeb_dir, people_id))):
            record = {
                '_id': f'vn_celeb_{people_id}_{file}',
                'people_id': f'vn_celeb_{people_id}',
                'file_id': file,
                'image': None,
                'face': None
            }
            person = mongo_driver.get_instance()['face_test']['face'].find_one({'_id': record['_id']})
            if person is not None:
                continue
            img_path = os.path.join(vn_celeb_dir, people_id, file)
            with open(img_path, 'rb') as f:
                base64_image = base64.b64encode(f.read()).decode('utf8')
            record['image'] = base64_image

            img = cv2.imread(img_path)
            try:
                faces = embed_face(img)
                embedding = get_embedding_from_model(faces)
                base64_embedding = encode_array(embedding)
                record['face'] = base64_embedding
            except NoFaceDetection as e:
                print("Not detect face", people_id, file)

            string_json = json.dumps(record, ensure_ascii=False)
            file_output_log.write(string_json + "\n")

            mongo_driver.get_instance()['face_test']['face'].insert_one(record)

    file_output_log.close()


def insert_es():
    elasticsearch_driver.initialize_driver(Config.es_hosts)
    es = elasticsearch_driver.get_instance()._es
    duplicate_count = 100
    for i in range(50, duplicate_count):
        with open(log_file_path, 'r') as f:
            for line in f:
                record = json.loads(line.strip())
                record['_id'] = f"{record['_id']}_copy_{i}"
                if record['face'] is not None:
                    res = es.exists(index='face_test_perform', id=record['_id'])
                    if not res:
                        data = {
                            'people_id': record['people_id'],
                            'face': record['face'],
                            'timestamp': datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                            'created_at': datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                        }
                        res = es.index(index='face_test_perform', id=record['_id'], body=data)
                        print(res)
                    else:
                        print("exist:", record['_id'])


def insert_es_multi():
    elasticsearch_driver.initialize_driver(Config.es_hosts)
    actions = []
    duplicate_count = 100
    for i in range(0, duplicate_count):
        with open(log_file_path, 'r') as f:
            for line in f:
                record = json.loads(line.strip())
                record['_id'] = f"{record['_id']}_copy_{i}"
                if record['face'] is not None:
                    data = {
                        'people_id': record['people_id'],
                        'face': record['face'],
                        'timestamp': datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                        'created_at': datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                    }
                    actions.append({
                        '_index': Config.face_index,
                        '_id': record['_id'],
                        '_source': data
                    })
                    if len(actions) > 5000:
                        print(f'{i}:', elasticsearch_driver.get_instance().insert_bulk(actions))
                        actions.clear()
    if len(actions) > 0:
        elasticsearch_driver.get_instance().insert_bulk(actions)
        actions.clear()


def insert_open_distro():
    elasticsearch_driver.initialize_driver(Config.es_hosts)
    # fout = open(file_data_insert_open_distro, "w")
    actions = []
    duplicate_count = 1
    for i in range(0, duplicate_count):
        with open(log_file_path, 'r') as f:
            for line in f:
                record = json.loads(line.strip())
                record['_id'] = f"{record['_id']}_copy_{i}"

                if record['face'] is not None:
                    embedding = base64.b64decode(record['face'])
                    embedding = np.frombuffer(embedding, dtype=dfloat32)
                    record['face'] = embedding.astype(np.float32).tolist()
                    data = {
                        'people_id': record['people_id'],
                        'face': record['face'],
                        'timestamp': datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                        'created_at': datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                    }
                    # line1 = json.dumps({'index': {'_index': 'face_test_perform', '_id': record['_id']}}, ensure_ascii=False)
                    # line2 = json.dumps(data, ensure_ascii=False)
                    # fout.write(line1 + "\n")
                    # fout.write(line2 + "\n")
                    actions.append({
                        '_index': Config.face_index,
                        '_id': record['_id'],
                        '_source': data
                    })
                    if len(actions) > 5000:
                        print(f'{i}:', elasticsearch_driver.get_instance().insert_bulk(actions))
                        actions.clear()
    if len(actions) > 0:
        print(f'{i}:', elasticsearch_driver.get_instance().insert_bulk(actions))
        actions.clear()
    # fout.close()


def insert_open_distro_faster():
    elasticsearch_driver.initialize_driver(Config.es_hosts)
    actions = []
    duplicate = 100
    for i in range(0, duplicate):
        with open(file_data_insert_open_distro, 'r') as f:
            for line in f:
                record = json.loads(line.strip())
                id = f"{record['people_id']}_copy_{i}"

                actions.append({
                    '_index': Config.face_index,
                    '_id': id,
                    '_source': record
                })
                if len(actions) > 5000:
                    print(f'{i}:', elasticsearch_driver.get_instance().insert_bulk(actions))
                    actions.clear()
                i += 1
    if len(actions) > 0:
        print(f'{i}:', elasticsearch_driver.get_instance().insert_bulk(actions))
        actions.clear()


def generate_body_data_test():
    pre_id = {}
    with open("data/embedding_log.jl", 'r') as f:
        for line in f:
            record = json.loads(line.strip())
            id = int(record['people_id'].split("_")[-1])
            if id not in pre_id :
                pre_id[id] = 1
                data = {"image": record['image']}
                data_str = json.dumps(data, ensure_ascii=False)
                with open(f"data/sample_body_new/{id}_0.json", 'w') as f:
                    f.write(data_str)
            elif pre_id[id] < 4:
                pre_id[id] += 1
                data = {"image": record['image']}
                data_str = json.dumps(data, ensure_ascii=False)
                with open(f"data/sample_body_new/{id}_1.json", 'w') as f:
                    f.write(data_str)


if __name__ == '__main__':
    generate_body_data_test()
