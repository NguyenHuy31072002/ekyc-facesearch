import glob
import json
import os
import time
from typing import Tuple
import traceback

import requests

host = 'http://localhost:15000/search'


def send_request(data: dict) -> Tuple[int, int, int]:
    start_time = time.time_ns()
    res = requests.post(host, json=data)
    people_id = None
    if res.status_code == 200:
        content = res.json()
        if 'data' in content:
            if 'msg' in content['data']:
                people_id = -2
            elif len(content['data']) > 0:
                people_id = content['data'][0]['people_id']
    request_time = time.time_ns() - start_time
    return res.status_code, request_time, people_id


def load_data(file_path: str) -> dict:
    return json.load(open(file_path, 'r'))


def run(test_folder: str, file_result_path: str):
    file_result = open(file_result_path, 'w')
    file_result.write("filepath,people_id,status,request_time(ns),predict\n")
    for file_test in glob.glob(f"{test_folder}/*.json"):
        filename = os.path.basename(file_test)
        print(f"-------> {filename} <---------")
        people_id, ext = os.path.splitext(filename)
        data = load_data(file_test)
        status = -1
        request_time = -1
        predict = -1
        try:
            status, request_time, predict = send_request(data)
        except Exception as e:
            traceback.print_exc()
        line_result = f"{file_test},{people_id},{status},{request_time},{predict}\n"
        file_result.write(line_result)
    file_result.close()


def get_base64(img_path):
    import base64
    with open(img_path, 'rb') as f:
        base64_image = base64.b64encode(f.read()).decode('utf8')
    return base64_image

if __name__ == '__main__':
    # run('data/sample_body_new', 'result_predict_1k.csv')
    img_base64 = get_base64("/media/thiennt/projects/remote_lvt/ekyc-lvt/application/test/data/liveness/head_left/1.jpg")
    with open("/media/thiennt/projects/remote_lvt/ekyc-lvt/application/test/data/liveness/test.txt", 'w') as f:
        f.write(img_base64)