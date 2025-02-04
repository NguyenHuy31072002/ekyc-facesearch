import base64

import requests
import os
import json
url_register = "http://localhost:15000/register-face"
url_search = "http://localhost:15000/search"
source = "chamcong"

result_file = "results.txt"

def register(folder: str):
    for file in os.listdir(folder):
        people_id, ext = os.path.splitext(file)
        image_path = os.path.join(folder, file)
        with open(image_path, 'rb') as f:
            img_base64 = base64.b64encode(f.read()).decode('utf8')

        res = requests.post(url=url_register, json={
            "people_id": people_id,
            "created_at": "02/01/2021",
            "is_live_check": False,
            "image": img_base64,
            "source": source
        })
        print(f"{res.status_code}  ----  {res.json()}")


def search(folder: str):
    f_result = open(result_file, 'w')
    count_true = 0
    total = 0
    for file in os.listdir(folder):
        people_id, ext = os.path.splitext(file)
        image_path = os.path.join(folder, file)
        with open(image_path, 'rb') as f:
            img_base64 = base64.b64encode(f.read()).decode('utf8')

        res = requests.post(url=url_search, json={
            "image": img_base64,
            "source": source
        })
        print(f"{res.status_code}  ----  {res.json()}")
        if res.status_code == 200:
            result = json.dumps(res.json(), ensure_ascii=False)
            f_result.write(result + "\n")
            if len(result['data']) > 0:
                out_id = result['data'][0]['people_id']
                if out_id == people_id:
                    count_true += 1
        total += 1

    print(f"True: {count_true} / {total}")


if __name__ == '__main__':
    folder = ""
    register(folder)
    search(folder)
