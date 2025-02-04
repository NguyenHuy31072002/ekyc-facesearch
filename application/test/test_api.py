import base64
import os
import time
import unittest
from unittest import TestCase

import requests


class TestApi(TestCase):

    def setUp(self) -> None:
        self.port = 15000
        self.host = "localhost"
        data_dir = './images'
        self.images = {}
        self.peoples = []
        for people_id in sorted(os.listdir(data_dir)):
            self.peoples.append(people_id)
            self.images[people_id] = []
            for img_name in sorted(os.listdir(os.path.join(data_dir, people_id))):
                image_path = os.path.join(data_dir, people_id, img_name)
                with open(image_path, 'rb') as f:
                    img_base64 = base64.b64encode(f.read()).decode('utf8')
                self.images[people_id].append({
                    'image': img_base64,
                    'people_id': people_id,
                    'image_id': f'{people_id}_{img_name}'
                })

    def test_face_compare(self):
        people_id = 'thuychi'
        res = requests.post(url=f"http://{self.host}:{self.port}/check-2-face", json={
            "people_1": self.images[people_id][0]['image'],
            "people_2": self.images[people_id][1]['image'],
            "is_live_check": False
        })
        assert res.status_code == 200
        result = res.json()
        print(result)
        assert result["status"] == "SUCCESS"
        assert result['compare_result'] == "MATCH"
        assert result['similator_percent'] > 0.7

    def test_face_otp(self):
        people_id = 'thuychi'
        res = requests.post(url=f"http://{self.host}:{self.port}/face-otp", json={
            "people_1": {
                "image": self.images[people_id][0]['image'],
            },

            "people_2": self.images[people_id][1]['image'],
            "is_live_check": True,
            "people_2_liveness": [i['image'] for i in self.images[people_id][2:]]
        })
        result = res.json()
        assert res.status_code == 200
        assert result["status"] == "SUCCESS"
        assert result['compare_result'] == "MATCH"
        assert result['similator_percent'] > 0.7

    def test_insert_manual(self):
        people_id = "200008935"
        source = "test_viviet"
        res = requests.post(url=f"http://{self.host}:{self.port}/register-face", json={
            "people_id": people_id,
            "created_at": "02/01/2021",
            "is_live_check": False,
            "image": self.images["lvt_test1"][0]['image'],
            "source": source
        })
        assert res.status_code == 200
        result = res.json()
        assert result['status'] == "SUCCESS"
        assert result['_id'] == people_id
        assert result['source'] == source

        people_id = "200009313"
        source = "test_viviet"
        res = requests.post(url=f"http://{self.host}:{self.port}/register-face", json={
            "people_id": people_id,
            "created_at": "02/01/2021",
            "is_live_check": False,
            "image": self.images["lvt_test2"][0]['image'],
            "source": source
        })
        assert res.status_code == 200
        result = res.json()
        assert result['status'] == "SUCCESS"
        assert result['_id'] == people_id
        assert result['source'] == source

    def test_search_manual(self):
        source = "test_viviet"
        res = requests.post(url=f"http://{self.host}:{self.port}/search", json={
            "image": self.images['lvt_test1'][1]['image'],
            "source": source
        })
        assert res.status_code == 200
        result = res.json()
        print(result)
        assert result['status'] == 'SUCCESS'
        assert len(result['data']) > 0
        assert result['data'][0]['people_id'] == '200008935'
        assert result['data'][0]['score'] > 0.75
        assert result['data'][0]['source'] == source

        res = requests.post(url=f"http://{self.host}:{self.port}/search", json={
            "image": self.images['lvt_test2'][1]['image'],
            "source": source
        })
        assert res.status_code == 200
        result = res.json()
        print(result)
        assert result['status'] == 'SUCCESS'
        assert len(result['data']) > 0
        assert result['data'][0]['people_id'] == '200009313'
        assert result['data'][0]['score'] > 0.7
        assert result['data'][0]['source'] == source

    def test_face_register(self):
        source = "test"
        for people_id in self.peoples:
            if len(self.images[people_id]) == 4:
                res = requests.post(url=f"http://{self.host}:{self.port}/register-face", json={
                    "people_id": people_id,
                    "created_at": "02/01/2021",
                    "is_live_check": True,
                    "liveness": [i['image'] for i in self.images[people_id][1:2]],
                    "image": self.images[people_id][0]['image'],
                    "source": source
                })
                assert res.status_code == 200
                result = res.json()
                print(result)
                assert result['status'] == 'SUCCESS'
                assert result['_id'] == people_id
                assert result['source'] == source

    def test_face_search(self):
        res = requests.post(url=f"http://{self.host}:{self.port}/register-face", json={
            "people_id": 'thuychi',
            "created_at": "02/01/2021",
            "is_live_check": False,
            "image": self.images["thuychi"][0]['image'],
            "source": 'test'
        })
        assert res.status_code == 200
        result = res.json()
        print(result)

        source = "test"
        res = requests.post(url=f"http://{self.host}:{self.port}/search", json={
            "image": self.images['thuychi'][3]['image'],
            "source": source
        })
        assert res.status_code == 200
        result = res.json()
        print(result)
        assert result['status'] == 'SUCCESS'
        assert len(result['data']) > 0
        assert result['data'][0]['people_id'] == 'thuychi'
        assert result['data'][0]['score'] > 0.7
        assert result['data'][0]['source'] == source

    def test_face_search_v2(self):
        people_id1 = 'mytam'
        people_id2 = 'thuychi'
        people_id3 = 'tuanhung'

        res = requests.post(url=f"http://{self.host}:{self.port}/api/v2/search", json={
            'source': 'other',
            "images": [
                {'image_id': self.images[people_id1][2]['image_id'], 'image': self.images[people_id1][2]['image'],
                 'source': 'test'},
                {'image_id': self.images[people_id2][2]['image_id'], 'image': self.images[people_id2][2]['image']},
                {'image_id': self.images[people_id3][2]['image_id'], 'image': self.images[people_id3][2]['image'],
                 'source': 'test'},
            ]
        })

        assert res.status_code == 200
        result = res.json()
        assert result['status'] == 'SUCCESS'
        assert len(result['data']) > 0
        assert len(result['data'][0]['match_faces']) > 0
        assert len(result['data'][1]['match_faces']) == 0
        assert len(result['data'][2]['match_faces']) > 0

    def test_perform_face_search_multi(self):
        people_id1 = 'mytam'
        people_id2 = 'thuychi'
        people_id3 = 'tuanhung'
        start_time = time.time()

        res = requests.post(url=f"http://{self.host}:{self.port}/api/v2/search", json={
            'source': 'other',
            "images": [
                {'image_id': self.images[people_id1][2]['image_id'], 'image': self.images[people_id1][2]['image'],
                 'source': 'test'},
                {'image_id': self.images[people_id2][2]['image_id'], 'image': self.images[people_id2][2]['image']},
                {'image_id': self.images[people_id3][2]['image_id'], 'image': self.images[people_id3][2]['image'],
                 'source': 'test'},
            ] * 1
        })
        end_time = time.time()
        assert res.status_code == 200
        assert end_time - start_time <= 6
        result = res.json()
        from pprint import pprint
        pprint(result)

    def test_face_remove(self):
        res = requests.post(url=f"http://{self.host}:{self.port}/remove", json={
            "people_id": "thuychi",
            "source": "test"
        })
        assert res.status_code == 200
        result = res.json()
        assert result['status'] == 'SUCCESS'

        res = requests.post(url=f"http://{self.host}:{self.port}/remove", json={
            "people_id": "tuanhung"
        })
        assert res.status_code == 200
        result = res.json()
        assert result['status'] == 'PEOPLE_NOT_FOUND_ERROR'

    def test_face_live(self):
        res = requests.post(url=f"http://{self.host}:{self.port}/face-live", json={
            "image": self.images['head_stand'][0]['image'],
            "cmds": [
                {
                    "action": "HEAD_UP",
                    "images": [self.images['head_up'][0]['image'], self.images['head_up'][1]['image']]
                },
                {
                    "action": "HEAD_DOWN",
                    "images": [self.images['head_down'][0]['image'], self.images['head_down'][1]['image']]
                }
            ]
        })

        assert res.status_code == 200
        result = res.json()
        assert result['status'] == 'SUCCESS'
        assert len(result['cmds']) > 0

    def test_face_fake(self):
        res = requests.post(url=f"http://{self.host}:{self.port}/api/v1/anti-spoof", json={
            "image": self.images['mytam'][0]['image']
        })

        assert res.status_code == 200
        result = res.json()
        print(result)
        assert result['status'] == 'SUCCESS'
        assert result['is_fake'] == False

        res = requests.post(url=f"http://{self.host}:{self.port}/api/v1/anti-spoof", json={
            "image": self.images['fake'][0]['image']
        })

        assert res.status_code == 200
        result = res.json()
        print(result)
        assert result['status'] == 'SUCCESS'
        assert result['is_fake'] == True


if __name__ == '__main__':
    unittest.main()
