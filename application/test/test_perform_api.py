from locust import HttpUser, task, between, tag
import json


class ApiFace(HttpUser):
    host = 'http://171.244.7.233:15000'
    wait_time = between(1, 3)

    def on_start(self):
        self.data_face_compare = json.load(open('data/face_compare.json', 'r'))
        self.data_face_otp = json.load(open('data/face_otp.json', 'r'))
        self.data_face_register = json.load(open('data/face_register.json', 'r'))
        self.data_face_search = json.load(open('data/face_search.json', 'r'))
        self.data_face_remove = json.load(open('data/face_remove.json', 'r'))
        self.data_face_anti_spoof = json.load(open('data/face_anti_spoof.json', 'r'))

    @tag('compare')
    @task
    def face_compare(self):
        self.client.post('/check-2-face', json=self.data_face_compare)

    @tag('otp')
    @task
    def face_otp(self):
        self.client.post('/face-otp', json=self.data_face_otp)

    @tag('register')
    @task
    def face_register(self):
        self.client.post('/register', json=self.data_face_register)

    @tag('search')
    @task
    def face_search(self):
        self.client.post('/search', json=self.data_face_search)

    @tag('remove')
    @task
    def face_remove(self):
        self.client.post('/remove', json=self.data_face_remove)

    @tag('face_anti_spoof')
    @task
    def face_anti_spoof(self):
        self.client.post('/api/v1/anti-spoof', json=self.data_face_anti_spoof)
