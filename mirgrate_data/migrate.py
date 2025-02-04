import urllib.parse
import requests
import pymongo
from config import *


def register(data):
    try:
        print(f"Registering {data['people_id']}")
        req = requests.post(
            url=REGISTER_FACE_URL,
            json=data,
        )

        if req.status_code != 200:
            print(f"Error {data['people_id']}. Reason: {req.text}")
    except Exception as e:
        print(f"Error {data['people_id']}. Reason: {e}")


def migrate():
    client = pymongo.MongoClient(
        f"mongodb://{urllib.parse.quote_plus(MONGO_USERNAME)}"
        f":{urllib.parse.quote_plus(MONGO_PASSWORD)}"
        f"@{MONGO_HOST}:{MONGO_PORT}"
    )

    db = client.peple_face_image

    for item in db.people.find():
        for i in item['data']:
            data = {
                "people_id": i["people_id"],
                "created_at": i["created_at"],
                "is_live_check": True if i["liveness"] is not None and len(i["liveness"]) else False,
                "liveness": i["liveness"],
                "image": i["image"],
            }

            if REGISTER_FACE_SOURCE is not None:
                data.update({
                    "source": REGISTER_FACE_SOURCE,
                })
            elif i.get("source") is not None:
                data.update({
                    "source": i["source"],
                })

            register(data)


migrate()