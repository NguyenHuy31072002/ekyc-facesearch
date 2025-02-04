from datetime import datetime
from typing import List
import urllib.parse
from pymongo import MongoClient
import numpy as np
import logging

INSTANCE: MongoClient = None

LOGGER = logging.getLogger("mongo")


def initialize_driver(host, port, user_name, password):
    global INSTANCE
    if INSTANCE is None:
        if user_name is None or len(user_name) == 0:
            uri = f"mongodb://{host}:{port}"
        else:
            uri = f"mongodb://{urllib.parse.quote_plus(user_name)}:{urllib.parse.quote_plus(password)}@{host}:{port}"
        # LOGGER.info(f"Create connection to mongo: {uri}")
        INSTANCE = MongoClient(uri)


def get_instance():
    global INSTANCE
    if INSTANCE is None:
        raise Exception('Must be initialize Mongo driver before.')
    return INSTANCE


def insert_people(database: str,
                  collection: str,
                  image: str,
                  people_id: str,
                  liveness_image: str,
                  embedding: np.ndarray,
                  create_at: str,
                  source: str,
                  liveness_embedding: List[np.ndarray],
                  liveness_sol: List[float]):
    mongo_col = INSTANCE[database][collection]
    person = mongo_col.find_one({"data.0.people_id": people_id, "data.0.source": source})
    if person is None:
        data = {
            "current_version": 1,
            "data": [
                {
                    "_version": 1,
                    "people_id": people_id,
                    "image": image,
                    "liveness": liveness_image,
                    "face": embedding.tolist(),
                    "face_liveness": list(
                        map(lambda x: x.tolist(), liveness_embedding)) if liveness_embedding else None,
                    "liveness_sol": liveness_sol,
                    "created_at": create_at,
                    "source": source,
                    "timestamp": datetime.utcnow()
                }
            ],
            "created_at": datetime.utcnow()
        }
        LOGGER.info(f"Create new person: {people_id} from source: {source}")
        return mongo_col.insert_one(data)
    else:
        query = {
            'people_id': people_id
        }
        data = {
            "$inc": {"current_version": 1},
            "$push": {
                "data": {
                    "$each": [
                        {
                            "_version": person["current_version"] + 1,
                            "people_id": people_id,
                            "image": image,
                            "liveness": liveness_image,
                            "face": embedding.tolist(),
                            "face_liveness": list(
                                map(lambda x: x.tolist(), liveness_embedding)) if liveness_embedding else None,
                            "liveness_sol": liveness_sol,
                            "created_at": create_at,
                            "timestamp": datetime.utcnow()
                        },
                    ],
                    "$sort": {"timestamp": -1},
                    "$slice": 5
                }
            },
            "$set": {"updated_at": datetime.utcnow()}
        }
        LOGGER.info(f"Person is exists, update info to person {people_id} from source: {source}")
        return mongo_col.update_one(query, data).matched_count


def delete(database: str, collection: str, people_id: str, source: str):
    mongo_col = INSTANCE[database][collection]
    res_mongo_delete = mongo_col.delete_one(
        {"data.0.people_id": people_id, "data.0.source": source}).deleted_count
    LOGGER.info(f"Delete person: {people_id} from source: {source}: {res_mongo_delete}")
    return res_mongo_delete
