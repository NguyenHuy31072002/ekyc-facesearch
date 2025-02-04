import concurrent.futures
import logging
import os
from typing import Dict, List

from PIL import UnidentifiedImageError
from elasticsearch import TransportError
from flask import Blueprint
from flask_restful import Api, Resource, reqparse

from app.api import ResponseStatus
from app.utils import base64_to_image, get_embedding_from_model, get_embedding_from_es
from config_run import Config
from modules.driver import mongo_driver, elasticsearch_driver
from modules.face import embed_face, detect_face, NoFaceDetection, compare_face, check_liveness, get_similars, \
    check_action, embed_face_only, is_fake
from modules.face.liveness import command

blueprint = Blueprint(
    name='face',
    import_name=__name__,
    url_prefix='',
)

api = Api(blueprint, prefix='')

LOGGER = logging.getLogger("api")


@blueprint.before_app_first_request
def initialize_api():
    pass


class FaceDetectApi(Resource):
    def post(self):
        try:
            imgs, filenames = self.parse_args()
            faces = detect_face(imgs[0], size_threshold=0.01, max_faces=1, receive_mode='meta')
            if len(faces) > 0:
                return {
                    "status": "success",
                    "status_code": 200,
                    "has_face": True,
                    "face_box": faces[0].box
                }
            else:
                return {
                    "status": "success",
                    "status_code": 200,
                    "has_face": False
                }
        except Exception as e:
            LOGGER.exception(e)
            return {
                "status": "unsuccess",
                "status_code": 500,
                "message": str(e)
            }


class FaceCompareApi(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('people_1', required=True, type=str, location=['json'],
                            help='Required image: base64 of image ({error_msg})')
        parser.add_argument('people_2', required=True, type=str, location=['json'],
                            help='Required image: base64 of image ({error_msg})')
        parser.add_argument('is_live_check', required=True, type=bool, location=['json'],
                            help='Required variable: true/false ({error_msg})')
        parser.add_argument('people_2_liveness', required=False, type=str, location=['json'], action='append',
                            help='List of images liveness, required if is_live_check=true')
        args = parser.parse_args()

        try:
            people_1_img = base64_to_image(args.people_1)
            people_2_imgs = [base64_to_image(args.people_2)]
            if args.is_live_check:
                if not args.people_2_liveness:
                    return {
                               'status': ResponseStatus.INPUT_ERROR.value,
                               'msg': 'is_live_check=true then people_2_liveness is required.'
                           }, 400
                people_2_imgs.extend([base64_to_image(img) for img in args.people_2_liveness])
        except Exception as e:
            LOGGER.exception(e)
            return {
                       'status': ResponseStatus.INPUT_ERROR.value,
                       'msg': 'Wrong image input format.'
                   }, 400

        try:
            matches, sims = compare_face(people_1_img, people_2_imgs[:1], Config.similar_thresh_compare)
            if len(matches) > 0 and matches[0]:
                logging.info("COMPARE 2 PEOPLE SOLUTION : " + str(matches[0]) + "  " + str(sims[0]))
                if args.is_live_check:
                    matches_liveness, sims_liveness = compare_face(people_1_img, people_2_imgs[1:],
                                                                   Config.similar_thresh_liveness)
                    is_liveness = check_liveness(matches_liveness, sims_liveness)
                    return {
                               "status": ResponseStatus.SUCCESS.value,
                               "compare_result": "MATCH",
                               # 'is_fake': is_fake(people_1_img) or is_fake(people_2_imgs[0]),
                               "is_live": is_liveness,
                               "feature_vector": [],
                               "similator_percent": sims[0]
                           }, 200
                else:
                    return {
                               "status": ResponseStatus.SUCCESS.value,
                               "compare_result": "MATCH",
                               # 'is_fake': is_fake(people_1_img) or is_fake(people_2_imgs[0]),
                               "feature_vector": [],
                               "similator_percent": sims[0]
                           }, 200
            else:
                return {
                           "status": ResponseStatus.SUCCESS.value,
                           "compare_result": "DO_NOT_MATCH",
                           # 'is_fake': is_fake(people_1_img) or is_fake(people_2_imgs[0]),
                           "similator_percent": sims[0]
                       }, 200

        except NoFaceDetection:
            return {
                       'status': ResponseStatus.NO_FACE_ERROR.value,
                       'msg': 'Can\'t detect face. Please update another picture'
                   }, 200
        except Exception as e:
            LOGGER.exception(e)
            return {
                       'status': ResponseStatus.ERROR.value,
                       'msg': str(e)
                   }, 500


class FaceOTPApi(Resource):
    def __init__(self):
        self.es_driver = elasticsearch_driver.get_instance()

    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('people_1', required=True, type=dict, location=['json'])
        parser.add_argument('people_2', required=True, type=str, location=['json'])
        parser.add_argument('is_live_check', required=True, type=bool, location=['json'])
        parser.add_argument('people_2_liveness', required=False, type=str, location=['json'], action='append')

        try:
            args = parser.parse_args()

            if 'image' in args.people_1 and 'id' not in args.people_1:
                people_1_image = base64_to_image(args.people_1['image'])
                people_1_faces = embed_face(people_1_image, max_faces=1, fast=True)
                people_1_emb = get_embedding_from_model(people_1_faces)
            elif 'image' not in args.people_1 and 'id' in args.people_1:
                es_res = self.es_driver.query_by_id(Config.face_index, args.people_1['id'])
                people_1_emb = get_embedding_from_es(es_res)
            else:
                return {
                           'status': ResponseStatus.INPUT_ERROR.value,
                           'msg': 'image or id required in people_1'
                       }, 400

            people_2_imgs = [base64_to_image(args.people_2)]
            if args.is_live_check:
                if not args.people_2_liveness:
                    return {
                               'status': ResponseStatus.INPUT_ERROR.value,
                               'msg': 'is_live_check=true then people_2_liveness is required.'
                           }, 400
                people_2_imgs.extend([base64_to_image(img) for img in args.people_2_liveness])

            # Get embedding and compare
            people_2_embs = [get_embedding_from_model(embed_face(people_2_img)) for people_2_img in people_2_imgs]
            matches, sims = get_similars(people_1_emb, people_2_embs[:1], Config.similar_thresh_compare)
            if len(matches) > 0 and matches[0]:
                if args.is_live_check:
                    matches_liveness, sims_liveness = get_similars(people_1_emb, people_2_embs[1:],
                                                                   Config.similar_thresh_liveness)
                    is_liveness = check_liveness(matches_liveness, sims_liveness)
                    return {
                               "status": ResponseStatus.SUCCESS.value,
                               "compare_result": "MATCH",
                               "is_live": is_liveness,
                               "feature_vector": [],
                               "similator_percent": sims[0]
                           }, 200
                else:
                    return {
                               "status": ResponseStatus.SUCCESS.value,
                               "compare_result": "MATCH",
                               "feature_vector": [],
                               "similator_percent": sims[0]
                           }, 200
            else:
                return {
                           "status": ResponseStatus.SUCCESS.value,
                           "compare_result": "DO_NOT_MATCH",
                           "similator_percent": sims[0]
                       }, 200
        except UnidentifiedImageError as e:
            LOGGER.exception(e)
            return {
                       'status': ResponseStatus.INPUT_ERROR.value,
                       'msg': 'Wrong image input format.'
                   }, 400

        except NoFaceDetection as e:
            LOGGER.exception(e)
            return {
                       'status': ResponseStatus.NO_FACE_ERROR.value,
                       'msg': str(e)
                   }, 200

        except Exception as e:
            LOGGER.exception(e)
            return {
                       'status': ResponseStatus.ERROR.value,
                       'msg': str(e)
                   }, 500


class FaceRegistryApi(Resource):
    def __init__(self):
        self.es_driver = elasticsearch_driver.get_instance()
        # self.mongo_driver = mongo_driver.get_instance()[Config.mongo_database]

    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('image', type=str, required=True, location=['json'], help='Input image: {error_msg}')
        parser.add_argument('people_id', type=str, required=True, location=['json'])
        parser.add_argument('created_at', type=str, required=True, location=['json'])
        parser.add_argument('source', type=str, default='default', location=['json'])
        parser.add_argument('is_live_check', type=bool, required=True, location=['json'])
        parser.add_argument('liveness', type=str, required=False, default=None, location=['json'], action='append')
        args = parser.parse_args()

        try:
            imgs = [base64_to_image(args.image)]
            if args.is_live_check:
                if not args.liveness:
                    return {
                               'status': ResponseStatus.INPUT_ERROR.value,
                               'msg': 'is_live_check=true then "liveness" is required.'
                           }, 400
                imgs.extend([base64_to_image(img) for img in args.liveness])
        except AttributeError as e:
            LOGGER.exception(e)
            return {
                       'status': ResponseStatus.INPUT_ERROR.value,
                       'msg': 'is_live_check=true then "liveness" is required.'
                   }, 400
        except Exception as e:
            LOGGER.exception(e)
            return {
                       'status': ResponseStatus.INPUT_ERROR.value,
                       'msg': 'Wrong image input format.'
                   }, 400

        try:
            face_embs = []
            for img in imgs:
                faces = embed_face(img, max_faces=1, fast=False)
                face_embs.append(get_embedding_from_model(faces))

            sims = None
            is_liveness = False
            if args.is_live_check:
                matches, sims = get_similars(face_embs[0], face_embs[1:], Config.similar_thresh_liveness)
                is_liveness = check_liveness(matches, sims)

            if not args.is_live_check or is_liveness:
                res_es = self.es_driver.insert_face2(Config.face_index,
                                                     args.people_id,
                                                     face_embs[0],
                                                     args.created_at,
                                                     args.source)
                LOGGER.info(f'ES: updated: {res_es}')
                res_mongo = mongo_driver.insert_people(Config.mongo_database,
                                                       Config.mongo_collection,
                                                       args.image,
                                                       args.people_id,
                                                       args.liveness,
                                                       face_embs[0],
                                                       args.created_at,
                                                       args.source,
                                                       face_embs[1:],
                                                       sims)
                LOGGER.info(f'Mongo: {res_mongo} records is updated.')

                logging.info("Register success image " + args.people_id)

                return {
                           'status': ResponseStatus.SUCCESS.value,
                           '_id': args.people_id,
                           # 'is_fake': is_fake(imgs[0]),
                           'source': args.source,
                       }, 200

        except NoFaceDetection:
            return {
                       'status': ResponseStatus.NO_FACE_ERROR.value,
                       'msg': 'Can\'t detect face. Please update another picture'
                   }, 400
        except Exception as e:
            LOGGER.exception(e)
            return {
                       'status': ResponseStatus.ERROR.value,
                       'msg': str(e)
                   }, 500
        return {
                   'status': ResponseStatus.LIVENESS_FAIL.value,
                   'msg': 'check liveness fail'
               }, 400


class FaceSearchApi(Resource):
    def __init__(self):
        self.es_driver = elasticsearch_driver.get_instance()

    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('image', required=True, location=['json'])
        parser.add_argument('source', type=str, default='default', location=['json'])
        args = parser.parse_args()

        try:
            img = base64_to_image(args.image)
        except Exception as e:
            LOGGER.exception(e)
            return {
                       'status': ResponseStatus.INPUT_ERROR.value,
                       'msg': 'Wrong image input format.'
                   }, 400

        try:
            faces = embed_face(img, max_faces=1, fast=False)
            face_emb = get_embedding_from_model(faces)

            res_es = self.es_driver.query_embedding2(Config.face_index, face_emb.tolist(), args.source,
                                                     outputs=['people_id', 'created_at', 'source'],
                                                     min_score=Config.similar_thresh_compare)

            data_res = []
            for r in res_es:
                data_res.append({
                    'people_id': r['_source']['people_id'],
                    'created_at': r['_source']['created_at'],
                    'source': r['_source']['source'],
                    'score': r['_score']
                })
            return {
                       'status': ResponseStatus.SUCCESS.value,
                       # 'is_fake': is_fake(img),
                       'data': data_res
                   }, 200
        except NoFaceDetection as e:
            LOGGER.exception(e)
            return {
                'status': ResponseStatus.NO_FACE_ERROR.value,
                'data': {
                    'msg': str(e)
                }
            }
        except TransportError as e:
            LOGGER.exception(e)
            return {
                       'status': ResponseStatus.ERROR.value,
                       'data': {'msg': "Not found."}
                   }, 500
        except Exception as e:
            LOGGER.exception(e)
            return {
                       'status': ResponseStatus.ERROR.value,
                       'data': {'msg': str(e)}
                   }, 500


class FaceSearchApiV2(Resource):
    def __init__(self):
        self.es_driver = elasticsearch_driver.get_instance()

    def _search_face(self, image):
        img = image['image']
        source = image['source']
        image_id = image['image_id']
        try:
            faces = embed_face(img, max_faces=1, fast=True)
            face_emb = get_embedding_from_model(faces)

            res_es = self.es_driver.query_embedding2(Config.face_index, face_emb.tolist(), source,
                                                     outputs=['people_id', 'created_at', 'source'],
                                                     min_score=Config.similar_thresh_compare)

            data_res = []
            for r in res_es:
                data_res.append({
                    'people_id': r['_source']['people_id'],
                    'created_at': r['_source']['created_at'],
                    'source': r['_source']['source'],
                    'score': r['_score']
                })
            return {
                'image_id': image_id,
                # 'is_fake': is_fake(img),
                'match_faces': data_res
            }
        except NoFaceDetection as e:
            LOGGER.exception(e)
            return {
                'image_id': image_id,
                'match_faces': [],
                'error': str(e)
            }
        except TransportError as e:
            LOGGER.exception(e)
            return {
                'image_id': image_id,
                'match_faces': [],
                'error': 'Not found.',
            }
        except AssertionError as e:
            LOGGER.exception(e)
            return {
                'image_id': image_id,
                'match_faces': [],
                'error': 'Image input was wrong format. Please check your input image.',
            }
        except Exception as e:
            LOGGER.exception(e)
            return {
                'image_id': image_id,
                'match_faces': [],
                'error': str(e),
            }

    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('images', required=True, type=dict, location=['json'], action='append')
        parser.add_argument('source', type=str, default='default', location=['json'])
        args = parser.parse_args()

        images = []

        for image in args.images:
            try:
                img = base64_to_image(image['image'])
                images.append({
                    'image_id': image['image_id'],
                    'image': img,
                    'source': image['source'] if 'source' in image else args.source
                })

            except Exception as e:
                LOGGER.exception(e)
                images.append({
                    'image_id': image['image_id'],
                    'image': None,
                    'source': image['source'] if 'source' in image else args.source
                })
                # return {
                #            'status': ResponseStatus.INPUT_ERROR.value,
                #            'msg': 'Wrong image input format.'
                #        }, 400

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() - 1,
                                                         thread_name_prefix="face search")
        results = list(executor.map(self._search_face, images))

        return {
                   'status': ResponseStatus.SUCCESS.value,
                   'data': results
               }, 200


class FaceSearchApiV3(Resource):
    def __init__(self):
        self.es_driver = elasticsearch_driver.get_instance()

    @staticmethod
    def _detect_face(image: Dict) -> Dict:
        """Detect face in image
        Args:
            image: Dict info of image .
        Return:
            Dict info has face object
        """
        try:
            faces = detect_face(image['image'], max_faces=1, receive_mode="image", fast=True)
            if len(faces) > 0:
                return {
                    'image_id': image['image_id'],
                    'face': faces[0],
                    'source': image['source']
                }
            else:
                raise NoFaceDetection('Not found face in image.')
        except Exception as e:
            LOGGER.exception(e)
            return {
                'image_id': image['image_id'],
                'match_faces': [],
                'error': str(e)
            }

    @staticmethod
    def _get_embedding(images: List[Dict]) -> List[Dict]:
        img_faces = []
        for image in images:
            if 'face' in image:
                img_face = image['face'].image
                img_faces.append(img_face)

        embeddings = embed_face_only(img_faces)
        i = 0
        for image in images:
            if 'face' in image:
                embedding = embeddings[i].tolist()
                image['embedding'] = embedding
                i += 1
        return images

    def _search(self, images: List[Dict]) -> List[Dict]:
        embeddings = []
        image_ids = []
        sources = []
        for i, image in enumerate(images):
            if 'embedding' in image:
                embeddings.append(image['embedding'])
                image_ids.append(image['image_id'])
                sources.append(image['source'])

        try:
            result_search = self.es_driver.query_embedding_multi(Config.face_index, image_ids, embeddings, sources,
                                                                 outputs=['people_id', 'created_at', 'source'],
                                                                 min_score=Config.similar_thresh_compare)
            for i, image in enumerate(images):
                image_id = image['image_id']
                if image_id in result_search:
                    images[i] = {
                        'image_id': image_id,
                        'match_faces': result_search[image_id]
                    }
            return images
        except Exception as e:
            LOGGER.exception(e)
            for i, image in enumerate(images):
                image_id = image['image_id']
                if image_id in result_search:
                    images[i] = {
                        'image_id': image_id,
                        'match_faces': [],
                        'message': str(e)
                    }
            return images

    def _get_image_args(self, args):
        image, default_source = args
        return self._get_image(image, default_source)

    @staticmethod
    def _get_image(image: Dict, default_source: str) -> Dict:
        """Parse string image to array image
        Args:
            image: Dict info of image. Keys: image, image_id, source
            default_source: string, default source to search. Use if image not define source
        Return:
              Dict info of image but image is array
        """
        try:
            img = base64_to_image(image['image'])
            return {
                'image_id': image['image_id'],
                'image': img,
                'source': image['source'] if 'source' in image else default_source
            }
        except Exception as e:
            LOGGER.exception(e)
            return {
                'image_id': image['image_id'],
                'image': None,
                'source': image['source'] if 'source' in image else default_source
            }

    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('images', required=True, type=dict, location=['json'], action='append')
        parser.add_argument('source', type=str, default='default', location=['json'])
        args = parser.parse_args()

        images = [(image, args.source) for image in args.images]
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() - 1,
                                                         thread_name_prefix="face search")

        array_images = list(executor.map(self._get_image_args, images))

        faces = list(executor.map(self._detect_face, array_images))

        embeddings = self._get_embedding(faces)

        result_search = self._search(embeddings)
        return {
                   'status': ResponseStatus.SUCCESS.value,
                   'data': result_search
               }, 200


class FaceRemoveApi(Resource):
    def __init__(self):
        self.es_driver = elasticsearch_driver.get_instance()
        # self.mongo_driver = mongo_driver.get_instance()[Config.mongo_database]

    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('people_id', type=str, required=True, location=['json'])
        parser.add_argument('source', type=str, default='default', location=['json'])
        args = parser.parse_args()
        people_id = args.people_id
        source = args.source
        logging.info(f"REMOVE: {people_id} from {source}")

        # ============= Remove people in db ============
        res_es_delete = self.es_driver.delete_people(Config.face_index, people_id, source)
        res_mongo_delete = mongo_driver.delete(database=Config.mongo_database, collection=Config.mongo_collection,
                                               people_id=people_id, source=source)
        LOGGER.info(f'ES delete match {res_es_delete}')
        LOGGER.info(f'Mongo delete match {res_mongo_delete}')
        if res_es_delete['total'] == 0:
            return {
                       'status': ResponseStatus.PEOPLE_NOT_FOUND_ERROR.value,
                       'msg': f"Data doesn't exits for people '{people_id}' from '{source}'"
                   }, 200
        else:
            return {
                       'status': ResponseStatus.SUCCESS.value,
                       'msg': f"Success remove people '{people_id}' from '{source}'"
                   }, 200


class FaceLiveness(Resource):
    def __init__(self):
        pass

    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('image', required=True, location=['json'])
        parser.add_argument('cmds', type=dict, location=['json'], action='append')
        args = parser.parse_args()

        try:
            image = base64_to_image(args.image)
            cmds = args.cmds
            for cmd in cmds:
                cmd['action'] = command.get_command(cmd['action'])
                for i, img in enumerate(cmd['images']):
                    cmd['images'][i] = base64_to_image(img)

            main_faces = embed_face(image, max_faces=1, fast=True)
            main_face_emb = get_embedding_from_model(main_faces)

            action_results = []
            # Get embedding and compare
            for cmd in cmds:
                embs = [get_embedding_from_model(embed_face(img)) for img in cmd['images']]
                matches, sims = get_similars(main_face_emb, embs, Config.similar_thresh_compare)
                if len(matches) > 0:
                    is_liveness = check_liveness(matches, sims)
                    action_results.append({
                        'cmd': cmd['action'].name,
                        'pass_action': check_action(cmd['images'], cmd['action']),
                        'similar': is_liveness
                    })

                else:
                    action_results.append({
                        'cmd': cmd['action'].name,
                        'pass_action': check_action(cmd['images'], cmd['action']),
                        'similar': False
                    })

            return {
                       "status": ResponseStatus.SUCCESS.value,
                       "cmds": action_results
                   }, 200
        except UnidentifiedImageError as e:
            LOGGER.exception(e)
            return {
                       'status': ResponseStatus.INPUT_ERROR.value,
                       'msg': 'Wrong image input format.'
                   }, 400

        except NoFaceDetection as e:
            LOGGER.exception(e)
            return {
                       'status': ResponseStatus.NO_FACE_ERROR.value,
                       'msg': str(e)
                   }, 200

        except Exception as e:
            LOGGER.exception(e)
            return {
                       'status': ResponseStatus.ERROR.value,
                       'msg': str(e)
                   }, 500


class FaceAntiSpoofApi(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('image', required=True, location=['json'])
        args = parser.parse_args()
        try:
            image = base64_to_image(args.image)
            fake = is_fake(image)
            return {
                "status": ResponseStatus.SUCCESS.value,
                "fake_prob": round(fake['fake_prob'] * 100, 5),
                "is_fake": fake['is_fake']
            }, 200
        except UnidentifiedImageError as e:
            LOGGER.exception(e)
            return {
                       'status': ResponseStatus.INPUT_ERROR.value,
                       'msg': 'Wrong image input format.'
                   }, 400

        except NoFaceDetection as e:
            LOGGER.exception(e)
            return {
                       'status': ResponseStatus.NO_FACE_ERROR.value,
                       'msg': str(e)
                   }, 400
        except Exception as e:
            LOGGER.exception(e)
            return {
                "status": ResponseStatus.ERROR.value,
                "msg": str(e)
            }, 500


api.add_resource(FaceDetectApi, '/face-detect')
api.add_resource(FaceCompareApi, '/check-2-face')
api.add_resource(FaceOTPApi, '/face-otp')
api.add_resource(FaceRegistryApi, '/register-face')
api.add_resource(FaceSearchApi, '/search')
api.add_resource(FaceSearchApiV2, '/api/v2/search')
api.add_resource(FaceSearchApiV3, '/api/v3/search')
api.add_resource(FaceRemoveApi, '/remove')
api.add_resource(FaceLiveness, '/face-live')
api.add_resource(FaceAntiSpoofApi, '/api/v1/anti-spoof')
