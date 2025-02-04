# Face Service 1.1

# Change logs
+ 29/4/2021:
   - Update model check face anti spoof
   - Guide: download model anti spoof v2: https://drive.google.com/file/d/1j_m9JA5iWcCdx2MPTLFUWKBSNFcKSyUa/view?usp=sharing
    and copy to folder ```models/face/anti_spoof```  
+ 10/5/2021:
    - Update code training model face anti spoof: model training được xây dựng dựa trên mô hình phân loại 2 lớp fake vs real
    - Code training: [Notebook](classification_notebook.ipynb)
    - Data: [Google drive](https://drive.google.com/file/d/1zK2qy-oKX2QA3jTi0lKXZ7S-xmcz28NL/view?usp=sharing)
    - Other publish data: [Github](https://github.com/Davidzhangyuanhan/CelebA-Spoof)

## APIs
 - Face compare:
   + Feature: check if two face are similar
   + Endpoint: ```/check-2-face```
   + Format parameter: ```JSON```
   + Example: 
    ```json
      {
        "people_1": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAgAAAQABAAD/4QAwRXhpZgAASUkqAAgAA.....",
        "people_2": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAlgAAAJYCAYAAAC+ZpjcAAAABG.....",
        "is_live_check": true,
        "people_2_liveness": [
            "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAMCAgMCAgMDAgMDAwMDBAcFB.....",
            "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFB....."
        ]
    }
    ```

| Params            	| Sub params 	| Data Type    	| Required 	| Meaning                                	|
|-------------------	|------------	|--------------	|----------	|----------------------------------------	|
| people_1          	|            	| string       	| true     	| Base64 image of people 1               	|
| people_2          	|            	| string       	| true     	| Base64 image of people 2               	|
| is_live_check     	|            	| boolean      	| true     	| Option check liveness                  	|
| people_2_liveness 	|            	| list[string] 	| false    	| List of base64 image to check liveness 	|

 - Face OTP:
   + Feature: Check face of people by id or image
   + Endpoint: ```/face-otp```
   + Format parameter: ```JSON```
   + Example: 
    ```json
    {
      "people_1": {
        "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAgAAAQABAAD/4QAwRXhpZgAASUkqAAgAAAAB....."
      },
      "people_2": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAMCAgMCAgMDAgMDAwM.....",
      "is_live_check": true,
      "people_2_liveness": [
        "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEB....."
      ]
    }
    ```
    
| Params            	| Sub params 	| Data Type    	| Required             	| Meaning                                	|
|-------------------	|------------	|--------------	|----------------------	|----------------------------------------	|
| people_1          	|            	| dict         	| true                 	| info of people 1                       	|
|                   	| id         	| string       	| true, exclude: image 	| id of people 1                         	|
|                   	| image      	| string       	| true, exclude: id    	| Base64 image of people 1               	|
| people_2          	|            	| string       	| true                 	| Base64 image of people 2               	|
| is_live_check     	|            	| boolean      	| true                 	| Option check liveness                  	|
| people_2_liveness 	|            	| list[string] 	| false                	| List of base64 image to check liveness 	|

 - Face register:
   + Feature: register face information to database
   + Endpoint: ```/register-face```
   + Format parameter: ```JSON```
   + Example:
    ```json
    {
        "people_id": "test",
        "created_at": "02/01/2021",
        "is_live_check": true,
        "liveness": [
          "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAMCAgMCAgMDAgMDAwMDBAc.....",
          "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQg....."
        ],
        "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAgAAAQABAAD/4QAwRXhpZgAASUkqAAgA....."
    }
    ```
    
| Params        	| Sub params 	| Data Type    	| Required 	| Meaning                                	|
|---------------	|------------	|--------------	|----------	|----------------------------------------	|
| people_id     	|            	| string       	| true     	| id of people                           	|
| created_at    	|            	| string       	| true     	| time created records                   	|
| image         	|            	| string       	| true     	| Base64 image of people                 	|
| is_live_check 	|            	| boolean      	| true     	| Option check liveness                  	|
| liveness      	|            	| list[string] 	| false    	| List of base64 image to check liveness 	|

 - Face search:
   + Feature: search face information in database
   + Endpoint: ```/search```
   + Format parameter: ```JSON```
   + Example:
    ```json
    {
       "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAgAAAQABAAD/4QAwRXhpZgAASUkqAAgAA...."
    }
    ```
    
| Params            	| Sub params 	| Data Type    	| Required 	| Meaning                                	|
|-------------------	|------------	|--------------	|----------	|----------------------------------------	|
| image             	|            	| string       	| true     	| Base64 image of people                 	|

- Face search multi
   + Feature: search multi face information in database
   + Endpoint: ```/api/v2/search```
   + Format parameter: ```JSON```
   + Example: 
   ```json
   {
    "source":"source_default",
    "images": [
       {
         "image_id": "img1",
         "image": "base64....",
         "source": "source1"
       },
       {
         "image_id": "img2",
         "image": "base64....",
         "source": "source2"
       },
       {
         "image_id": "img3",
         "image": "base64...." /*không có source => lấy source mặc định ở trên, tức viviet*/
       }
    ]
   }
   ```

| Params 	| Sub params 	| Data Type 	| Required 	| Meaning                                	|
|--------	|------------	|-----------	|----------	|----------------------------------------	|
| source 	|            	| string    	| true     	| Default source                         	|
| images 	|            	| list      	| true     	| list of image                          	|
|        	| image_id   	| string    	| true     	| id of image                            	|
|        	| image      	| string    	| true     	| List of base64 image to check liveness 	|
|        	| source     	| string    	| false    	| source to query                        	|
|        	|            	|           	|          	|                                        	|

   - Response:
```json
{
    "status": "SUCCESS",
    "data": [
        {
            "image_id": "img1",
            "match_faces": [],
            "error": "Image input was wrong format. Please check your input image."  /*error xuất hiện khi có lỗi truy vấn ảnh đó*/
        },
        {
            "image_id": "img2",
            "match_faces": []  /*không tìm thấy khuôn mặt trùng khớp*/
        },
        {
            "image_id": "img3",
            "match_faces": [
                {
                    "people_id": "200008935",
                    "created_at": "1605606237",
                    "source": "viviet",
                    "score": 0.79422426
                }
            ]
        }
    ]
}
```
      + In case have error: 
         


 - Face remove:
   + Feature: remove face information in database
   + Endpoint: ```/remove```
   + Format parameter: ```JSON```
   + Example: 
   ```json
    {
      "people_id": "test"
    }
   ``` 

| Params            	| Sub params 	| Data Type    	| Required 	| Meaning                                	|
|-------------------	|------------	|--------------	|----------	|----------------------------------------	|
| people_id           	|            	| string       	| true     	| id of people                          	|   

## Frameworks
### Tensorflow Serving
   - Can serve multiple models, or multiple versions of the same model simultaneously. 
   - Exposes both gRPC as well as HTTP inference endpoints
   - Allows deployment of new model versions without changing any client code and zero downtime.
   - Adds minimal latency to inference time due to efficient, low-overhead implementation.

## Requirements
- Unix OS (Ubuntu is recommended)
- Docker CE: https://docs.docker.com/engine/install/
## Installation
### Setup databases (MongoDB, Elasticsearch) if you not have yet.
- docker-compose
```shell
docker-compose -f docker-compose-db.yml up --detach --no-recreate
```
- Docker Swarm
```shell
docker stack deploy --compose-file docker-compose-db.yml ekyc_db
```
### Install nvidia-docker (Optional)
If you want to deploy your models on GPU, it is necessary to install `nvidia-docker`. Just run the following commands as `sudo`:
<pre><code>cd serving
chmod +x ./install-nvidia-container-toolkit-ubuntu.sh
sh ./install-nvidia-container-toolkit-<b>your-OS-distro</b>.sh</code>
</pre>
Currently, there is only script for Ubuntu/Debian. You can follow this link to manually install `nvidia-docker`: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker
### Install TensorFlow Serving
```shell
mkdir serving
cd serving
wget https://github.com/tienthienhd/ekyc-lvt/releases/download/v1.0.0-model/model_release.zip
unzip model_release.zip
```

Then run TensorFlow Serving.
- CPU version:
```shell
# Run on CPU
./run.sh docker
```
- GPU version
```shell
./run.sh docker-gpu
```

### Create index face in ES
```shell
curl -XPUT "http://<ip host>:<port>/face" -H 'Content-Type: application/json' -d'{  "settings": {    "number_of_shards": 3,    "index": {      "knn": true,      "knn.space_type": "cosinesimil"    }  },  "mappings": {    "properties": {      "people_id": {        "type": "text",        "fields": {          "keyword": {            "type": "keyword",            "ignore_above": 256          }        }      },      "face": {        "type": "knn_vector",        "dimension": 512      },      "timestamp": {        "type": "date",        "format": "dd/MM/yyyy HH:mm:ss"      },      "created_at": {        "type": "text"      },      "source": {        "type": "text",        "fields": {          "keyword": {            "type": "keyword",            "ignore_above": 256          }        }      }    }  }}'
```

```shell
curl -XPUT "http://<ip host>:9200/face/_mapping" -H 'Content-Type: application/json' -d'{  "properties": {    "source": {      "type": "text",      "fields": {        "keyword": {          "type": "keyword",          "ignore_above": 256        }      }    }  }}'
```

### Create mongo document
```shell
docker exec -it ekyc-lvt_mongo_1 bash
use peple_face_image
db.createCollection('people')
```

### Install app and serve requests
   
- Build docker image using
```shell
cd application
./script_build.sh
```

- Change configuration in directory `application/config`
```yaml
DEBUG: False  # Debug mode or not

log_config_file: "logger.yaml"
log_folder: "logs"
log_images: "images"

# ES database config
enable_es: true
api_log_index: "api_log"
face_index: "face"
es_hosts: ["localhost:9200"]  # Host(s) to Elasticsearch

# Mongo config
enable_mongo: true
mongo_host: "localhost"  # Host to MongoDB
mongo_port: 27017  # Port to MongoDB
mongo_username: "root"  # MongoDB username
mongo_password: "lvT@123456"  # MongoDB password

# Face config. Change it if you know what you doing
similar_thresh_compare: 0.7
similar_thresh_liveness: 0.6
thresh_liveness: 0.6
blueprints: []
apis: ['face']

# TF Serving connection config
service_host: "localhost"  # Host to TF Serving
service_port:  8500  # Port to TF Serving
service_timeout: 10  # Timeout in seconds when connect to TF serving
```
- Run:
```shell
  # Create volume
   docker run -p <host_port>:15000 -v <config path>:/workspace/config/production.yml -e CONFIG_MODE=Production -v <path-log-folder>:/workspace/logs ekyc_lvt:lastest
   ```
Now the service is running on `host_port`.

# Docker compose
```shell
# load app image
docker load -i face_lvt.tar
docker-compose -f docker-compose.yml up -d
```

# Old documents
docker run -d --gpus all --name face -p 8082:80 -e WEB_CONCURRENCY=2 -e WORKERS_PER_CORE=0.5 -e MONGO_USERNAME=root -e MONGO_PASSWORD=lvT@123456 -e TOLERANCE=73.5,75,75 -e ELASTIC_SEARCH_HOST=es01 -e ELASTIC_SEARCH_USERNAME=admin -e ELASTIC_SEARCH_PASSWORD=admin -e LIVENESS_UNRECOGNIZE_TOLERANCE_THRESHOLD=60 -e LIVENESS_RECOGNIZE_THRESHOLD=0.665 -e GUNICORN_CMD_ARGS='--timeout=1600' --network face_reg_new_default face-gpu-4

## Backup Restore data from mongo to elastic search
```shell
 docker run --name restore -e MONGO_USERNAME=root -e MONGO_PASSWORD=lvT@123456 -e ELASTIC_SEARCH_HOST=es01 -e ELASTIC_SEARCH_USERNAME=admin -e ELASTIC_SEARCH_PASSWORD=admin -e START_FROM=1171682 -e NUMBER_RESTORE=450000 --network face_reg_new_default restore-es
```

### Models zoo
All models had compressed in: https://drive.google.com/file/d/1iA7y4LmiIEeH9NfCsDCEhSvSWw-LgEUN/view?usp=sharing
