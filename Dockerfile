FROM tensorflow/tensorflow:latest-gpu
#FROM nvidia/cuda:11.0-runtime

#RUN apt-get update
#RUN apt-get install -y software-properties-common
#RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y cmake
#RUN apt-get install -y  libopenblas-dev liblapack-dev

RUN apt-get install -y python3
RUN apt-get install -y python3-pip
RUN pip3 install --upgrade pip
RUN apt-get install -y libgl1-mesa-glx libgtk2.0-dev
COPY ./app/requirements.txt  /tmp/requirements.txt 
RUN pip3 install -r /tmp/requirements.txt
RUN pip3 install -v dlib
RUN pip3 install meinheld gunicorn

COPY --from=tiangolo/meinheld-gunicorn:python3.8 ./entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

COPY --from=tiangolo/meinheld-gunicorn:python3.8 ./start.sh /start.sh
RUN chmod +x /start.sh

COPY --from=tiangolo/meinheld-gunicorn:python3.8  ./gunicorn_conf.py /gunicorn_conf.py

COPY --from=tiangolo/meinheld-gunicorn:python3.8 ./app /app
WORKDIR /app/

ENV PYTHONPATH=/app

EXPOSE 80

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/start.sh"]

COPY ./app /app
RUN mkdir -p running_data/logs running_data/uploads running_data/data 
#ENV WORKERS_PER_CORE="0.5"
#RUN echo "uwsgi_read_timeout 300s; uwsgi_connect_timeout 300s; uwsgi_next_upstream_timeout 300s; uwsgi_send_timeout 300s;" > /etc/nginx/conf.d/custom_timeout.conf
