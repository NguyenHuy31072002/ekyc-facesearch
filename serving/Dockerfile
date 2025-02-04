FROM tensorflow/serving:latest

MAINTAINER Tien Thien <tienthienhd@gmail.com>


COPY . /workspace

WORKDIR /workspace

CMD ["chmod +x ./run.sh"]

CMD ["./run.sh", "server"]
