FROM python:3.8-slim
LABEL maintainer="Tien Thien <tienthienhd@gmail.com>"

# install build utilities
RUN apt-get update \
    && apt-get install -y build-essential cmake libgl1-mesa-glx libgtk2.0-dev

COPY requirements.txt .

# install dependencies
RUN pip install -r requirements.txt

# Remove build-essential
RUN apt-get autoremove build-essential cmake -y

COPY . /workspace

# Set the working directory for container
WORKDIR /workspace

CMD ["uwsgi", "face_lvt.ini"]