FROM ubuntu:latest
MAINTAINER <fouad.mazen@epfl.ch>

ADD . /
WORKDIR /

RUN apt-get update && apt-get install -y python3-pip && \
pip3 install torch

ENTRYPOINT ["python3", "/test.py"]
