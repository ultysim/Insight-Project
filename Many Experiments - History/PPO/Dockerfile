FROM ubuntu:18.04

RUN apt-get update; \
    apt-get install -y software-properties-common; \
    apt-get install python3.7 curl -y; \
    apt-get install python3-pip -y; \
    pip3 install torch; \
    pip3 install numpy gym gym[atari]

RUN pip3 install awscli

RUN echo "hello"

COPY ./ /usr/local/

WORKDIR /usr/local/