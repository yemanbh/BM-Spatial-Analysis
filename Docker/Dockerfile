FROM tensorflow/tensorflow:2.2.0-gpu
MAINTAINER yeman.hagos@icr.ac.uk

RUN apt-get update && apt-get install -y --no-install-recommends \
                                        build-essential \
                                        python3-pip \
                                        python3-setuptools \
                                        python3-dev \
                                        vim

# install libiraries for opencv
RUN apt-get update && apt install -y libsm6 \
                                     libxext6 \
                                     libxrender-dev
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt