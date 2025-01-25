FROM ubuntu:latest
FROM python:3.10
RUN apt update && apt install -y curl
RUN apt install -y fish git 
RUN pip install tensorflow==2.15.0 -v --break-system-packages
RUN git clone https://github.com/davisking/dlib.git
RUN apt install -y cmake
WORKDIR dlib
RUN pip install git+https://github.com/davisking/dlib.git --break-system-packages
WORKDIR /blind-eyes
RUN git clone https://github.com/MBUYt0n/blind-eye-dealers.git
WORKDIR blind-eye-dealers
RUN mkdir models
WORKDIR models
RUN apt install -y wget
RUN wget https://raw.githubusercontent.com/ChiragSaini/facial-emotion-detector/master/emotion_little_vgg_2.h5
RUN wget https://github.com/yu4u/age-gender-estimation/releases/download/v0.5/weights.28-3.73.hdf5
RUN pip install ultralytics torch opencv-python lapx paddlepaddle paddleocr --break-system-packages
RUN pip cache purge
RUN pip install tensorflow==2.15.0 -v --break-system-packages
WORKDIR /
RUN rm -rf dlib
RUN rm -rf root/.cache
CMD ["fish"]
