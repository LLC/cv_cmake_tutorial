FROM ubuntu:20.04

WORKDIR /app
COPY . .

# Localtime
ENV TZ=Asia/Taipei
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update && apt install -y build-essential git cmake protobuf-compiler make
RUN DEBIAN_FRONTEND="noninteractive" apt install -y libopencv-dev
RUN mkdir build && cd build && cmake .. && make
RUN /app/build/train1
# RUN /app/build/test1 /app/data/test_data/3.png

ENTRYPOINT ["/app/build/test1"]