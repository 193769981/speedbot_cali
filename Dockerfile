FROM ubuntu:18.04
MAINTAINER Zhenwei Bian version: 1.0.0
RUN  sed -i s@/archive.ubuntu.com/@/mirrors.tuna.tsinghua.edu.cn/@g /etc/apt/sources.list
RUN  sed -i s@/security.ubuntu.com/@/mirrors.tuna.tsinghua.edu.cn/@g /etc/apt/sources.list
SHELL ["/bin/bash", "-c"]
VOLUME /data
RUN apt-get update && \
    apt-get install -y git python3 python3-pip libgl1-mesa-dev vim && \
    pip3 install -i https://mirrors.aliyun.com/pypi/simple/ --upgrade pip
RUN pip3 install -i https://mirrors.aliyun.com/pypi/simple/ --no-cache-dir scipy==1.5.2 
RUN pip3 install -i https://mirrors.aliyun.com/pypi/simple/ --no-cache-dir opencv_contrib_python==4.4.0.46
RUN pip3 install -i https://mirrors.aliyun.com/pypi/simple/ --no-cache-dir matplotlib==3.3.2
RUN pip3 install -i https://mirrors.aliyun.com/pypi/simple/ --no-cache-dir transforms3d==0.3.1
RUN pip3 install -i https://mirrors.aliyun.com/pypi/simple/ --no-cache-dir numpy==1.19.2
RUN pip3 install -i https://mirrors.aliyun.com/pypi/simple/ --no-cache-dir dt_apriltags==3.1.1
RUN pip3 install -i https://mirrors.aliyun.com/pypi/simple/ --no-cache-dir pupil_apriltags==1.0.4
RUN pip3 install -i https://mirrors.aliyun.com/pypi/simple/ --no-cache-dir PyYAML==5.3.1
RUN pip3 install -i https://mirrors.aliyun.com/pypi/simple/ --no-cache-dir llvmlite==0.35.0
RUN pip3 install -i https://mirrors.aliyun.com/pypi/simple/ --no-cache-dir numba==0.52.0
RUN pip3 install -i https://mirrors.aliyun.com/pypi/simple/ --no-cache-dir progressbar==2.5
RUN pip3 install -i https://mirrors.aliyun.com/pypi/simple/ --no-cache-dir QtAwesome==0.5.7
RUN pip3 install -i https://mirrors.aliyun.com/pypi/simple/ --no-cache-dir qtconsole==4.5.1
RUN pip3 install -i https://mirrors.aliyun.com/pypi/simple/ --no-cache-dir QtPy==1.8.0
RUN apt-get -y install language-pack-zh-hans  && \
 localedef -c -f UTF-8 -i zh_CN zh_CN.utf8 && \
 apt-get -y install libxcb-xinerama0

ENV LC_ALL zh_CN.UTF-8
# RUN git clone https://git.speedbot.net/weiyi/auto_calibrition_python.git
# RUN cd auto_calibrition_python
# RUN chmod 777 start.sh 
# RUN ./start.sh > result.log &

# RUN echo "#!/bin/bash
# cd /opt
# PhoXiControl &
# cd ~
# cd qt_calibration/dkqt_calibration/build
# ./main" > start.sh
# RUN chmod 777 start.sh
# RUN ./start.sh



