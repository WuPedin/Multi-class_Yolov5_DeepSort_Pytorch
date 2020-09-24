# Base image
FROM nvidia/cuda:10.1-cudnn7-devel-centos7


# Install packages specified in requirements_yum.txt and requirements.txt
COPY requirements.txt ./
COPY requirements_yum.txt ./
COPY Yolov5_DeepSort_Pytorch ./Yolov5_DeepSort_Pytorch
RUN yum install $(cat requirements_yum.txt) ./
RUN pip3 install --upgrade pip
RUN pip3 install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install -r requirements.txt

# Make port available outside this container
EXPOSE 5000  

# Working directory for container
WORKDIR Multi-class_Yolov5_DeepSort_Pytorch

