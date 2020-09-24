# Multi-class Yolov5 + Deep Sort with PyTorch


![](nba_inf.gif)

## Introduction

This repository is modified from mikel-brostrom/Yolov5_DeepSort_Pytorch (https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch). I fixed some bugs and extend it to multi-class version.It contains YOLOv5 (https://github.com/ultralytics/yolov5) and Deep Sort (https://github.com/ZQPei/deep_sort_pytorch). The deep sort model in this repository was only trained by pedestrians.

## Requirements

Python 3.6 or later with all requirements.txt dependencies installed, including torch>=1.6. To install run:

`pip install -U -r requirements.txt`

All dependencies are included in the associated docker images. Docker requirements are: 
- `nvidia-docker`
- Nvidia Driver Version >= 440.44

Alternatively, you can build a docker image by Dockerfile supplied here if you use Centos7.
- `sudo docker pull nvidia/cuda:10.1-cudnn7-devel-centos7`
- `sudo docker build -t [image_name] .`
- `sudo docker run --runtime=nvidia --name [container_name] --shm-size [8G] -t -i [image_name:tag] /bin/bash`

## Download Weights

- Yolov5 pedestrian weight from https://drive.google.com/file/d/1BsWywxaQtuz2Tq3i0M3qFsscZC6a18u8/view?usp=sharing. Place the downlaoded `.pt` file under `yolov5/weights/`
- Yolov5 nba weight from https://drive.google.com/file/d/12qDKovSi9PRdY-77zFJx_7gi41zE3BJY/view?usp=sharing. Place the downlaoded `.pt` file under `yolov5/weights/`. It was trained by a very small dataset.
- Deep sort weights from https://drive.google.com/file/d/18qIFaoPWu4OFiH1kO2JiJ2Lq2D3lhXYY/view?usp=sharing. Place ckpt.t7 file under`deep_sort/deep/checkpoint/`

## Download Sample Video

- Sample nba video from https://drive.google.com/file/d/19ESDqwvO5LRQQ5nQqgjxwsD85eFsW0pp/view?usp=sharing

## Tracking

Tracking can be run on most video formats. Results are saved to ./inference/output.

```bash
python3 track.py --source nba.mp4 --weights nba.pt --device ...
```

- Video:  `--source file.mp4`
- Webcam:  `--source 0`
- RTSP stream:  `--source rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa`
- HTTP stream:  `--source http://wmccpinetop.axiscam.net/mjpg/video.mjpg`

## Train Yolov5
- Put your images in dataset/images and annotations(in PASCAL format) in dataset/annotations.
- Modify yolov5/data/data.yaml
- `python3 label_split.py`
- `cd yolov5`
- `CUDA_VISIBLE_DEVICE=... python3 train.py --img 640 --batch 16 --epochs 500 --data ./data/data.yaml --cfg ./models/yolov5s.yaml --weights weights/nba.pt`

## Reference

For more details, you can check three orgin repositories.
- Simple Online and Realtime Tracking with a Deep Association Metric
https://arxiv.org/abs/1703.07402
- YOLOv4: Optimal Speed and Accuracy of Object Detection
https://arxiv.org/pdf/2004.10934.pdf
