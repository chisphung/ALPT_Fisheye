#!/bin/bash

yolo detect train \
  model=yolo11m.yaml \
  data=dataset/dataset.yaml \
  epochs=300 imgsz=768 batch=16 \
  optimizer=AdamW lr0=0.0015 lrf=0.01 cos_lr=True \
  patience=50 \
  project=alpr_runs name=v11m_768_yaml
