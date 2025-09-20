
import argparse
import os
import sys

# python train.py --batch-size 8 --img 1280 1280 --data ../../dataset/visdrone_fisheye8k.yaml --cfg models/yolor-w6.yaml --weights './yolor-w6-paper-555.pt' --device 0 --name yolor_w6 --hyp hyp.scratch.1280.yaml --epochs 250
def train(model_type, batch, imgsz, yaml_file, device, epochs, weights, hyp, name):
    if model_type == "ultralytics":
        from model.yolo.ultralytics.models.yolo.model import YOLO
        YOLO(name).train(
            batch_size=batch,
            imgsz=imgsz,
            data=yaml_file,
            device=device,
            epochs=epochs,
            weights=weights,
            hyp=hyp,
            name=name,
        )
    else:
        print("model_type error, only support ultralytics now.")
        exit(0)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='ultralytics', help='model type, now only support ultralytics')
    parser.add_argument('--weights', nargs='+', type=str, default='yolo11x.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default='dataset/dataset.yaml', help='file/dir')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[1280], help='inference size h,w')
    parser.add_argument('--batch', type=int, default=8, help='total batch size for all GPUs')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print(opt)
    train(
        model_type=opt.model_type,
        batch=opt.batch,
        imgsz=opt.imgsz,
        yaml_file=opt.source,
        device=opt.device,
        epochs=300,
        weights=opt.weights,
        hyp='utils/hyp.scratch.1280.yaml',
        name='alpr_yolor_w6',
    )