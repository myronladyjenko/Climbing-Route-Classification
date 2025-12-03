import os
import cv2
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo

def format_to_yolo(boxes, classes, image_height, image_width):
    yolo_bbs = []
    
    count = 0
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        class_id = classes[count]
        count += 1

        center_x = (x_min + x_max) / 2 / image_width
        center_y = (y_min + y_max) / 2 / image_height
        width = (x_max - x_min) / image_width
        height = (y_max - y_min) / image_height

        yolo_bbs.append(f"{class_id} {center_x:.7f} {center_y:.7f} {width:.7f} {height:.7f}")
    
    with open("yolo_data", "w") as f:
        f.write("\n".join(yolo_bbs))

def main():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = os.path.join("output", "model_final_2000.pth") 
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  
    cfg.DATASETS.TEST = ("dataset_valid", )  
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2 

    predictor = DefaultPredictor(cfg)
    image_path = "sample_test_images/good2.png"
    image = cv2.imread(image_path)
    image_height, image_width, _ = image.shape
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return

    outputs = predictor(image)
    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TEST[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    instances = outputs["instances"].to("cpu")
    boxes = instances.pred_boxes.tensor.numpy()         
    classes = instances.pred_classes.numpy()   
    format_to_yolo(boxes, classes, image_height, image_width)

    output_image_path = "predicted.jpg"
    cv2.imwrite(output_image_path, out.get_image()[:, :, ::-1])

if __name__ == "__main__":
    main()