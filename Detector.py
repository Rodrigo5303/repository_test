from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo

import cv2
import numpy as np

class Detector:
    def __init__(self, model_type = "OD"):
        self.cfg = get_cfg()

        # Cargar la configuración del modelo y el modelo preentrenado
        if model_type == "OD":
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
        elif model_type == "IS":
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.cfg.MODEL.DEVICE = "cpu"  #cpu or cuda
        self.predictor = DefaultPredictor(self.cfg)

    def onImage(self, imagePath):
        image = cv2.imread(imagePath)
        predictions = self.predictor(image)

        viz = Visualizer(image[:,:,::-1], metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
                         , instance_mode = ColorMode.SEGMENTATION)
        output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))

        cv2.imshow("Result", output.get_image()[:, :, ::-1])
        cv2.waitKey(0)
    def onVideo(self, videoPath):
        cap = cv2.VideoCapture(0)

        if (cap.isOpened() == False):
            print("Error opening video stream or file")
            return

        (success, image) = cap.read()

        while success:
            predictions = self.predictor(image)

            viz = Visualizer(image[:, :, ::-1], metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
                             , instance_mode=ColorMode.SEGMENTATION)
            output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))
            cv2.imshow("Result", output.get_image()[:, :, ::-1])

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            (success, image) = cap.read()

