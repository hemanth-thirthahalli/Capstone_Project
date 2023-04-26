# Here we using YOLOv3
# Instance segmentation (YOLOv3) to count the number of people using its pre-trained weights with TensorFlow and OpenCV in python
# using Mask-RCNN with PixelLib and Python

#opencv-python, pixllib, tensorflow, tensorflow-gpu

import pixellib
from pixellib.instance import instance_segmentation
import cv2

# model_path = 'mask_rcnn_coco.h5'
# segmentation_model.load_model(model_path)

segmentation_model = instance_segmentation()
segmentation_model.load_model('mask_rcnn_coco.h5')

cap = cv2.VideoCapture(0)

while cap.isOpened() :
    res, frame = cap.read()
    
    res = segmentation_model.segmentFrame(frame,show_bboxes=True)
    image = res[1]
    
    cv2.imshow('Instance Segmentation', image)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()