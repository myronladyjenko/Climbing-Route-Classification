# Bouldering Route Classification

**Note:** Commits done from a different person's computer. 

## Setup

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Overview
- The YoloV11 folder contains all relevant files needed for the Yolo model. Inside ./src you can find the files needed for training and model testing. You will need to change image names and such in the files as it is not passed by commandline
- The Mask-R-CNN folder contains all relevant files needed for the Mask-R-CNN model
- hold_clustering.py is the file that takes detections in a yolo format and classifies them by route
