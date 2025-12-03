# Bouldering Route Classification: Object Detection and Clustering Algorithms

## Setup

Create environment:
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

After finished developing:
```
pip freeze > requirements.txt
deactivate
```

### Development

The code is located in the folder ```src/``` (from the root folder of the project).
Once the development is finished, please run the following commands to save the environment:


To run the inference (```detection.py```) you can do the following:
```
python3 detection.py wo/w b/s
```
The options are:
- ```w```: with training
- ```wo```: without training
- ```b```: batch image inference
- ```s```: single image inference
**Note: the image/directory/model paths might need to be altered in the ```detection.py``` or ```config.py``` depending on the context. 


To ensure that the dataset contains valid data and formated in the bboxes format, we can run ```segment_to_bbox.py```:
```
python3 segment_to_bbox.py
```
**Note: Currently the data is expected to be in: ```COMBINED_DATASET_ROOT``` from the ```config.py```


To run colour and hold clusreting for route detection, do the following:
```
python3 hold_detection.py
```
**Note: The paths are specified in the file itself: ```hold_detection.py```


## Overview
This project focuses on exploring YOLO (You Only Look Once) object detection model as well different clustering algorithms, such as:
1. K-Means 
2. Gaussian Misture Model (GMM)
3. HSV (hue, saturation, value)
4. Mean-Shift
5. DBSCAN
6. HDBSCAN 
7. Agglomerative Clustering

The project uses first four to best identify the colour of the hold inside a bounding box, after the YOLO hold detection inference (i.e. inside the bounding box). Next, we use DBSCAN, HDBSCAN and Agglomerative clustering to cluster holds of the same color into one route. The algorithms are compared for results accuracy of detection and route identification. 

The code can be found in the ```src/``` folder from the root directory of the project. 
The report can be found at the root level of the directory: ```CIS4020_FinalReport```.
