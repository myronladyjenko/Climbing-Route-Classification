import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import os
import shutil

def rgb_distance(color1, color2):
    # Convert the colors to numpy arrays to facilitate calculation
    color1 = np.array(color1)
    color2 = np.array(color2)
    
    background_colour = np.array([170, 157, 145])

    if np.linalg.norm(color1 - background_colour) > np.linalg.norm(color2 - background_colour):
        return color1
    else:
        return color2
    
def extract_roi(image, detection):
    # Image dimensions
    image_height, image_width, _ = image.shape
    
    # Extract values from the detection dictionary
    x_center = detection['x_center']
    y_center = detection['y_center']
    width = detection['width']
    height = detection['height']
    
    # Denormalize the coordinates (convert to pixel values)
    x_center_pixel = int(x_center * image_width)
    y_center_pixel = int(y_center * image_height)
    width_pixel = int(width * image_width)
    height_pixel = int(height * image_height)
    
    # Calculate top-left corner and bottom-right corner coordinates
    x1 = int(x_center_pixel - width_pixel / 2)
    y1 = int(y_center_pixel - height_pixel / 2)
    x2 = int(x_center_pixel + width_pixel / 2)
    y2 = int(y_center_pixel + height_pixel / 2)
    
    # Extract the region of interest (ROI)
    roi = image[y1:y2, x1:x2]
    
    return roi

def dominant_color(image, detection, k=2, plotting=False):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    roi = extract_roi(image, detection)
    pixels = roi.reshape((-1, 3))
    
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)

    dominant_colors = kmeans.cluster_centers_.astype(int)
    # print(f'Dominant Colors: {dominant_colors}')
    rgb_dist = rgb_distance(dominant_colors[0], dominant_colors[1])
    color_image = np.full((100, 100, 3), rgb_dist, dtype=np.uint8)
    # print(f'RGB Distance: {rgb_dist}')

    labels = kmeans.predict(pixels)
    # print(f'Labels: {labels}')


    new_pixels = dominant_colors[labels]
    new_image = new_pixels.reshape(roi.shape)


    # Plot Images
    if plotting:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(new_image)
        plt.subplot(1, 3, 2)
        plt.imshow(roi)
        plt.subplot(1, 3, 3)
        plt.imshow(color_image)
        plt.show()
        # plt.close()

    
    return rgb_dist

def group_colours(colours):
    colours = np.array([colour[0] for colour in colours])

    dbscan = DBSCAN(eps=20, min_samples=2).fit(colours)

    labels = dbscan.labels_

    return labels

def draw_yolo_bounding_box(image, detection):
    # Get the image dimensions
    image_height, image_width, _ = image.shape
    
    # Extract values from the dictionary
    class_id = detection['class']
    x_center = detection['x_center']
    y_center = detection['y_center']
    width = detection['width']
    height = detection['height']
    
    # Denormalize the coordinates (convert to pixel values)
    x_center_pixel = int(x_center * image_width)
    y_center_pixel = int(y_center * image_height)
    width_pixel = int(width * image_width)
    height_pixel = int(height * image_height)

    # Calculate top-left corner and bottom-right corner coordinates
    x1 = int(x_center_pixel - width_pixel / 2)
    y1 = int(y_center_pixel - height_pixel / 2)
    x2 = int(x_center_pixel + width_pixel / 2)
    y2 = int(y_center_pixel + height_pixel / 2)

    # Draw the rectangle (bounding box) on the image
    color = (0, 255, 0)  # Green color (BGR format)
    thickness = 2  # Thickness of the bounding box
    image = cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    # Optionally, you can also draw the class label text near the box
    label = f"Class {class_id}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, label, (x1, y1 - 10), font, 0.5, (255, 255, 255), 2)

    return image

def classify_holds(hold_colours, labels):
    filename = './classified_images/test3.png'
    max_label = max(labels)
    detections = load_yolo_detections('./classified_images/test3_detection_results.txt')
    os.makedirs('./classified_images/classified_holds', exist_ok=True)

    for i in range(0, max_label + 1):
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for index, label in enumerate(labels):
            if label != i:
                continue
            
            detection = detections[index]

            image = draw_yolo_bounding_box(image, detection)
                
        plt.imshow(image)
        plt.show()
        plt.close()
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f'./classified_images/classified_holds/test3_class{i}.png', image)
        print(f'./classified_images/classified_holds/test3_class{i}.png')
        # cv2.imshow('image', image)
        # cv2.waitKey(0)
        # break


def load_yolo_detections(file_path):
    detections = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            
            # Convert to numerical values
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
            
            # Append detection as a dictionary
            detections.append({
                'class': class_id,
                'x_center': x_center,
                'y_center': y_center,
                'width': width,
                'height': height,
            })
    return detections
        

def sort_holds_by_colour(hold_colours, labels):
    destination_dir = "./classified_images/0"

    # Process each image and class
    for label, hold_colour in zip(labels, hold_colours):
        # Create the class folder if it doesn't exist
        class_folder = os.path.join(destination_dir, str(label))
        os.makedirs(class_folder, exist_ok=True)

        shutil.copy(destination_dir + '/' + hold_colour[0], class_folder)


if __name__ == "__main__":
    colours = []
    detections = load_yolo_detections('./classified_images/test3_detection_results.txt')
    id = 0

    for detection in detections:
        detection['id'] = id
        image = cv2.imread('./classified_images/test3.png')
        # colour = dominant_color(image, detection, plotting=True)
        colour = dominant_color(image, detection)
        colours.append((colour, id))
        id += 1

    labels = group_colours(colours)

    classify_holds(colours, labels)
    # sort_holds_by_colour(colours, labels)

