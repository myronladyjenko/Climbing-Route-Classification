import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def dominant_color(image, bbox, k=3):
    """
    Find the dominant color in a given bounding box using K-means clustering.
    
    Parameters:
    - image: The image in which to find the dominant color.
    - bbox: A tuple (x, y, w, h) representing the bounding box.
    - k: The number of clusters for K-means (default is 3).
    
    Returns:
    - The RGB value of the dominant color.
    """
    # Crop the image to the bounding box
    x, y, w, h = bbox
    cropped_image = image[y:y+h, x:x+w]
    
    # Reshape the cropped image to a 2D array (pixels, 3 color channels)
    pixels = cropped_image.reshape((-1, 3))
    
    # Apply K-means clustering to find dominant colors
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)
    
    # Get the cluster centers (dominant colors)
    dominant_color = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]
    
    return dominant_color

# Load the image
image = cv2.imread('your_image.jpg')

# Example bounding box (x, y, width, height)
bbox = (50, 50, 200, 200)  # Modify this based on the region of interest in your image

# Find the dominant color
dominant_color = dominant_color(image, bbox)

# Convert the dominant color from BGR to RGB (OpenCV uses BGR by default)
dominant_color_rgb = dominant_color[::-1]

# Display the dominant color
plt.imshow([[dominant_color_rgb]])
plt.axis('off')
plt.show()

print(f"Dominant color in RGB: {dominant_color_rgb}")
