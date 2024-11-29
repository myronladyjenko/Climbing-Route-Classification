from ultralytics import YOLO
import os
import shutil


input_image = 'test3.png'
trained_model = YOLO(f'eric_laptop_models/yolo11n_custom_trained_{15}.pt')
result = trained_model(input_image)
for r in result:
    shutil.copy(input_image, 'classified_images/')
    r.save(filename=f'classified_images/{input_image[:-4]}_bbox.png')
    r.save_crop('classified_images', file_name='hold.png')
    r.save_txt(f'classified_images/{input_image[:-4]}_detection_results.txt', save_conf=False)
# print(result)
# results = trained_model.predict(source=f'test3.png', conf=conf, save=True, save_dir='./testing/')
# for r in results:
#     print(r.boxes.data)
#     json_result = r.to_json()
#     print(json_result)
# results = trained_model.predict(source='test3.png', conf=0.80, save=True, save_dir='./testing/')
# print(results)
# boxes = results.xywh[0]  # The first image's results (xywh format: [x_center, y_center, width, height])
# labels = results.names  # Class names (e.g., 'person', 'car', etc.)
# confidences = results.conf[0]  # Confidence scores for the bounding boxes

# # Example: printing bounding boxes
# for box, label, conf in zip(boxes, labels, confidences):
#     x_center, y_center, width, height = box
#     print(f"Detected: {label}, Confidence: {conf}, Box: {x_center}, {y_center}, {width}, {height}")
