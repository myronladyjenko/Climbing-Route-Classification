from ultralytics import YOLO
import os
import shutil

os.makedirs('./classified_images', exist_ok=True)

# Change for different test images
input_image = 'test3.png' 

# Change based on model you want to load
trained_model = YOLO(f'full_models/yolo11n_custom_trained_{5}.pt')

result = trained_model(input_image)
for r in result:
    shutil.copy(input_image, 'classified_images/')
    r.save(filename=f'classified_images/{input_image[:-4]}_bbox.png')
    r.save_crop('classified_images', file_name='hold.png')
    r.save_txt(f'classified_images/{input_image[:-4]}_detection_results.txt', save_conf=False)
