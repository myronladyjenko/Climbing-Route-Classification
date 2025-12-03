from ultralytics import YOLO
import os
import cv2
import sys
from config import (
    YOLO_DATA_YAML,
    MODELS_DIR,
    DETECTION_OUTPUT_DIR,
    BASE_YOLO_MODEL_n,
    BASE_YOLO_MODEL_s,
    BASE_YOLO_MODEL_m,
    NUM_EPOCHS,
    IMAGE_SIZE,
    DEVICE,
    BASE_MODEL_NAME,
)
from preprocessing import preprocess_image_file

# For batch YOLO object detection (testing)
TEST_DIR = "../test_image_set"
MODEL_PATHS = {
    "m_10": "../models/yolo_hold_detector_m_10/weights/best.pt",
    "s_20": "../models/yolo_hold_detector_s_40/weights/best.pt",
    "s_40": "../models/yolo_hold_detector_s_40/weights/best.pt",
    "n_90": "../models/yolo_hold_detector_n_90/weights/best.pt",
    "n_21": "../models/yolo_hold_detector_n_10/weights/best.pt",
    "n_10": "../models/yolo_hold_detector_n_10/weights/best.pt",
    "n_1":  "../models/yolo_hold_detector_n_1/weights/best.pt",
}

def ensure_directory_exists(path):
    os.makedirs(path, exist_ok=True)


def train_yolo(model_path, model_identification, continue_training = False, upto_num_epochs = NUM_EPOCHS):
    if model_path is None or model_identification is None:
        raise NameError(f"[ERROR] Model path or model identification weren't provided")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"YOLO model not found at {model_path}. Can't perform training")

    results = None
    if continue_training:
        print(f"[INFO] Resuming training from last.pt for model: {model_path} and for upto epochs {upto_num_epochs}")
        model = YOLO(model_path)
        results = model.train(
            resume=True
        )
        print(f"[INFO] Finished resumed training for upto epochs: {upto_num_epochs}")
    else:
        print("[INFO] Starting new training from an existing trained model")
        model = YOLO(model_path)
        results = model.train(
            data=YOLO_DATA_YAML,
            epochs=NUM_EPOCHS,
            imgsz=IMAGE_SIZE,
            device=DEVICE,
            project=MODELS_DIR,
            name=BASE_MODEL_NAME + model_identification,
            exist_ok=True,
        )
        print(f"[INFO] Training of the model {model_path} completed finished.")

    return results


def load_trained_model(model_path):
    if model_path is None:
        print(f"[ERROR] Model path to load wasn't provided")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"YOLO model not found at {model_path}. Train the model and specify a correct path.")
    return YOLO(model_path)


def run_inference_on_image(image_path, model, preprocess = True, output_dir = DETECTION_OUTPUT_DIR):
    # Preprocess image: CLAHE + Gaussing blur
    if preprocess:
        processed_path = preprocess_image_file(image_path)
    else:
        processed_path = image_path

    # perofrm inference using YOLO 
    results = model(processed_path, save=False)

    # parse results and save them 
    for _, r in enumerate(results):
        # save labels into labels directory
        txt_dir_out = os.path.join(output_dir, "labels")
        ensure_directory_exists(txt_dir_out)
        base_name = os.path.splitext(os.path.basename(processed_path))[0]
        txt_file = os.path.join(txt_dir_out, base_name + ".txt")

        # a way to overwrite existing file, otherwise just appends text
        if os.path.exists(txt_file):
            os.remove(txt_file)
        r.save_txt(txt_file, save_conf=True)

        # save images with bboxes into images directory
        visual_dir_out = os.path.join(output_dir, "images")
        ensure_directory_exists(visual_dir_out)
        img = cv2.imread(processed_path)
        h, w, _ = img.shape

        xywhn = r.boxes.xywhn.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        for (xc, yc, bw, bh), conf in zip(xywhn, confs):
            x_center = int(xc * w)
            y_center = int(yc * h)
            width_px = int(bw * w)
            height_px = int(bh * h)

            x1 = int(x_center - width_px / 2)
            y1 = int(y_center - height_px / 2)
            x2 = int(x_center + width_px / 2)
            y2 = int(y_center + height_px / 2)

            # Draw the bounding box 
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # for confidence levels
            # cv2.putText(img, f"{conf:.2f}", (x1, y1 - 5),
            #             cv2.FONT_HERSHEY_PLAIN, 0.5, (255, 0, 0), 1)

        fname = os.path.basename(processed_path)
        base, _ = os.path.splitext(fname)
        out_path = os.path.join(visual_dir_out, f"{base}_bbox.png")
        cv2.imwrite(out_path, img)
        print(f"[INFO] Succesfully Saved results of YOLO detection inference into directory {output_dir}")

    return results

# run multiple inferences for testing of multiple images
def run_inference_batch():
    for model_name, model_path in MODEL_PATHS.items():
        print(f"\n[INFO] Loading model: {model_name} from {model_path}")
        model = load_trained_model(model_path)

        model_out_dir = os.path.join(DETECTION_OUTPUT_DIR, f"inference_{model_name}")
        ensure_directory_exists(model_out_dir)

        for fname in os.listdir(TEST_DIR):
            if fname.lower().endswith((".png", ".jpg")):
                image_path = os.path.join(TEST_DIR, fname)
                print(f"[INFO] Running inference on: {fname} using model {model_name}")

                # Run your inference function; adjust preprocessing for testing if needed
                _ = run_inference_on_image(image_path, model, preprocess=True, output_dir=model_out_dir)
        print(f"[INFO] Finished inference for model '{model_name}'. Results saved in {model_out_dir}\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("[ERROR] Usage: python script.py <w or w/o> <b or s>. Pass 'w' or 'wo' for training or without; pass 'b' or 's' for batch or signle inference")
        exit(0)

    ensure_directory_exists(MODELS_DIR)

    # Setup to train the model, other configurations can be found in the config.py file 
    model_path = os.path.join(MODELS_DIR, "yolo_hold_detector_s_40", "weights", "last.pt")
    model_identification = "n_33"
    if (sys.argv[1] == 'w'):
        # change the base model if needed
        train_yolo(BASE_YOLO_MODEL_n, model_identification)

    if sys.argv[2] == 'b':
        # Inference testing framework
        run_inference_batch()
    else:
        # Single image inference
        # load trained model for inference 
        model_path_to_be_loaded = os.path.join(MODELS_DIR, "yolo_hold_detector_n_21", "weights", "best.pt")
        model = load_trained_model(model_path_to_be_loaded)
        _ = run_inference_on_image("../NOT_USED_raw_images/1.png", model)
