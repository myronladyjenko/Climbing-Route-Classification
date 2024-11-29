from ultralytics import YOLO
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

NUM_EPOCHS = 10

def on_train_epoch_end(trainer):
    epoch = trainer.epoch
    print(f"Saving model after epoch: {epoch}")
    trainer.save_model()

def on_train_epoch_start(trainer):
    print(trainer)
    epoch = trainer.epoch
    # model = trainer.model
    # yolo_model = YOLO(trainer.model)
    # print(type(yolo_model))
    print(f"Saving model before epoch: {epoch}")
    # yolo_model.save(f"yolo11n_custom_after_epoch_{epoch}.pt")

    # save_path = f"yolo11n_custom_before_epoch_{epoch}.pt"
    trainer.save_model()

def main():
    # pass parameter here
    # model = YOLO('yolo11n.pt')
    # results = model.train(data='./dataset/data.yaml', epochs=1)
    # model.save('yolo11n_custom_trained.pt')

    # # NOT NEEDED? NOT IMPORTANT
    # image = Image.open('test3.png').resize((640, 640), Image.Resampling.LANCZOS)
    # image = np.array(image).transpose(2, 0, 1)  
    # image = np.expand_dims(image, axis=0)


    # trained_model.add_callback('on_train_epoch_end', on_train_epoch_end)
    # trained_model.add_callback('on_train_epoch_start', on_train_epoch_start)

    for epoch in range(1, NUM_EPOCHS + 1): 
        trained_model = YOLO(f'yolo11n_custom_trained_{epoch - 1}.pt')
        # results = trained_model.train(data='./dataset/data.yaml', epochs=NUM_EPOCHS, device='cpu')
        results = trained_model.train(data='./dataset/data.yaml', epochs=1, device=0)
        trained_model.save(f'yolo11n_custom_trained_{epoch}.pt')

    trained_model = YOLO(f'yolo11n_custom_trained_{NUM_EPOCHS}.pt')
    # results = trained_model.predict(source='test3.png', conf=0.90, save=True, save_dir='/Users/myronladyjenko/Desktop/Guelph_School/F24_Guelph/CIS_4780/bouldering-holds-detection/runs/detect/predict')
    print(results)


if __name__ == "__main__":
    # import torch
    # print(torch.version.cuda)
    # print(torch.cuda.is_available())
    main()