from ultralytics import YOLO
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

NUM_EPOCHS = 100

def on_train_epoch_end(trainer):
    epoch = trainer.epoch
    print(f"Saving model after epoch: {epoch}")
    trainer.save_model()

def on_train_epoch_start(trainer):
    print(trainer)
    epoch = trainer.epoch
    print(f"Saving model before epoch: {epoch}")
    trainer.save_model()

def main():
    for epoch in range(6, NUM_EPOCHS + 1): 
        trained_model = YOLO(f'full_models/yolo11n_custom_trained_{epoch - 1}.pt')
        results = trained_model.train(data='./dataset/data.yaml', epochs=1, device=0) # Currently only training on GPU
        trained_model.save(f'full_models/yolo11n_custom_trained_{epoch}.pt')

    trained_model = YOLO(f'full_models/yolo11n_custom_trained_{NUM_EPOCHS}.pt')
    print(results)


if __name__ == "__main__":
    main()