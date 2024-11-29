import sys
import os

config = {
    'source_dir': 'dataset_copy',
    'destination_dir': 'preprocessed_dataset'
}

def preprocess_dataset():
    pass

def create_dir_structre():
    for root, dirs, files in os.walk(config['source_dir']):
        dest_path = os.path.join(config['destination_dir'], os.path.relpath(root, config['destination_dir']))
        os.makedirs(dest_path, exist_ok=True)
    
def main(args: list[str]):
    if len(args) > 0:
        print("Note: This command does not accept any arguments")
        return
    create_dir_structre()
    # preprocess_dataset()

if __name__ == "__main__":
    main(sys.argv[1:])