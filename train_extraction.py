from torch.utils.data import DataLoader
import os
import json
import torch
import torch.backends.cudnn
import argparse
from sampling_function import sample_data
import traceback
import re
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from socket import error as SocketError
import errno
from tqdm import tqdm
from config import system_configs
from model_factory import Network
import sys
# Add the path to your config.py to sys.path
sys.path.append(r'C:\Users\saksh\OneDrive\Desktop\stuffs\Chartreader-with-gpu\db\datasets.py')
# Now you can import system_configs from config.py
# from db.datasets import datasets
from db.datasets import load_datasets
import time
from torch.multiprocessing import Process, Queue
import wandb
#sys.path.append(r'C:\Users\saksh\OneDrive\Desktop\stuffs\Chartreader-with-gpu\config\KPDetection.json')
import json
with open(r'C:\Users\saksh\OneDrive\Desktop\stuffs\Chartreader-with-gpu\config\KPDetection.json', "r") as f:
    configs = json.load(f)
# # Open and load the JSON file
# with open('your_file.json', 'r') as file:
#     data = json.load(file)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# Platform-specific multiprocessing setup
if os.name == 'nt':
    torch.multiprocessing.set_start_method('spawn', force=True)

def prefetch_data(db, queue, sample_data, data_aug):
    ind = 0
    print("Starting data prefetching process...")
    np.random.seed(os.getpid())
    while True:
        try:
            data, ind = sample_data(db, ind, data_aug=data_aug)
            queue.put(data)
        except Exception as e:
            print(f'An error occurred during data prefetching: {e}')
            traceback.print_exc()

def pin_memory(data_queue, pinned_data_queue, sema):
    while True:
        try:
            data = data_queue.get()
            data["xs"] = [x.pin_memory() for x in data["xs"]]
            data["ys"] = [y.pin_memory() for y in data["ys"]]
            pinned_data_queue.put(data)
            if sema.acquire(blocking=False):
                return
        except SocketError as e:
            if e.errno != errno.ECONNRESET:
                raise
            pass

def init_parallel_jobs(dbs, queue, fn, data_aug):
    tasks = [Process(target=prefetch_data, args=(db, queue, fn, data_aug)) for db in dbs]
    for task in tasks:
        task.daemon = True
        task.start()
    return tasks

def train(training_db, validation_db, start_iter=0):
    learning_rate = system_configs.learning_rate
    max_iter = system_configs.max_iter
    pretrained_model = system_configs.pretrain
    val_iter = system_configs.val_iter
    decay_rate = system_configs.decay_rate
    stepsize = system_configs.stepsize
    val_ind = 0

    print("Initializing model...")
    nnet = Network()
    if pretrained_model is not None:
        if not os.path.exists(pretrained_model):
            raise ValueError("The requested pretrained model does not exist.")
        print("Loading pretrained model...")
        nnet.load_pretrained_model(pretrained_model)
        
    if start_iter:
        if start_iter == -1:
            print("Training from latest iter...")
            save_list = os.listdir(system_configs.snapshot_dir)
            save_list = [f for f in save_list if f.endswith('.pkl')]
            save_list.sort(reverse=True, key=lambda x: int(x.split('_')[1][:-4]))
            if len(save_list) > 0:
                target_save = save_list[0]
                start_iter = int(re.findall(r'\d+', target_save)[0])
                learning_rate /= (decay_rate ** (start_iter // stepsize))
                nnet.load_model(start_iter)
            else:
                start_iter = 0
        nnet.set_lr(learning_rate)
        print(f"Starting training from iter {start_iter + 1}, LR: {learning_rate}...")
    else:
        nnet.set_lr(learning_rate)

    total_training_loss = []
    ind = 0
    error_count = 0
    scaler = GradScaler()
    optimizer = nnet.optimizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nnet.to(device)
    best_val_loss = float('inf')

    for iteration in tqdm(range(start_iter + 1, max_iter + 1)):
        try:
            training, ind = sample_data(training_db, ind)
            training_data = []
            for d in training.values():
                if isinstance(d, torch.Tensor):
                    training_data.append(d.to(device))
                elif isinstance(d, list):
                    training_data.append([item.to(device) if isinstance(item, torch.Tensor) else item for item in d])
                else:
                    training_data.append(d)

            optimizer.zero_grad()

            with autocast():
                training_loss = nnet.train_step(*training_data)

            scaler.scale(training_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_training_loss.append(training_loss.item())
        except:
            print('Data extraction error occurred.')
            traceback.print_exc()
            error_count += 1
            if error_count > 10:
                print('Too many extraction errors. Terminating...')
                time.sleep(1)
                break
            continue

        if iteration % 500 == 0:
            avg_training_loss = sum(total_training_loss) / len(total_training_loss)
            print(f"Training loss at iter {iteration}: {avg_training_loss}")
            wandb.log({"train_loss": training_loss.item()})
            total_training_loss = []

        if val_iter and validation_db.db_inds.size and iteration % val_iter == 0:
            validation, val_ind = sample_data(validation_db, val_ind)
            validation_data = []
            for d in validation.values():
                if isinstance(d, torch.Tensor):
                    validation_data.append(d.to(device))
                elif isinstance(d, list):
                    validation_data.append([item.to(device) if isinstance(item, torch.Tensor) else item for item in d])
                else:
                    validation_data.append(d)
            validation_loss = nnet.validate_step(*validation_data)
            wandb.log({"val_loss": validation_loss.item()})
            print(f"Validation loss at iter {iteration}: {validation_loss.item()}")
            if validation_loss < best_val_loss:
                best_val_loss = validation_loss
                print(f"New best validation loss: {best_val_loss.item()}. Saving model...")
                nnet.save_model("best")

        if iteration % stepsize == 0:
            learning_rate /= decay_rate
            nnet.set_lr(learning_rate)

def parse_args():
    parser = argparse.ArgumentParser(description="Train the model with the given configs.")
    parser.add_argument("--cfg_file", dest="cfg_file", help="Name of the configuration file to be used for training.", default="KPDetection", type=str)
    parser.add_argument("--start_iter", dest="start_iter", help="Specify the iter to start training from. Default is 0.", default=0, type=int)
    parser.add_argument("--pretrained_model", dest="pretrained_model", help="Name of the pre-trained model file. Default is 'KPDetection.pkl'.", default="KPDetection.pkl", type=str)
    parser.add_argument("--threads", dest="threads", help="Number of threads to use for data loading. Default is 1.", default=1, type=int)
    parser.add_argument("--cache_path", dest="cache_path", help="Path to cache preprocessed data.", default="./data/cache/", type=str)
    parser.add_argument("--data_dir", dest="data_dir", help="Directory containing the dataset for training. Default is './data'.", default="./data", type=str)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    wandb.init(
        project="ChartLLM-Extraction",
        name="bar only",
        group="grouping",
        notes="Test KP Grouping with Only Bars-Mixed Precision-No Crop or Bump",
        tags=["ChartLLM", "KP Grouping"],
        config=args
    )

    print(f"Training args: {args}")
    cfg_file = os.path.join(system_configs.config_dir, args.cfg_file + ".json")
    with open(cfg_file, "r") as f:
        configs = json.load(f)
    configs["system"]["data_dir"] = args.data_dir
    configs["system"]["cache_dir"] = args.cache_path
    configs["system"]["dataset"] = "Chart"
    file_list_data = os.listdir(args.data_dir)
    configs["system"]["snapshot_name"] = args.cfg_file

    if args.cfg_file == "KPGrouping":
        if args.start_iter == 0:
            configs["system"]["pretrain"] = os.path.join(args.cache_path, 'nnet/KPDetection', args.pretrained_model)
        else:
            configs["system"]["pretrain"] = os.path.join(args.cache_path, 'nnet/KPGrouping', args.pretrained_model)
    else:
        if args.start_iter != 0:
            configs["system"]["pretrain"] = os.path.join(args.cache_path, 'nnet/KPDetection', args.pretrained_model)

    system_configs.update_config(configs["system"])

    system_configs.initialize_dataset()
    
    train_split = system_configs.train_split
    val_split = system_configs.val_split

    print("Loading all datasets...")
    dataset = system_configs.dataset
    threads = args.threads
    print(f"Using {threads} threads.")
    training_db = datasets[dataset](split=train_split)
    validation_db = datasets[dataset](split=val_split)

    print("Beginning training process...")
    train(training_db, validation_db, start_iter=args.start_iter)

    # print("Loading all datasets...")
    # dataset = system_configs.dataset  # Assuming dataset is a string like 'coco' or 'imagenet'
    # threads = args.threads  # Get number of threads from command-line arguments
    # print(f"Using {threads} threads.")
    
    # # Assume datasets[dataset] returns the correct dataset object
    # training_db = datasets[dataset](split=train_split,db_config=configs)
    # validation_db = datasets[dataset](split=val_split,db_config=configs)

    # # Wrap datasets into DataLoader for multi-threaded data loading
    # # train_loader = DataLoader(training_db, batch_size=32, num_workers=threads, shuffle=True)
    # # val_loader = DataLoader(validation_db, batch_size=32, num_workers=threads, shuffle=False)

    # # print("Beginning training process...")

    # # # Call train function with DataLoaders
    # # train(train_loader, val_loader, start_iter=args.start_iter)