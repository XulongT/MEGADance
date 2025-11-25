import torch
import torch.optim as optim
from utils.metric import Metric
from utils.savefig import savefig
from utils.utils import seed_everything
from models.GPT.dataloader import createDemoDataset
from models.GPT.trainer import Trainer
from datetime import datetime
from tqdm import tqdm
from utils.load_model import load_model
import argparse

def main(root_dir, exp_name, epoch):
    seed_everything(seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Loading Data')
    print('Construct Demo Data')
    eval_loader = createDemoDataset(root_dir=root_dir, batch_size=16)
    print('Start Demo!')

    model_args = {'device': device}
    model = Trainer(**model_args).to(device)
    model = load_model(model, exp_name, epoch)

    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(eval_loader)):
            model.demo(data, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demo script with hyperparameters')
    parser.add_argument('--root_dir', type=str, default='./demo/4',
                        help='Root directory for the demo data')
    parser.add_argument('--exp_name', type=str, default='GPT',
                        help='Experiment name')
    parser.add_argument('--epoch', type=int, default=50,
                        help='Epoch number for model loading')

    args = parser.parse_args()

    main(args.root_dir, args.exp_name, args.epoch)
