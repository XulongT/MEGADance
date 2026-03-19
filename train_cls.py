import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from utils.metric import CLSMetric
from utils.savefig import save_model
from utils.visual import visual
from utils.utils import seed_everything
from models.CLS.dataloader import createTrainDataset, createEvalDataset
from models.CLS.classification import CLS
from datetime import datetime
from tqdm import tqdm
from utils.adan import Adan
from lion_pytorch import Lion

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    losses = []
    for batch_idx, data in tqdm(enumerate(train_loader)):
        optimizer.zero_grad()
        _, loss = model(data, device)
        loss['total'].backward()
        losses.append(loss['total'])
        optimizer.step()

    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'Train Epoch: {epoch} | Timestep: {current_time} | Loss: {float(sum(losses)/len(losses))}')


def validate(model, device, eval_loader, epoch, root_dir, exp_name):
    model.eval()
    metric = CLSMetric(root_dir)
    losses = []
    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(eval_loader)):
            result, loss = model.inference(data, device)
            losses.append(loss['total'])
            metric.update(result)
            
    save_model(model, epoch, exp_name)
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'Eval Epoch: {epoch} | Timestep: {current_time} | Loss: {float(sum(losses)/len(losses))}')
    metric.result()


def main():

    seed_everything(seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32

    root_dir = './data/FineDance'
    exp_name = 'cls'

    print('Loading Data')
    print('Construct Training Data')
    train_loader = createTrainDataset(root_dir=root_dir, batch_size=batch_size)
    print('Construct Eval Data')
    eval_loader = createEvalDataset(root_dir=root_dir, batch_size=batch_size)

    model_args = {'device': device}
    model = CLS(**model_args).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-5, betas=[0.5, 0.999])
    scheduler = StepLR(optimizer, step_size=5, gamma=0.33)
    # optimizer = Adan(model.parameters(), lr=4e-4, weight_decay=0.02)
    # optimizer = Lion(model.parameters(), lr=3e-5, betas=(0.9, 0.999))

    print('Start Training!')
    for epoch in range(1, 31):
        train(model, device, train_loader, optimizer, epoch)
        validate(model, device, eval_loader, epoch, root_dir, exp_name)
        scheduler.step()

if __name__ == '__main__':
    main()
