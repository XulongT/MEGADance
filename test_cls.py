import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from utils.metric import CLSMetric
from utils.savefig import save_model
from utils.visual import visual
from utils.utils import seed_everything
from models.CLS.dataloader import createEvalDataset
from models.CLS.classification import CLS
from datetime import datetime
from tqdm import tqdm
from utils.adan import Adan
from lion_pytorch import Lion
from utils.load_model import load_model

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
    exp_name = 'CLS'
    epoch = 30

    print('Loading Data')
    print('Construct Eval Data')
    eval_loader = createEvalDataset(root_dir=root_dir, batch_size=batch_size)

    model_args = {'device': device}
    model = CLS(**model_args).to(device)
    model = load_model(model, exp_name, epoch)

    validate(model, device, eval_loader, epoch, root_dir, exp_name)

if __name__ == '__main__':
    main()
