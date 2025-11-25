import torch
import torch.optim as optim
from utils.metric import Metric
from utils.savefig import savefig
from utils.utils import seed_everything
from models.GPT.dataloader import createEvalDataset
from models.GPT.trainer import Trainer
from datetime import datetime
from tqdm import tqdm
from utils.load_model import load_model

def validate(model, device, eval_loader, epoch, root_dir, exp_name):
    # model.eval()
    losses = []
    metric = Metric(root_dir)
    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(eval_loader)):
            result, loss = model.inference(data, device)
            losses.append(loss['total'])
            metric.update(result)
            savefig(model, result, epoch, exp_name)
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'Eval Epoch: {epoch} | Timestep: {current_time} | Loss: {float(sum(losses)/len(losses))}')
    metric.result()

def main():

    seed_everything(seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128

    root_dir = './data/FineDance'
    exp_name = 'GPT'
    epoch = 50

    print('Loading Data')
    print('Construct Eval Data')
    eval_loader = createEvalDataset(root_dir=root_dir, batch_size=batch_size)
    print('Start Training!')

    model_args = {'device': device}
    model = Trainer(**model_args).to(device)
    model = load_model(model, exp_name, epoch)
        
    validate(model, device, eval_loader, epoch, root_dir, exp_name)

if __name__ == '__main__':
    main()
