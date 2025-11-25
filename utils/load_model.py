import torch
import os
import pickle

output_dir = './output/'
os.makedirs(output_dir, exist_ok=True)
def load_model(model, exp_name, epoch):
    model_path = os.path.join('./output', exp_name, str(epoch), 'Model', 'model.pth')
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'])
    return model



