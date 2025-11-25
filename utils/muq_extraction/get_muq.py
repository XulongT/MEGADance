import os
import glob
import torch
import librosa
import pickle
from tqdm import tqdm
from muq import MuQ
import torch.nn.functional as F

def upsample_features(features, original_fps=25, target_fps=30):

    b, t, c = features.shape
    

    target_t = int(t * target_fps / original_fps)  
    

    features = features.permute(0, 2, 1)
    

    upsampled = F.interpolate(
        features,
        size=target_t,
        mode='linear',
        align_corners=False
    )
    

    return upsampled.permute(0, 2, 1)

def extract_muq(a_folder, b_folder):
    os.makedirs(b_folder, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    muq = MuQ.from_pretrained("./Pretrained/MuQ-large-msd-iter", local_files_only=True)
    muq = muq.to(device).eval()
    model = muq

    for wav_path in tqdm(glob.glob(os.path.join(a_folder, "*.wav"))):
        try:

            wav, _ = librosa.load(wav_path, sr=24000)
            wav_tensor = torch.as_tensor(wav).unsqueeze(0).to(device)
            

            with torch.no_grad():
                output = model(wav_tensor, output_hidden_states=True)
            

            last_hidden = output.last_hidden_state
                

            upsampled = upsample_features(last_hidden)[0].cpu().numpy()

            print(wav_path, upsampled.shape)
            

            pkl_path = os.path.join(
                b_folder,
                os.path.basename(wav_path).replace(".wav", ".pkl")
            )
            

            with open(pkl_path, "wb") as f:
                pickle.dump(
                    {"music": upsampled},
                    f,
                )
                
        except Exception as e:
            print(f"error: {wav_path} - {str(e)}")
            continue

if __name__ == "__main__":

    
    input_folder = "./data/FineDance/music"
    output_folder = "./data/FineDance/muq"
    
    process_audio_folder(input_folder, output_folder)