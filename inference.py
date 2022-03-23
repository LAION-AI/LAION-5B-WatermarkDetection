import torch
import torch.nn as nn
import torchvision.transforms as T
import webdataset as wds
import pandas as pd
import timm
import mmh3
import os
from braceexpand import braceexpand
from data import create_webdataset
from tqdm import tqdm

def inference(device, args):
    """
    Load the model, initialize DDP, and run inference.
    """
    WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    NODE_RANK = int(os.environ['NODE_RANK'])
    RANK = NODE_RANK * torch.cuda.device_count() + device

    model, transforms = load_model(device)

    # Get webdataset
    urls = list(braceexpand(args.urls))[NODE_RANK::WORLD_SIZE]
    dataset = create_webdataset(
        urls,
        transforms,
        enable_metadata=True,
    ).batch(args.batch_size)
    dataloader = wds.WebLoader(
        dataset, batch_size=None, shuffle=False, num_workers=8,
    )
    dataloader.num_batches = args.num_samples // (WORLD_SIZE * args.batch_size)
    dataloader.num_samples = dataloader.num_batches * (WORLD_SIZE * args.batch_size)

    # Run inference
    current_samples = []
    for batch in dataloader:
        img = batch['image_tensor'].to(device)
        with torch.no_grad():
            out = model(img)
            out = torch.nn.functional.softmax(out, dim=1)
        current_samples.extend(
            [[out[i][0].item(),
            out[i][1].item(),
            compute_hash(batch['metadata']['url'][i], batch['text'][i])]
            for i in range(len(out))])

        # Save current samples to parquet
        if len(current_samples) >= int(1e6):  
            df = pd.DataFrame(current_samples, columns=['P Watermark', 'P Clean', 'hash'])
            df.to_parquet(f'{WORLD_RANK}.parquet')
            current_samples = []            
    df = pd.DataFrame(current_samples, columns=['P Watermark', 'P Clean', 'hash'])
    df.to_parquet(f'{WORLD_RANK}.parquet')


def load_model(device):
    """
    Loads model.pt into a pretrained timm model.
    """
    transforms = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    model = timm.create_model('efficientnet_b3a', pretrained=False, num_classes=2)
    model.classifier = nn.Sequential(
        nn.Linear(in_features=1536, out_features=625),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(in_features=625, out_features=256),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=2)
    )

    # Load model weights
    state_dict = torch.load('./model.pt')['weights']
    model.load_state_dict(state_dict)
    model.eval().to(device)

    return model, transforms

def compute_hash(url, text):
  if url is None:
    url = ''

  if text is None:
    text = ''
  
  total = (url + text).encode("utf-8")
  return mmh3.hash64(total)[0]