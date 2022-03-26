from ast import arg
import imp
from numpy import append
from rsa import compute_hash
import torch
import torch.nn as nn
import torchvision.transforms as T
import webdataset as wds
import pandas as pd
import timm
import mmh3
import os
import json
import fsspec
import uuid
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
    fs, output_folder = fsspec.core.url_to_fs(args.bucket_dir,
        client_kwargs={"endpoint_url":"https://bucket.vpce-06aadfc9fc5aabd58-bv32itci.s3.us-east-1.vpce.amazonaws.com/"})
    output_folder += '/'

    # Get webdataset
    urls = list(braceexpand(args.urls))[RANK::WORLD_SIZE]
    dataset = create_webdataset(
        urls,
        transforms,
        enable_metadata=True,
    ).batch(args.batch_size)
    dataloader = wds.WebLoader(
        dataset, batch_size=None, shuffle=False, num_workers=8, collate_fn=collate
    )
    dataloader.num_batches = args.num_samples // (WORLD_SIZE * args.batch_size)
    dataloader.num_samples = dataloader.num_batches * (WORLD_SIZE * args.batch_size)

    # Run inference
    current_samples = []
    if device == 0:
        pbar = tqdm(total=dataloader.num_samples)
    for batch in dataloader:
        img = batch['image_tensor'].to(device)
        with torch.no_grad():
            out = model(img)
            out = torch.nn.functional.softmax(out, dim=1)
        current_samples.extend(statistics_to_array(out, batch))

        # Save current samples to parquet
        if len(current_samples) >= int(1e6):  
            df = pd.DataFrame(current_samples, columns=['P Watermark', 'P Clean', 'hash'])
            with fs.open(os.path.join(output_folder, str(uuid.uuid4())) + '.parquet', 'wb') as f:
                df.to_parquet(f)
            current_samples = []            
        if device == 0:
            pbar.update(WORLD_SIZE * args.batch_size)
    df = pd.DataFrame(current_samples, columns=['P Watermark', 'P Clean', 'hash'])
    with fs.open(os.path.join(output_folder, str(uuid.uuid4())) + '.parquet', 'wb') as f:
        df.to_parquet(f)
    if device == 0:
        pbar.close()


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

def collate(arr):
    keys = arr[0].keys()
    ret_dict = {}
    for k in keys:
        ret_dict[k] = [x[k] for x in arr]
        if k == 'image_tensor':
            ret_dict[k] = torch.stack(ret_dict[k])
    
    return ret_dict

def statistics_to_array(out, batch):
    output = []
    for i in range(len(batch['image_tensor'])):
        output.append([
            out[i][0].item(),
            out[i][1].item(),
            compute_hash(json.loads(batch['metadata'][i]).get('url'), batch['text'][i])
        ])
    return output

def compute_hash(url, text):
  if url is None:
    url = ''

  if text is None:
    text = ''
  
  total = (url + text).encode("utf-8")
  return mmh3.hash64(total)[0]