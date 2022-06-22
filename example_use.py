import argparse

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from PIL import Image
from torchvision import transforms as T


preprocessing = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def parse_output(water_sym, clear_sym):
    return f"{'watermark' if water_sym > clear_sym else 'clear'}\n{water_sym:.3f}%w {clear_sym:.3f}%c"


if __name__ == '__main__':
    model = timm.create_model(
        'efficientnet_b3a', pretrained=True, num_classes=2)

    model.classifier = nn.Sequential(
        # 1536 is the orginal in_features
        nn.Linear(in_features=1536, out_features=625),
        nn.ReLU(),  # ReLu to be the activation function
        nn.Dropout(p=0.3),
        nn.Linear(in_features=625, out_features=256),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=2),
    )

    state_dict = torch.load('models/watermark_model_v1.pt.pt')

    model.load_state_dict(state_dict)
    model.eval()

    if torch.cuda.is_available():
        model.cuda()
    
    watermark_im = preprocessing(Image.open('./images/watermark_example.png'))
    clear_im = preprocessing(Image.open('./images/clear_example.png'))

    batch = torch.stack([watermark_im, clear_im])

    with torch.no_grad():
        pred = model(batch)
        syms = F.softmax(pred, dim=1).detach().cpu().numpy().tolist()
        for sym in syms:
            water_sym, clear_sym = sym
            if water_sym > clear_sym:
                # watermark
                pass
            else:
                # clear
                pass
            parse_output(water_sym, clear_sym)
