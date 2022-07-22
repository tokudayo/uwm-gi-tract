import cv2
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torchvision.transforms import Resize, ToTensor

from unet import Unet


def load_img(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = np.tile(img[...,None], [1, 1, 3]) # gray to rgb
    img = img.astype('float32') # original is uint16
    mx = np.max(img)
    if mx:
        img/=mx # scale image to [0, 1]
    return img

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    img_path = './img/t1.png'
    weight_path = './best_f2.pt'

    transform = torchvision.transforms.Compose([
        ToTensor(),
        Resize((224, 224)),
    ])

    img_orig = load_img(img_path)
    h, w, _ = img_orig.shape
    img = transform(img_orig)
    img = img.unsqueeze(0).to(device)

    # Define and load state dict
    model = Unet(encoder_name='efficientnet-b0',      # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=3,        # model output channels (number of classes in your dataset)
                activation=None).to(device)
    model.load_state_dict(torch.load(weight_path))

    # Predict
    model.eval()
    with torch.no_grad():
        output = model.predict(img)
        mask = torch.nn.Sigmoid()(output)
        mask = mask.squeeze(0).cpu().numpy().transpose(1, 2, 0)

    # Visualize mask
    mask = mask * 255
    mask = cv2.resize(mask, (w, h))
    mask = mask.astype('uint8')
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    plt.axis('off')
    ax1.imshow(mask)
    ax2.imshow(img_orig)
    ax3.imshow(mask*0.3 + img_orig*0.7)
    plt.show()
