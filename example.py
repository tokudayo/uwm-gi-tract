import cv2
import numpy as np
import torch
import torchvision
import os
from matplotlib import pyplot as plt
from torchvision.transforms import Resize, ToTensor
import base64
from kafka import KafkaConsumer, KafkaProducer
import json

from unet import Unet


def readb64(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    return nparr


def load_img(nparr):
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    img = np.tile(img[..., None], [1, 1, 3])  # gray to rgb
    img = img.astype('float32')  # original is uint16
    mx = np.max(img)
    if mx:
        img /= mx  # scale image to [0, 1]
    return img


def toBase64(img):
    retval, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')


def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalized data
    """
    return np.array((x - np.min(x)) / (np.max(x) - np.min(x)))


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    weight_path = './best_f2.pt'

    transform = torchvision.transforms.Compose([
        ToTensor(),
        Resize((224, 224)),
    ])
    consumer = KafkaConsumer(
        'process.payload', bootstrap_servers='localhost:29091', group_id='my-group')
    producer = KafkaProducer(bootstrap_servers='localhost:29091')

    for msg in consumer:
        print('Retreive')
        nparr = readb64(msg.value.decode('utf-8'))

        img_orig = load_img(nparr)
        h, w, _ = img_orig.shape
        img = transform(img_orig)
        img = img.unsqueeze(0).to(device)

        # Define and load state dict
        model = Unet(encoder_name='efficientnet-b0',      # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                     # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                     in_channels=3,
                     # model output channels (number of classes in your dataset)
                     classes=3,
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
        res = []

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        plt.imsave('mask.png', mask)
        plt.imsave('img_orig.png', img_orig)
        plt.imsave('end.png', normalize(mask*0.3 + img_orig*0.7))
        with open("mask.png", "rb") as image_file:
            res.append(base64.b64encode(image_file.read()).decode('utf-8'))
        os.remove("mask.png")
        with open("img_orig.png", "rb") as image_file:
            res.append(base64.b64encode(image_file.read()).decode('utf-8'))
        os.remove("img_orig.png")
        with open("end.png", "rb") as image_file:
            res.append(base64.b64encode(image_file.read()).decode('utf-8'))
        os.remove("end.png")
        res.append(msg.key.decode('utf-8'))
        producer.send('process.payload.reply', str.encode(
            json.dumps(res, separators=(',', ':'))))
