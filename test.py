from kafka import KafkaConsumer, KafkaProducer
import numpy as np
import base64
import cv2

consumer = KafkaConsumer('process.payload', bootstrap_servers='localhost:29091', group_id='my-group')
producer = KafkaProducer(bootstrap_servers='localhost:29091')
real_img = cv2.imread('./img/t1.png', cv2.IMREAD_UNCHANGED)
def readb64(uri):
   encoded_data = uri.split(',')[1]
   print(encoded_data)
   nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
   return nparr

def load_img(nparr):
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    img = np.tile(img[...,None], [1, 1, 3]) # gray to rgb
    img = img.astype('float32') # original is uint16
    mx = np.max(img)
    if mx:
        img/=mx # scale image to [0, 1]
    return img

for msg in consumer:
  print(msg.value.decode('utf-8'))
  print(readb64(msg.value.decode('utf-8')))
  # producer.send('process.payload.reply', str.encode('hello ' + msg.value.decode('utf-8')))