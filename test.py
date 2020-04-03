import re
import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from matplotlib import pyplot as plt

# подключаем скачанную библиотеку tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# папка с исходными изображениями
IMG_DIR = 'checks/'

image = cv2.imread(IMG_DIR + '1.jpg')
d = pytesseract.image_to_data(image, lang='eng', output_type=Output.DICT)
print('DATA KEYS: \n', d.keys())

n_boxes = len(d['text'])
for i in range(n_boxes):
    # condition to only pick boxes with a confidence > 60%
    if int(d['conf'][i]) > 60:
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

b, g, r = cv2.split(image)
rgb_img = cv2.merge([r, g, b])
plt.figure(figsize=(5, 5))
plt.imshow(rgb_img)
plt.title('Пример')
plt.show()

