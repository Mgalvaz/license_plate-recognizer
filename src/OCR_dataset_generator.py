"""
Script that generates the dataset to train the OCR CRNN using numpy and Pillow.

Usage:
    python generate_ocr_dataset.py [num_examples]

Input:
    num_examples (int, optional): Number of examples to generate. Default is 10000.

Output:
    Dataset is saved in 'ocr_dataset.pt' in PyTorch tensor format.

    - x shape: (num_examples, 1, H, W)
    - y shape: (num_examples, 1, 7)
"""
import random
import os
from sys import argv
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance

def generate_plate_text():
    nums = ''.join(random.choices('0123456789', k=4))
    letters = ''.join(random.choices('BCDFGHJKLMNPQRSTVWXYZ', k=3))
    return f'{nums}  {letters}'

def add_gaussian_noise(img, mean=0, std=8):
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(mean, std, arr.shape)
    arr_noisy = arr + noise
    arr_noisy = np.clip(arr_noisy, 0, 255)
    return Image.fromarray(arr_noisy.astype(np.uint8))

def apply_perspective_transform(img, max_shift=3):
    width, height = img.size
    src = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    dst = src + np.random.randint(-max_shift, max_shift + 1, size=(4, 2))

    def find_perspective_coeffs(src_points, dst_points):
        matrix = []
        for p1, p2 in zip(dst_points, src_points):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])
        A = np.matrix(matrix, dtype=float)
        B = np.array(src_points).reshape(8)
        res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
        return np.array(res).reshape(8)

    coeffs = find_perspective_coeffs(dst, src)
    return img.transform(img.size, Image.PERSPECTIVE, coeffs, fillcolor=100)

def random_brightness_contrast(img, brightness_range=(0.8,1.2), contrast_range=(0.8,1.2)):
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(np.random.uniform(*brightness_range))
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(np.random.uniform(*contrast_range))
    return img

def augment_image(img):
    img = img.rotate(np.random.uniform(-5, 5), expand=False, fillcolor=100) # Rotation
    img = apply_perspective_transform(img) # Transformation to simulate depth
    img = add_gaussian_noise(img) # Noise
    img = random_brightness_contrast(img) # Change brightness
    return img

font = ImageFont.truetype('arial.ttf', 15)
num_examples = int(argv[1]) if len(argv) > 1 and argv[1].isnumeric() else 10000
translator = dict((l, i) for i, l in enumerate('BCDFGHJKLMNPQRSTVWXYZ0123456789'))

x = torch.empty((num_examples, 1, 20, 94))
y = []

for i in range(num_examples):
    plate_text = generate_plate_text()
    imag = Image.new("L", (94, 20), color=100)
    draw = ImageDraw.Draw(imag)
    draw.rectangle([4, 2, 16, 19], fill=55)
    draw.rectangle([16, 2, 89, 19], fill=230)
    draw.text((8, 12), 'E', font=ImageFont.truetype('arial.ttf', 5), fill=230)
    draw.text((18, 2), plate_text, font=font, fill=50)
    imag = augment_image(imag)
    x[i] =  torch.tensor(np.array(imag), dtype=torch.float32).unsqueeze(0) / 255.0
    y.append(torch.tensor(np.array([translator[l] for l in plate_text if l != ' '])))

os.makedirs('datasets',exist_ok=True)
torch.save((x,y), 'datasets/ocr_dataset.pt')




