import json
import random
from typing import Literal

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from typing_extensions import deprecated


def generate_plate_text() -> str:
    consonants = 'BCDFGHJKLMNPQRSTVWXYZ'
    vowels = 'AEIOU'
    alphabet = consonants + vowels

    nums = ''.join(random.choices('0123456789', k=4))
    weights = [10] * len(consonants) + [1] * len(vowels)
    letters = ''.join(random.choices(alphabet, weights=weights, k=3))
    return f'{nums} {letters}'

def generate_plate_text_v2() -> str:
    chars = ' ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    probs = [8] + [1]*36
    return ''.join(random.choices(chars, weights=probs, k=8))


def add_gaussian_noise(img: Image, mean: int = 0, std: int = 8) -> Image:
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(mean, std, arr.shape)
    arr_noisy = arr + noise
    arr_noisy = np.clip(arr_noisy, 0, 255)
    return Image.fromarray(arr_noisy.astype(np.uint8))


def apply_perspective_transform(img: Image.Image, max_shift: int = 3) -> Image:
    width, height = img.size
    src = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    dst = src + np.random.randint(-max_shift, max_shift + 1, size=(4, 2))

    def find_perspective_coeffs(src_points: np.ndarray, dst_points: np.ndarray) -> list[int]:
        matrix = []
        for p1, p2 in zip(dst_points, src_points):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])
        A = np.matrix(matrix, dtype=np.float64)
        B = np.array(src_points).reshape(8)
        res, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
        return res.tolist()

    coeffs = find_perspective_coeffs(src, dst)
    return img.transform(img.size, Image.PERSPECTIVE, coeffs, fillcolor=100)


def random_brightness_contrast(img: Image, brightness_range: tuple = (0.8, 1.2), contrast_range: tuple = (0.8, 1.2)) -> Image:
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(np.random.uniform(*brightness_range))
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(np.random.uniform(*contrast_range))
    return img


@deprecated('Use augment_image_v2 instead')
def augment_image(img: Image) -> Image:
    img = img.rotate(np.random.uniform(-5, 5), expand=False, fillcolor=100)  # Rotation
    img = apply_perspective_transform(img)  # Transformation to simulate depth
    img = add_gaussian_noise(img)  # Noise
    img = random_brightness_contrast(img)  # Change brightness
    return img


def augment_image_v2(img: Image.Image) -> torch.Tensor:
    tr = v2.Compose([
        v2.PILToTensor(),
        v2.ToDtype(torch.float, True),
        v2.RandomRotation(degrees=8, fill=0.392),
        v2.RandomPerspective(distortion_scale=0.25, p=1.0, fill=0.392),
        v2.GaussianNoise(sigma=0.05),
        v2.RandomApply([v2.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2))], p=0.7),
    ])
    return tr(img)


class SyntheticPlateDataset(Dataset):
    """
    Dataset that generates fake synthetic spanish license plates and augments them to simulate perspective.
    """

    def __init__(self, num_samples: int = 10000) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.font_main = ImageFont.truetype('arial.ttf', 27)
        self.font_small = ImageFont.truetype('arial.ttf', 6)
        self.transform = augment_image_v2
        self.translator = dict((l, n) for n, l in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', start=1))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        plate_text = generate_plate_text_v2()
        plate = Image.new("L", (150, 32), color=230)
        draw = ImageDraw.Draw(plate)
        start = random.randint(0, 10)
        draw.text((start, 2), plate_text, font=self.font_main, fill=50)
        plate = self.transform(plate)
        label = torch.tensor([self.translator[l] for l in plate_text if l != ' '], dtype=torch.long)
        return plate, label

class LPDataset(Dataset):
    def __init__(self, path: str, split: Literal['test', 'train']) -> None:
        super().__init__()
        self.translator = dict((l, n) for n, l in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', start=1))
        self.path = path + '\\' + split + '\\'
        with open(path + r'\lp' + split + '.txt', 'r') as f:
            self.dataset_ids = [line.rstrip('\n') for line in f]

    def __len__(self) -> int:
        return len(self.dataset_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        file_id, box_num = self.dataset_ids[index].split('_')
        image_path = self.path + file_id + '.jpg'
        label_path = self.path + file_id + '.json'
        with open(label_path) as f:
            full_label = json.load(f)
        lp = full_label['lps'][int(box_num)]
        # Crop the image to get the license plate
        poly = np.array(lp['poly_coord'])
        xmin = poly[:, 0].min()
        ymin = poly[:, 1].min()
        xmax = poly[:, 0].max()
        ymax = poly[:, 1].max()
        box = [xmin, ymin, xmax, ymax]
        image = Image.open(image_path).crop(box).convert('L')
        image_tensor = v2.PILToTensor()(image)
        # License plate ID
        plate_text = lp['lp_id'][3:]
        label = torch.tensor([self.translator[l] for l in plate_text if l != '*'], dtype=torch.long)
        return image_tensor, label


if __name__ == '__main__':
        # Path to the dataset
    path = r'C:\Repositorio\license_plate-recognizer\dataset'+'\\'
    split = 'train'
    # Obtain the file names of the specified split
    with open(path + split + '.txt', 'r') as f:
        dataset_ids = [line.rstrip('\n') for line in f]
    # Save each license plate as a different file
    lps_names = []
    for file_id in dataset_ids:
        label_path = path + split + '\\' + file_id + '.json'
        with open(label_path) as f:
            full_label = json.load(f)
        for i, lbl in enumerate(full_label['lps']):
            lps_names.append(file_id + '_' + str(i))
    # Write the file names
    with open(path+r'\lp'+split+'.txt', 'x') as f:
        f.write('\n'.join(lps_names))


    exit()
    image = Image.open(r'C:\Repositorio\license_plate-recognizer\src\3245_LCX.png')
    transform = v2.Compose([
        v2.Grayscale(num_output_channels=1),
        v2.Resize((32, 150)),
        v2.PILToTensor(),
        v2.ToDtype(torch.float, True),
    ])
    ten = transform(image)
    v2.ToPILImage()(ten.squeeze(0)).show()
    ten = ten.unsqueeze(0)
    dataset = SyntheticPlateDataset(num_samples=1)
    dataloader = DataLoader(dataset, batch_size=1)
    inv_translator = dict((n, l) for n, l in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', start=1))
    for img, label in dataloader:
        v2.ToPILImage()(img.squeeze(0)).show()
        print([inv_translator[t.item()] for t in label[0]])