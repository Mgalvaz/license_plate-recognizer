import random
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from typing_extensions import deprecated


def generate_plate_text() -> str:
    nums = ''.join(random.choices('0123456789', k=4))

    consonants = "BCDFGHJKLMNPQRSTVWXYZ"
    vowels = "AEIOU"
    alphabet = consonants + vowels
    weights = [10] * len(consonants) + [1] * len(vowels)
    letters = ''.join(random.choices(alphabet, weights=weights, k=3))

    return f"{nums} {letters}"


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

    def __init__(self, num_samples: int = 10000):
        super().__init__()
        self.num_samples = num_samples
        self.font_main = ImageFont.truetype('arial.ttf', 27)
        self.font_small = ImageFont.truetype('arial.ttf', 6)
        self.transform = augment_image_v2
        self.translator = dict((l, n) for n, l in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', start=1))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple:
        plate_text = generate_plate_text()
        plate = Image.new("L", (150, 32), color=230)
        draw = ImageDraw.Draw(plate)
        draw.rectangle([3, 3, 15, 29], fill=134)
        draw.text((7, 19), 'E', font=self.font_small, fill=230)
        draw.text((20, 2), plate_text, font=self.font_main, fill=50)
        plate = self.transform(plate)
        label = torch.tensor([self.translator[l] for l in plate_text if l != ' '], dtype=torch.long)
        return plate, label