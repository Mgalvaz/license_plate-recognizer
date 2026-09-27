import json
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import v2
from PIL import Image

class CarPlateDetectionDataset(Dataset):
    """
    Dataset with GT boxes for license plates in cars. See https://github.com/ramajoballester/UC3M-LP.
    """

    def __init__(self, path: str, split: str) -> None:
        """
        Dataset with GT boxes for license plates in cars. See https://github.com/ramajoballester/UC3M-LP.
        :param path: Root directory of the dataset.
        :param split: Type of split to use. Can be 'train' or 'test'.
        """
        super().__init__()
        self.path = path + split + '/'
        with open(path + split + '.txt', 'r') as f:
            self.train = [x.rstrip('\n') for x in f]

    def __len__(self) -> int:
        return len(self.train)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image_path = self.path + self.train[index] + '.jpg'
        label_path = self.path + self.train[index] + '.json'
        image = Image.open(image_path).convert('RGB')
        transform_image = v2.Compose([
            v2.PILToTensor(),
            v2.ToDtype(torch.float, True),
        ])
        image = transform_image(image)
        with open(label_path) as f:
            full_label = json.load(f)
        boxes = []

        for lp in full_label['lps']:
            poly = np.array(lp['poly_coord'])
            xmin = poly[:, 0].min()
            ymin = poly[:, 1].min()
            xmax = poly[:, 0].max()
            ymax = poly[:, 1].max()
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.ones(len(boxes), dtype=torch.int64)
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([index])
        }
        return image, target