import json
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.utils import draw_bounding_boxes
from PIL import Image

def transform_img(image: Image.Image) -> torch.Tensor:
    tr = v2.Compose([
        v2.PILToTensor(),
        v2.Resize((1536, 1536)),
        v2.ToDtype(torch.float, True),
    ])
    return tr(image)

class CarPlateTrainDataset(Dataset):

    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path + 'train/'
        self.train = []
        with open(path+'train.txt', 'r') as f:
            self.train = [x.rstrip('\n') for x in f]

    def __len__(self) -> int:
        return len(self.train)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = self.path + self.train[item] + '.jpg'
        label_path = self.path + self.train[item] + '.json'
        image = Image.open(image_path)
        w, h = image.size
        scale_x = 1536 / w
        scale_y = 1536 / h
        image = transform_img(image)
        with open(label_path) as f:
            full_label = json.load(f)
        labels = []
        for lbl in full_label['lps']:
            lp = torch.Tensor(lbl['poly_coord'])
            x_min = lp[:, 0].min() * scale_x
            y_min = lp[:, 1].min() * scale_y
            x_max = lp[:, 0].max() * scale_x
            y_max = lp[:, 1].max() * scale_y
            labels.append(torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32))
        labels = torch.stack(labels)
        return image, labels

class CarPlateTestDataset(Dataset):

    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path + 'test/'
        self.test = []
        with open(path+'test.txt', 'r') as f:
            self.test = [x.rstrip('\n') for x in f]

    def __len__(self) -> int:
        return len(self.test)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = self.path + self.test[item] + '.jpg'
        label_path = self.path + self.test[item] + '.json'
        image = Image.open(image_path)
        w, h = image.size
        scale_x = 1536 / w
        scale_y = 1536 / h
        image = transform_img(image)
        with open(label_path) as f:
            full_label = json.load(f)
        labels = []
        for lbl in full_label['lps']:
            lp = torch.Tensor(lbl['poly_coord'])
            x_min = lp[:, 0].min() * scale_x
            y_min = lp[:, 1].min() * scale_y
            x_max = lp[:, 0].max() * scale_x
            y_max = lp[:, 1].max() * scale_y
            labels.append(torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32))
        labels = torch.stack(labels)
        return image, labels

if __name__ == '__main__':
    dataset = CarPlateTrainDataset('dataset/')
    """minx = 100000
    miny = 100000
    minw = 100000
    minh = 100000
    maxw = 0
    maxh = 0
    for img, label in dataset:
        x = img.size(1)
        y = img.size(2)
        w = float((label[:,2] - label[:,0]).min())
        h = float((label[:,3] - label[:,1]).min())
        if w < minw: minw = w
        if h < minh: minh = h
        w = float((label[:, 2] - label[:, 0]).max())
        h = float((label[:, 3] - label[:, 1]).max())
        if w > maxw: maxw = w
        if h > maxh: maxh = h
        if x < minx: minx = x
        if y < miny: miny = y"""
    img, label = dataset[26]
    print(label)
    img = draw_bounding_boxes(img, label, fill=True, colors=['yellow']*label.size(0))
    v2.ToPILImage()(img).show()