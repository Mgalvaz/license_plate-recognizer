import json
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.utils import draw_bounding_boxes
from PIL import Image
import matplotlib.pyplot as plt

def transform_img(image: Image.Image) -> torch.Tensor:
    tr = v2.Compose([
        v2.PILToTensor(),
        v2.Resize((768, 768)),
        v2.ToDtype(torch.float, True),
    ])
    return tr(image)

class CarPlateTrainDataset(Dataset):

    def __init__(self, path: str, compact: bool =False) -> None:
        super().__init__()
        self.compact = compact
        if compact:
            self.images = torch.load(path + 'images.train.pt', weights_only=True)
            self.labels = torch.load(path + 'labels.train.pt', weights_only=True)
        else:
            self.path = path + 'train/'
            self.train = []
            with open(path + 'train.txt', 'r') as f:
                self.train = [x.rstrip('\n') for x in f]

    def __len__(self) -> int:
        if self.compact:
            return self.images.size(0)
        else:
            return len(self.train)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.compact:
            return self.images[item], self.labels[item]
        else:
            image_path = self.path + self.train[item] + '.jpg'
            label_path = self.path + self.train[item] + '.json'
            image = Image.open(image_path)
            w, h = image.size
            scale_x = 768 / w
            scale_y = 768 / h
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
        scale_x = 768 / w
        scale_y = 768 / h
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

def main():
    dataset = CarPlateTrainDataset('dataset/', compact=True)
    images = []
    lbls = []
    gt_widths = []
    gt_heights = []

    for img, label in dataset:
        images.append(img)
        lbls.append(label)

        ws = label[:, 2] - label[:, 0]
        hs = label[:, 3] - label[:, 1]
        for width in ws:
            gt_widths.append(width)
        for height in hs:
            gt_heights.append(height)

    images = torch.stack(images)
    torch.save(images, 'dataset/images.train.pt')
    torch.save(lbls, 'dataset/labels.train.pt')

    plt.figure()
    plt.scatter(gt_widths, gt_heights, alpha=0.4)
    plt.xlabel("GT width")
    plt.ylabel("GT height")
    plt.title("GT Height vs Width")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.hist(gt_widths, bins=30)
    plt.xlabel("GT width")
    plt.ylabel("count")
    plt.title("Histograma de GT width")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.hist(gt_heights, bins=30)
    plt.xlabel("GT height")
    plt.ylabel("count")
    plt.title("Histograma de GT height")
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()