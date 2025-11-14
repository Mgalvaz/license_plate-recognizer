import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.nn import Sequential
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split
from PIL import Image
from torchvision.transforms import transforms, v2

from OCR_dataset import SyntheticPlateDataset

TRANSLATOR = dict((l, n) for n, l in enumerate('BCDFGHJKLMNPQRSTVWXYZ0123456789', start=1))

def collate_fn(batch: list):
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs)
    labels = pad_sequence(labels, batch_first=True, padding_value=-1)  # rellena con -1
    return imgs, labels

def ctc_decode(pred_seq, blank=0):
    decoded = []
    prev = None
    for p in pred_seq:
        if p != blank and p != prev:
            decoded.append(p)
        prev = p
    return decoded

class MyModel(nn.Module):

    def __init__(self):
        super(MyModel, self).__init__()

        self.cnn = Sequential(
            nn.Conv2d(1, 128, (3,3), padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, (3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 256, (3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, (3, 3), padding=1),
            nn.ReLU()
        )

        self.rnn = nn.GRU(4608, 64, batch_first=True, bidirectional=True) #256 = 64 * 8 = (channels * height) as if it were reading the image left to right

        self.fc = nn.Linear(128, 32) #128 = 64 * 2 because GRU is bidirectional, 32 = num_classes + 1 (for <blank>)

    def forward(self, x):
        _, ic, ih, iw = x.size()
        assert (ic, ih, iw) == (1, 18, 91), f'Input size ({ic}, {ih}, {iw}) does not correspond to expected size (1, 18, 91)'
        x = self.cnn(x)
        #print('CNN', x.min(), x.max())

        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2)  # (batch, width, channels, height)
        x = x.reshape(b, w, c * h)  # (batch, seq_len, input_size)

        x, _ = self.rnn(x)
        #print('RNN', x.min(), x.max())
        x = self.fc(x)
        return x.log_softmax(2)


if __name__ == "__main__":
    #Datset loading
    total = 700
    num_test = total//7
    num_train = total - num_test
    num_epochs = 1
    dataset = SyntheticPlateDataset(num_samples=total)
    train_dataset, test_dataset = random_split(dataset, [num_train, num_test])

    train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64,  collate_fn=collate_fn)

    #Model loading
    model = MyModel()
    criterion = nn.CTCLoss(zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=0.01)


    #Training
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1} out of {num_epochs}')
        iteration = 1
        num_iterations = len(train_loader)
        for batch_images, batch_labels in train_loader:
            print(f'iteration: {iteration} out of {num_iterations}')
            iteration += 1
            optimizer.zero_grad()

            batch_output = model(batch_images)
            log_probs = batch_output.permute(1, 0, 2)
            input_lengths = torch.full(size=(batch_output.size(0),), fill_value=batch_output.size(1), dtype=torch.long)
            batch_labels = [lbl[lbl != -1] for lbl in batch_labels] # Remove -1 padding
            targets = torch.cat(batch_labels)
            target_lengths = torch.tensor([len(l) for l in batch_labels], dtype=torch.long)

            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()
        print()


    #Testing
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch_images, batch_labels in test_loader:
            output = model(batch_images).permute(1, 0 ,2)
            preds = output.argmax(dim=2).cpu().numpy().T

            batch_labels = [lbl[lbl != -1] for lbl in batch_labels]
            decoded_preds = [ctc_decode(seq) for seq in preds]

            for pred, target in zip(decoded_preds, batch_labels):
                target = target.cpu().numpy().tolist()
                if pred == target:
                    correct += 1
    print(f"Exact match accuracy: {correct / num_test:.2%}")


    input('Predecir la imagen')
    transform = transforms.Compose([
        v2.Grayscale(num_output_channels=1),
        v2.Resize((18, 91)),
        v2.PILToTensor(),
        v2.ToDtype(torch.float, True),
    ])

    with torch.no_grad():
        image = Image.open('src/3245_LCX.png')
        #image.show()
        #input('siguiente')
        ten = transform(image)
        ten = ten.unsqueeze(0)
        #v2.ToPILImage()(ten).show()
        #input('siguiente')
        out = model(ten)
        pred = out.argmax(dim=2).cpu().numpy().T
        decoded_preds = ctc_decode(pred)
        print(decoded_preds)