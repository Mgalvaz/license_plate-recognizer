import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.nn import Sequential
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from OCR_dataset_generator import SyntheticPlateDataset

def collate_fn(batch: list):
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs)
    lengths = [len(lbl) for lbl in labels]
    labels = pad_sequence(labels, batch_first=True, padding_value=-1)  # rellena con -1
    return imgs, labels, lengths

class MyModel(nn.Module):

    def __init__(self):
        super(MyModel, self).__init__()

        self.cnn = Sequential(
            nn.Conv2d(1, 32, (3,3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, (3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.rnn = nn.GRU(320, 64, batch_first=True, bidirectional=True) #320 = 64 * 5 = (channels * height) as if it were reading the image left to right

        self.fc = nn.Linear(128, 10) #128 = 64 * 2 because GRU is bidirectional

    def forward(self, x):
        _, ic, ih, iw = x.size()
        assert (ic, ih, iw) == (1, 20, 94), f'Input size ({ic}, {ih}, {iw}) does not correspond to expected size (1, 20, 94)'
        x = self.cnn(x)

        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2)  # (batch, width, channels, height)
        x = x.reshape(b, w, c * h)  # (batch, seq_len, input_size)

        x, _ = self.rnn(x)
        x = self.fc(x)
        return x.log_softmax(2)


if __name__ == "__main__":
    #Datset loading
    dataset = SyntheticPlateDataset(num_samples=400, precomputed_file='datasets/ocr_dataset.pt')
    loader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

    #Model loading
    model = MyModel()
    criterion = nn.CTCLoss(zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    #Training
    for batch_images, batch_labels in loader:
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

    #Testing
    with torch.no_grad():
        output = model(img.unsqueeze(0))  # (1, T, C)
        pred = output.argmax(2).squeeze(0).cpu().numpy()

    decoded = []
    prev = -1
    for p in pred:
        if p != prev and p != 0:  # 0 = blank
            decoded.append(p)
        prev = p

