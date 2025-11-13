import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.nn import Sequential
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split
from OCR_dataset import SyntheticPlateDataset

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
    total = 700
    num_test = total//7
    num_train = total - num_test
    dataset = SyntheticPlateDataset(num_samples=total, precomputed_file='../datasets/ocr_dataset.pt')
    train_dataset, test_dataset = random_split(dataset, [num_train, num_test])

    train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64,  collate_fn=collate_fn)

    #Model loading
    model = MyModel()
    criterion = nn.CTCLoss(zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=0.01)


    #Training
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