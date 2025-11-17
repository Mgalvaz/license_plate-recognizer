import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split
from PIL import Image
from torchvision.transforms import transforms, v2

from OCR_dataset import SyntheticPlateDataset

TRANSLATOR = dict((l, n) for n, l in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', start=1))

def collate_fn(batch: list) -> tuple[torch.Tensor, torch.Tensor]:
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs)
    labels = pad_sequence(labels, batch_first=True, padding_value=-1)  # rellena con -1
    return imgs, labels

def ctc_decode(pred_seq: torch.Tensor, blank: int=0) -> list[int]:
    decoded = []
    prev = None
    for p in pred_seq:
        if p != blank and p != prev:
            decoded.append(p)
        prev = p
    return decoded

class CRNN(nn.Module):

    def __init__(self):
        super(CRNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3), padding=1),  # (1, 32, 150) -> (32, 32, 150)
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # (32, 32, 150) -> (32, 16, 75)
            nn.Conv2d(32, 64, (3, 3), padding=1),  # (32, 16, 75) -> (64, 16, 75)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (64, 16, 75) -> (64, 8, 37)
            nn.Conv2d(64, 128, (3, 3), padding=1),  # (64, 8, 37) -> (128, 8, 37)
            nn.ReLU(),
            nn.MaxPool2d((1, 2), (2,1)),  # (128, 8, 37) -> (128, 4, 36)
            nn.Conv2d(128, 256, (3, 3), padding=1),  # (128, 4, 36) -> (256, 4, 36)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), padding=1),  # (256, 4, 37) -> (256, 4, 36)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), 1),  # (256, 4, 36) -> (256, 3, 36)
            nn.Conv2d(256, 512, (3, 3), padding=0),  # (256, 3, 36) -> (512, 1, 34)
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.rnn = nn.GRU(512, 256, num_layers=2, batch_first=True, bidirectional=True) # (36, 512) -> (512, 36)

        self.fc = nn.Linear(512, 37) # (512, 36) -> (37, 36)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, ic, ih, iw = x.size() # (batch, channels=1, height=32, width=150)
        assert (ic, ih, iw) == (1, 32, 150), f'Input size ({ic}, {ih}, {iw}) does not correspond to expected size (1, 32, 150)'
        x = self.cnn(x) # (batch, channels=512, height=1, width=34)

        x = x.squeeze(2).permute(0, 2, 1)  # (batch, width=34, channels=512)

        x, _ = self.rnn(x) # (batch, seq_len=34, channels=512)
        x = self.fc(x) # (batch, seq_len=34, label=37)
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
    model = CRNN()
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

    torch.stack()
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
        v2.Resize((32, 150)),
        v2.PILToTensor(),
        v2.ToDtype(torch.float, True),
    ])

    torch.argmax()

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