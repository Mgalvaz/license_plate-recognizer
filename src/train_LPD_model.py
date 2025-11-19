"""
Script that trains the OCR model with synthetic generated license plates.

Usage:
    python train_OCR_model.py [--model-path path] [--output-path out] [--epochs num] [N].

Input:
    path (str, optional): Trained model path (.pth) for loading.
    out (str, optional): Path to save the trained model (.pth).
    num (int, optional): Number of epochs during the training.
    N (int, optional): Number of train images in the dataset. Default is 10000

Output:
    If out argument was passed, the model is saved in the following format:
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),
    }, out)
"""
import argparse
import torch
from torch import nn

class InceptionResidual(nn.Module):

    def __init__(self, inc, outc, *args, include_max_pool=True):
        super().__init__()

        self.inception: list[nn.Module] = [nn.Conv2d(inc, outc, kernel, padding=1) for kernel in args]

        if include_max_pool:
            self.inception.append(nn.MaxPool2d(2,2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return x

class CRNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, (3, 3), padding=1),  # (1, 32, 150) -> (32, 32, 150)
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # (32, 32, 150) -> (32, 16, 75)
            nn.Conv2d(64, 128, (3, 3), padding=1),  # (32, 16, 75) -> (64, 16, 75)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (64, 16, 75) -> (64, 8, 37)
            nn.Conv2d(128, 256, (3, 3), padding=1),  # (64, 8, 37) -> (128, 8, 37)
            nn.ReLU(),
            nn.MaxPool2d((1, 2), (2,1)),  # (128, 8, 37) -> (128, 4, 36)
            nn.Conv2d(256, 512, (3, 3), padding=1),  # (128, 4, 36) -> (256, 4, 36)
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), padding=1),  # (256, 4, 37) -> (256, 4, 36)
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.MaxPool2d((2, 1), 1),  # (256, 4, 36) -> (256, 3, 36)
            nn.Conv2d(512, 512, (3, 3), padding=0),  # (256, 3, 36) -> (512, 1, 34)
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.rnn = nn.GRU(512, 256, num_layers=2, batch_first=True, bidirectional=True) # (34, 512) -> (512, 34)

        self.decoder = nn.Linear(512, NUM_CLASSES) # (512, 34) -> (37, 34)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, ic, ih, iw = x.size() # (batch, channels=1, height=32, width=150)
        assert (ic, ih, iw) == (1, 32, 150), f'Input size ({ic}, {ih}, {iw}) does not correspond to expected size (1, 32, 150)'
        x = self.cnn(x) # (batch, channels=512, height=1, width=34)

        x = x.squeeze(2).permute(0, 2, 1)  # (batch, width=34, channels=512)

        x, _ = self.rnn(x) # (batch, seq_len=34, channels=512)

        x = self.decoder(x) # (batch, seq_len=34, label=37)
        return x



def main():

    parser = argparse.ArgumentParser(description='OCR_model training')
    parser.add_argument('total', metavar='N', type=int, nargs='?', default=10000, help='Number of train images in the dataset')
    parser.add_argument('--model-path', type=str, default=None, help='Trained model path (.pth) for loading')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs during the training')
    parser.add_argument('--output-path', type=str, default='models/OCR_model.pth', help='Path to save the trained model (.pth)')
    args = parser.parse_args()

    # Load dataset
    num_test = args.total//7
    num_train = args.total - num_test
    dataset = SyntheticPlateDataset(num_samples=args.total)
    train_dataset, test_dataset = random_split(dataset, [num_train, num_test])

    train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64,  collate_fn=collate_fn)

    for img, _ in train_loader:
        pass

    # Model loading
    device = torch.device('cpu')
    model = CRNN()
    criterion = nn.CTCLoss(blank=0, zero_infinity=True, reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    if args.model_path:
        print(f'Loading model from {args.model_path}')
        checkpoint = torch.load(args.model_path, weights_only=True, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        last_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f'Checkpoint loaded. Last epoch: {last_epoch}, with loss: {loss}')
    else:
        last_epoch = 0


    # Training
    if args.total > 0:
        model.train()
        num_epochs = args.epochs + last_epoch
        for epoch in range(last_epoch+1, num_epochs+1):
            print(f'Epoch {epoch}/{num_epochs}', end=' ')
            for batch_images, batch_labels in train_loader:
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                batch_output = model(batch_images)
                log_probs = F.log_softmax(batch_output, dim=2).permute(1, 0 ,2)
                input_lengths = torch.full(size=(batch_output.size(0),), fill_value=batch_output.size(1), dtype=torch.long)
                targets = batch_labels
                batch_labels = [lbl[lbl != -1] for lbl in batch_labels]
                target_lengths = torch.tensor([len(l) for l in batch_labels], dtype=torch.long)
                loss = criterion(log_probs, targets, input_lengths, target_lengths)
                # backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                # optimize
                optimizer.step()
            print(f'loss: {loss.item()}')

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

    with torch.no_grad():
        image = Image.open('src/3245_LCX.png')
        ten = transform(image)
        v2.ToPILImage()(ten.squeeze(0)).show()
        ten = ten.unsqueeze(0)
        input('siguiente')
        out = model(ten)
        pred = out.argmax(dim=2).cpu().numpy().T
        decoded_preds = ctc_decode(pred)
        print([REVERSE_TRANSLATOR[n[0]] for n in decoded_preds])


    """torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),
    }, args.output_path)"""


if __name__ == "__main__":
    main()