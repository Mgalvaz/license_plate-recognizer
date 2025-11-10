import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from OCR_dataset_generator import SyntheticPlateDataset

class MyModel(nn.Module):

    def __init__(self):
        super(MyModel, self).__init__()
        #CNN
        self.c2d_1 = nn.Conv2d(1, 32, (3,3), padding=1)
        self.pool_1 = nn.MaxPool2d(2, 2)
        self.c2d_2 = nn.Conv2d(32, 64, (3, 3), padding=1)
        self.pool_2 = nn.MaxPool2d(2, 2)

        #RNN
        self.rnn = nn.RNN(320, 64, batch_first=True) #320 = 64 * 5 = (channels * height) as if it were reading the image left to right

        #Classifier
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        _, c, h, w = x.size()
        assert (c, h, w) == (1, 20, 94), f'Input size ({c}, {h}, {w}) does not correspond to expected size (1, 20, 94)'
        x = F.relu(self.c2d_1(x))
        x = self.pool_1(x)
        x = F.relu(self.c2d_2(x))
        x = self.pool_2(x)

        b, c, h, w = x.size()

        # convertimos a secuencia: ancho -> seq_len, features = channels*height
        x = x.permute(0, 3, 1, 2)  # (batch, width, channels, height)
        x = x.reshape(b, w, c * h)  # (batch, seq_len, input_size)

        out, _ = self.rnn(x)  # (batch, seq_len, hidden_size)
        out = self.fc(out)
        return out

if __name__ == "__main__":
    # Parámetros
    seq_len = 4  # longitud de "matrícula"
    input_size = 10  # codificamos dígitos 0-9 como one-hot
    output_size = 10  # número de clases posibles (0-9)
    num_samples = 1000

    dataset = SyntheticPlateDataset(num_samples=400)


    # Crear dataset sencillo
    def random_seq():
        return [random.randint(0, 9) for _ in range(seq_len)]


    X = []
    Y = []
    for _ in range(num_samples):
        seq = random_seq()
        # Entrada como one-hot
        x = torch.zeros(seq_len, input_size)
        for t, digit in enumerate(seq):
            x[t, digit] = 1.0
        X.append(x)
        # Salida como índices
        y = torch.tensor(seq)
        Y.append(y)

    X = torch.stack(X)  # (batch_size, seq_len, input_size)
    Y = torch.stack(Y)  # (batch_size, seq_len)

    model = MyModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(20):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output.view(-1, output_size), Y.view(-1))
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    i = 'y'
    while i == 'y':
        x = input('Escribe la matricula: ')
        m = torch.zeros(len(x), 10)
        for j, l in enumerate(x):
            m[j, int(l)] = 1
        pred = model(m.unsqueeze(0))
        pred_digits = ''.join(map(str, pred.argmax(dim=2).squeeze().tolist()))
        print("Secuencia predicha:", pred_digits)
        i = input('Continuar? y/n')

