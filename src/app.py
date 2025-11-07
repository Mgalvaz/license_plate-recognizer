import torch
import random
import torch.nn as nn
import torch.optim as optim

class MyModel(nn.Module):

    def __init__(self):
        super(MyModel, self).__init__()
        self.c2d_1 = nn.Conv2d(1, 32, (3,3))
        self.pool_1 = nn.MaxPool2d(2, 1)
        self.c2d_2 = nn.Conv2d(1, 32, (3, 3))
        self.pool_2 = nn.MaxPool2d(2, 1)
        self.rnn = nn.RNN(10, 20, batch_first=True)
        self.fc = nn.Linear(20, 10)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out)
        return out

if __name__ == "__main__":
    # Parámetros
    seq_len = 4  # longitud de "matrícula"
    input_size = 10  # codificamos dígitos 0-9 como one-hot
    output_size = 10  # número de clases posibles (0-9)
    num_samples = 1000


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

