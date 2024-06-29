import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.rnn.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

def train_and_predict_rnn(X_train, y_train, X_test, num_epochs=30, learning_rate=0.0005, hidden_dim=256):
    input_dim = X_train.shape[1]
    output_dim = 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = RNNModel(input_dim, hidden_dim, output_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).unsqueeze(1)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).unsqueeze(1)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor),
                                               batch_size=32, shuffle=True)

    model.train()
    for epoch in tqdm(range(num_epochs), desc="Training RNN"):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        rnn_outputs = model(X_test_tensor.to(device))
        rnn_pred = (torch.sigmoid(rnn_outputs).cpu().numpy() > 0.5).astype(int).squeeze()

    torch.save(model.state_dict(), 'rnn_model.h5')
    joblib.dump(scaler, 'scaler_rnn.pkl')

    return rnn_pred
