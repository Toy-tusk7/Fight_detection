import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np

SEQ_DIR = "dataset/sequences"
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)

# ============================
#  Dataset Loader
# ============================
class ViolenceDataset(Dataset):
    def __init__(self):
        self.files = [
            os.path.join(SEQ_DIR, f)
            for f in os.listdir(SEQ_DIR)
            if f.endswith(".npz")
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        x = torch.tensor(data["x"], dtype=torch.float32)
        y = torch.tensor(data["y"], dtype=torch.long)
        return x, y


# ============================
#  LSTM Model
# ============================
class ViolenceLSTM(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, num_layers=2, num_classes=2):
        super(ViolenceLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        lstm_out, (h, c) = self.lstm(x)
        last_hidden = h[-1]
        out = self.fc(last_hidden)
        return out


# ============================
#  TRAINING CODE
#  (Runs ONLY when file executed directly)
# ============================
if __name__ == "__main__":

    dataset = ViolenceDataset()
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = ViolenceLSTM().to(device)

    # Weighted loss — non-violence is smaller class
    weights = torch.tensor([0.4, 1.0], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    EPOCHS = 10

    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            _, predicted = torch.max(pred, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

        accuracy = correct / total * 100
        print(f"Epoch {epoch+1}/{EPOCHS}  Loss: {total_loss:.4f}  Accuracy: {accuracy:.2f}%")

    torch.save(model.state_dict(), "violence_lstm.pth")
    print("\n🎉 Training complete! Model saved as violence_lstm.pth")
