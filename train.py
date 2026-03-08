import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from model import DeclipCNN
from dataset import DeclipDataset

# Hyperparameters
EPOCHS = 1
LR = 0.001
BATCH_SIZE = 8
DATA_DIR = "/Users/andrewlockett/.cache/kagglehub/datasets/mozillaorg/common-voice/versions/2/cv-valid-train/cv-valid-train"

# Load Dataset
print("Loading Dataset: ")
dataset = DeclipDataset(DATA_DIR)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# Initialize Model, Loss, Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda")
model = DeclipCNN().to(device)
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

count = 0

# Training Loop
for epoch in range(EPOCHS):
    for wave_clipped, wave_original in dataloader:
        print("Loading data for epoch " + str(epoch))
        wave_clipped, wave_original = wave_clipped.to(device), wave_original.to(device)

        count += 1
        optimizer.zero_grad()
        print("Pass through the model")
        output = model(wave_clipped)
        if count % 100 == 0:
            print(count)
        loss = criterion(output, wave_original)
        # loss_value = loss.item()

        print("Backward propagation: ")
        loss.backward()
        optimizer.step()

        max_mem = torch.cuda.max_memory_allocated() / 1024 ** 3
        print(f"[GPU] Max Allocated: {max_mem:.2f} GB")
        if max_mem > 6.0:
            print(wave_clipped.shape)
        elif max_mem < 2.0:
            print(wave_clipped.shape)

        # torch.cuda.reset_peak_memory_stats()

        del wave_clipped, wave_original, output, loss
        torch.cuda.empty_cache()

    print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {loss.item():.6f}")
