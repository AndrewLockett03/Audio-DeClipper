import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from model import DeclipCNN
from dataset import DeclipDataset, collate_fn

# Hyperparameters
EPOCHS = 1
LR = 0.001
BATCH_SIZE = 8
DATA_DIR = "/media/andrew/DATASETS/clips"

# Load Dataset
dataset = DeclipDataset(DATA_DIR)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# Initialize Model, Loss, Optimizer
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")
model = DeclipCNN().to(device)
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

count = 0

# Training Loop
for epoch in range(EPOCHS):
    for spec_clipped, spec_original in dataloader:
        spec_clipped, spec_original = spec_clipped.to(device), spec_original.to(device)

        count += 1
        optimizer.zero_grad()
        output = model(spec_clipped)
        if count % 100 == 0:
            print(count)
        loss = criterion(output, spec_original)
        # loss_value = loss.item()
        loss.backward()
        optimizer.step()

        max_mem = torch.cuda.max_memory_allocated() / 1024 ** 3
        print(f"[GPU] Max Allocated: {max_mem:.2f} GB")
        if max_mem > 6.0:
            print(spec_clipped.shape)
        elif max_mem < 2.0:
            print(spec_clipped.shape)

        torch.cuda.reset_peak_memory_stats()

        del spec_clipped, spec_original, output, loss
        torch.cuda.empty_cache()

    print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {loss.item():.6f}")
