import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from model import DeclipCNN
from dataset import DeclipDataset
from pathlib import Path
from preprocess import chunk_dataset

# Hyperparameters
EPOCHS = 8
LR = 0.001
BATCH_SIZE = 64
CHUNK_SIZE = 16384
TEST_DIR = "/home/andrew/.cache/kagglehub/datasets/mozillaorg/common-voice/versions/2/cv-valid-test/cv-valid-test"
TEST_CHUNKS = "/home/andrew/.cache/kagglehub/datasets/mozillaorg/common-voice/versions/2/cv-valid-test/chunks"

# Check if data has been chunked previously
if not Path(TEST_CHUNKS).exists():
    print("CHUNKING DATASET")
    chunk_dataset(TEST_DIR, TEST_CHUNKS, CHUNK_SIZE)

# Load Dataset
print("Loading Dataset: ")
dataset = DeclipDataset(TEST_CHUNKS)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# Initialize Model, Loss, Optimizer
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")
criterion = torch.nn.MSELoss()

saved_models = ["./031626_EPOCH_1.pth", "./031626_EPOCH_2.pth", "./031626_EPOCH_3.pth", "./031626_EPOCH_4.pth",
                "./031626_EPOCH_5.pth", "./031626_EPOCH_6.pth", "./031626_EPOCH_7.pth", "./031626_EPOCH_8.pth"]

for model_path in saved_models:
    print("Loading Model: " + model_path)
    checkpoint = torch.load(model_path)
    model = DeclipCNN()  # instantiate your model first
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    model.eval()  # Set model to evaluation mode

    test_loss = 0.0
    count = 0

    with torch.no_grad():  # Disable gradient computation
        for wave_clipped, wave_original in dataloader:
            wave_clipped, wave_original = wave_clipped.to(device), wave_original.to(device)

            count += 1
            output = model(wave_clipped)
            loss = criterion(output, wave_original)

            test_loss += loss.item()

            del wave_clipped, wave_original, output, loss

    avg_test_loss = test_loss / count
    print(f"Test Loss: {avg_test_loss:.6f}")