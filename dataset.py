from torch.utils.data import Dataset
from preprocess import process_audio
import glob


# MAX_LEN = 220500
CHUNK_SIZE = 16384


class DeclipDataset(Dataset):
    def __init__(self, data_dir):

        self.files = glob.glob(f"{data_dir}/*.mp3")
        self.chunks = []


    def __len__(self):
        return len(self.files)


    def __getitem__(self, idx):

        _spec_clip, _spec_orig, wav_clip, wav_orig = process_audio(self.files[idx])

        return wav_clip, wav_orig



