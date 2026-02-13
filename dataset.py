import torch
from torch.utils.data import Dataset
from preprocess import process_audio
import glob


MAX_LEN = 1600

class DeclipDataset(Dataset):
    def __init__(self, data_dir):
        self.files = glob.glob(f"{data_dir}/*.mp3")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            file_path = self.files[idx]
            spec_clipped, spec_original = process_audio(file_path)
            return spec_clipped, spec_original
        except Exception as e:
            print(f"[WARNING] Skipping file {self.files[idx]} due to error: {e}")

            new_idx = torch.randint(0, len(self), (1,)).item()
            return self.__getitem__(new_idx)


def collate_fn(batch):
    inputs, targets = zip(*batch)

    lengths = (max(inp.shape[2], tgt.shape[2]) for inp, tgt in zip(inputs, targets))
    max_len = max(lengths)

    max_len = min(max_len, MAX_LEN)
    if max_len % 2 != 0:
        max_len += 1

    # print("Max Length: " + str(max_len))

    padded_inputs = []
    padded_targets = []

    for inp, tgt in zip(inputs, targets):
        inp_len = inp.shape[2]
        tgt_len = tgt.shape[2]
        # print("input length " + str(inp_len))
        # print("target length " + str(tgt_len))

        # Pad both to max_len along time dimension
        pad_inp = torch.nn.functional.pad(inp, (0, max_len - inp_len))
        pad_tgt = torch.nn.functional.pad(tgt, (0, max_len - tgt_len))

        # print("pad_inp " + str(pad_inp.shape[2]))
        # print("pad_tgt " + str(pad_tgt.shape[2]))

        padded_inputs.append(pad_inp)
        padded_targets.append(pad_tgt)

    return torch.stack(padded_inputs), torch.stack(padded_targets)

