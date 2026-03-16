import torch
from torch.utils.data import Dataset
from preprocess import process_audio
import glob


# MAX_LEN = 220500
CHUNK_SIZE = 16384


class DeclipDataset(Dataset):
    def __init__(self, data_dir):

        self.files = glob.glob(f"{data_dir}/*.mp3")
        self.chunks = []

        # for file in self.files:
        #     waveform = process_audio(file, return_waveform_only=True)
        #     length = waveform.shape[-1]
        #
        #     for start in range(0, length - CHUNK_SIZE, CHUNK_SIZE):
        #         self.chunks.append((file, start))
        # print(len(self.chunks))

    def __len__(self):
        return len(self.files)

    # def __getitem__(self, idx):
    #     try:
    #         file, start = self.chunks[idx]
    #         _spec_clipped, _spec_original, waveform_clipped, waveform_original = process_audio(file)
    #         return waveform_clipped, waveform_original
    #     except Exception as e:
    #         print(f"[WARNING] Skipping file {self.files[idx]} due to error: {e}")
    #
    #         new_idx = torch.randint(0, len(self), (1,)).item()
    #         return self.__getitem__(new_idx)

    def __getitem__(self, idx):

        _spec_clip, _spec_orig, wav_clip, wav_orig = process_audio(self.files[idx])

        # clipped = induce_clipping_from_tensor(waveform)

        return wav_clip, wav_orig


# def collate_fn(batch):
#     inputs, targets = zip(*batch)
#     print(inp.shape for inp, tgt in zip(inputs, targets))
#
#     lengths = (max(inp.shape[2], tgt.shape[2]) for inp, tgt in zip(inputs, targets))
#     max_len = max(lengths)
#
#     max_len = min(max_len, MAX_LEN)
#     if max_len % 2 != 0:
#         max_len += 1
#
#     # print("Max Length: " + str(max_len))
#
#     padded_inputs = []
#     padded_targets = []
#
#     for inp, tgt in zip(inputs, targets):
#         inp_len = inp.shape[2]
#         tgt_len = tgt.shape[2]
#         # print("input length " + str(inp_len))
#         # print("target length " + str(tgt_len))
#
#         # Pad both to max_len along time dimension
#         pad_inp = torch.nn.functional.pad(inp, (0, max_len - inp_len))
#         pad_tgt = torch.nn.functional.pad(tgt, (0, max_len - tgt_len))
#
#         # print("pad_inp " + str(pad_inp.shape[2]))
#         # print("pad_tgt " + str(pad_tgt.shape[2]))
#
#         padded_inputs.append(pad_inp)
#         padded_targets.append(pad_tgt)
#
#     return torch.stack(padded_inputs), torch.stack(padded_targets)


# def collate_fn(batch):
#     inputs, targets = zip(*batch)
#
#     max_len = max(inp.shape[-1] for inp in inputs)
#     max_len = min(max_len, MAX_LEN)
#     print("max_len " + str(max_len))
#
#     padded_inputs = []
#     padded_targets = []
#
#     count = 0
#     for inp, tgt in zip(inputs, targets):
#         pad_inp = max_len - inp.shape[-1]
#         pad_tgt = max_len - tgt.shape[-1]
#         inp = torch.nn.functional.pad(inp, (0, pad_inp))
#         tgt = torch.nn.functional.pad(tgt, (0, pad_tgt))
#
#         padded_inputs.append(inp)
#         padded_targets.append(tgt)
#         print(padded_inputs[count].shape)
#         print(padded_targets[count].shape)
#         count += 1
#
#     return torch.stack(padded_inputs), torch.stack(padded_targets)


