import lightning as L
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
from pathlib import Path
import librosa
import glob

from utils import *


default_label_mapping = {
    "bonafide": 0,
    "replay_bonafide": 1,
    "fake": 2,
    "replay_fake": 3,
}


class HuggingFaceAudioDataset(Dataset):
    def __init__(
        self, hf_dataset, max_len=64000, pad_mode="random", label_mapping=None
    ):
        self.dataset = hf_dataset
        self.max_len = max_len
        if pad_mode == "random":
            self.pad = pad_random
        else:
            self.pad = pad

        self.label_mapping = label_mapping or default_label_mapping

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        audio = item["path"]
        waveform = self.pad(audio["array"], max_len=self.max_len)
        waveform = torch.tensor(waveform, dtype=torch.float32)
        label = torch.tensor(self.label_mapping[item["label"]], dtype=torch.long)
        return waveform, label


class EchoFake(L.LightningDataModule):
    def __init__(
        self,
        dataset_path,
        batch_size=8,
        sample_rate=16000,
        num_workers=4,
        max_len=64000,
        label_mapping=None,
        test_mode="closed",
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.num_workers = num_workers
        self.max_len = max_len
        self.label_mapping = label_mapping
        self.test_mode = test_mode

    def setup(self, stage=None):
        dataset = load_from_disk(self.dataset_path)

        self.train_dataset = HuggingFaceAudioDataset(
            dataset["train"], self.max_len, label_mapping=self.label_mapping
        )
        self.val_dataset = HuggingFaceAudioDataset(
            dataset["dev"],
            self.max_len,
            pad_mode="normal",
            label_mapping=self.label_mapping,
        )

        if "close" in self.test_mode:
            ds = dataset["closed_set_eval"]
        else:
            ds = dataset["open_set_eval"]
        self.test_dataset = HuggingFaceAudioDataset(
            ds, self.max_len, pad_mode="normal", label_mapping=self.label_mapping
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class ASVspoof2019Dataset(Dataset):
    def __init__(
        self,
        base_dir,
        protocol_dir,
        pad_mode="random",
        max_len=64000,
        config=None,
    ):
        self.list_IDs = []
        self.labels = []
        self.base_dir = base_dir
        self.protocol_dir = protocol_dir
        self.config = config
        with open(self.protocol_dir, "r") as f:
            lines = f.readlines()
            for line in lines:
                # example line: LA_0098 LA_T_9779812 - - bonafide
                _, key, _, _, label = line.strip().split(" ")
                self.list_IDs.append(key)

                if label == "bonafide":
                    self.labels.append(1)
                else:
                    self.labels.append(0)

        assert pad_mode in ["normal", "random"]
        if pad_mode == "random":
            self.pad = pad_random
        else:
            self.pad = pad

        self.max_len = max_len

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        x = torchaudio.load(str(self.base_dir / f"flac/{key}.flac"))[0].squeeze(0)
        x = torch.Tensor(self.pad(x, self.max_len))
        y = torch.LongTensor([1]) if self.labels[index] == 1 else torch.LongTensor([0])
        y = y.squeeze()
        return x, y


class ASVspoof2019(L.LightningDataModule):
    def __init__(self, dataset_path, track="LA", max_len=64000, **dataloaderArgs):
        """
        ASVspoof 2019 LA datamodule for training

        """
        super().__init__()
        self.max_len = max_len
        self.track = track
        self.base_dir = Path(dataset_path) / track
        assert track in ["LA", "PA"]
        self.protocol_dir = self.base_dir / f"ASVspoof2019_{track}_cm_protocols"
        self.train_cm = self.protocol_dir / f"ASVspoof2019.{track}.cm.train.trn.txt"
        self.dev_cm = self.protocol_dir / f"ASVspoof2019.{track}.cm.dev.trl.txt"
        self.eval_cm = self.protocol_dir / f"ASVspoof2019.{track}.cm.eval.trl.txt"
        self.dev_asv_scores = (
            self.base_dir
            / f"ASVspoof2019_{track}_asv_scores"
            / f"ASVspoof2019.{track}.asv.dev.gi.trl.scores.txt"
        )
        self.eval_asv_scores = (
            self.base_dir
            / f"ASVspoof2019_{track}_asv_scores"
            / f"ASVspoof2019.{track}.asv.eval.gi.trl.scores.txt"
        )
        self.dataloaderArgs = dataloaderArgs

    def setup(self, stage: str):
        # read ASVSpoof 2019 LA protocol
        dataset_dir = self.base_dir / f"ASVspoof2019_{self.track}_train"
        self.train_data = ASVspoof2019Dataset(
            dataset_dir,
            self.train_cm,
            pad_mode="random",
            max_len=self.max_len,
        )

        dataset_dir = self.base_dir / f"ASVspoof2019_{self.track}_dev"
        self.val_data = ASVspoof2019Dataset(
            dataset_dir, self.dev_cm, pad_mode="normal", max_len=self.max_len
        )

        dataset_dir = self.base_dir / f"ASVspoof2019_{self.track}_eval"
        self.test_data = ASVspoof2019Dataset(
            dataset_dir, self.eval_cm, pad_mode="normal", max_len=self.max_len
        )

    def train_dataloader(self):
        return DataLoader(self.train_data, shuffle=True, **self.dataloaderArgs)

    def val_dataloader(self):
        return DataLoader(self.val_data, shuffle=False, **self.dataloaderArgs)

    def test_dataloader(self):
        return DataLoader(self.test_data, shuffle=False, **self.dataloaderArgs)


class ASVspoofEvalDataset(Dataset):
    def __init__(self, base_dir, protocol_dir, pad_mode="random", max_len=64000):
        """
        ASVspoof 2019LA / 2021LA / 2021DF datamodule for evaluation
        """
        self.base_dir = base_dir
        self.protocol_dir = protocol_dir
        self.list_IDs = []
        self.labels = []
        assert pad_mode in ["normal", "random"]
        if pad_mode == "random":
            self.pad = pad_random
        else:
            self.pad = pad
        self.max_len = max_len
        self.parse_protocol()

    def parse_protocol(self):
        with open(self.protocol_dir, "r") as f:
            lines = f.readlines()
            for line in lines:
                _, key, _, _, label = line.strip().split(" ")
                self.list_IDs.append(key)
                if label == "bonafide":
                    self.labels.append(1)
                else:
                    self.labels.append(0)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        audio_path = str(self.base_dir / f"flac/{key}.flac")
        x = torchaudio.load(audio_path)[0].squeeze()
        x = torch.Tensor(self.pad(x, self.max_len))
        y = torch.LongTensor([1]) if self.labels[index] == 1 else torch.LongTensor([0])
        y = y.squeeze()
        return x, y


class ASVspoofEval(L.LightningDataModule):
    def __init__(self, base_dir, max_len=64000, **dataloaderArgs):
        super().__init__()
        self.base_dir = Path(base_dir)
        self.protocol_dir = self.base_dir / "labels.txt"
        self.max_len = max_len
        self.dataloaderArgs = dataloaderArgs

    def setup(self, stage: str):
        self.testset = ASVspoofEvalDataset(
            self.base_dir, self.protocol_dir, max_len=self.max_len
        )

    def test_dataloader(self):
        return DataLoader(self.testset, shuffle=False, **self.dataloaderArgs)


class IntheWildDataset(Dataset):
    def __init__(self, wav_dir, protocol_dir, pad_mode="random", max_len=64000):
        """
        In-the-Wild datamodule for evaluation
        """
        self.wav_dir = wav_dir
        self.protocol_dir = protocol_dir
        self.list_IDs = []
        self.labels = []
        if pad_mode == "random":
            self.pad = pad_random
        else:
            self.pad = pad
        self.max_len = max_len
        self.parse_protocol()

    def parse_protocol(self):
        with open(self.protocol_dir, "r") as f:
            lines = f.readlines()
            for line in lines:
                _, key, _, _, label = line.strip().split(" ")
                self.list_IDs.append(key)
                if label == "bonafide":
                    self.labels.append(1)
                else:
                    self.labels.append(0)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        x = torchaudio.load(str(self.wav_dir / f"{key}.wav"))[0].squeeze()
        x = torch.Tensor(self.pad(x, self.max_len))
        y = torch.LongTensor([1]) if self.labels[index] == 1 else torch.LongTensor([0])
        y = y.squeeze()
        return x, y


class IntheWild(L.LightningDataModule):
    def __init__(self, base_dir, max_len=64000, **dataloaderArgs):
        super().__init__()
        base_dir = Path(base_dir)
        self.wav_dir = base_dir / "release_in_the_wild"
        self.protocol_dir = base_dir / "wild_labels.txt"
        self.max_len = max_len
        self.dataloaderArgs = dataloaderArgs

    def setup(self, stage: str):
        self.trainset = IntheWildDataset(
            self.wav_dir, self.protocol_dir, pad_mode="random", max_len=self.max_len
        )
        self.valset = IntheWildDataset(
            self.wav_dir, self.protocol_dir, pad_mode="random", max_len=self.max_len
        )
        self.testset = IntheWildDataset(
            self.wav_dir, self.protocol_dir, pad_mode="normal", max_len=self.max_len
        )

    def train_dataloader(self):
        return DataLoader(self.trainset, shuffle=True, **self.dataloaderArgs)

    def val_dataloader(self):
        return DataLoader(self.valset, shuffle=False, **self.dataloaderArgs)

    def test_dataloader(self):
        return DataLoader(self.testset, shuffle=False, **self.dataloaderArgs)


class WaveFakeDataset(Dataset):
    def __init__(self, base_dir, pad_mode="random", max_len=64000):
        """
        In-the-Wild datamodule for evaluation
        """
        self.base_dir = base_dir
        self.wav_paths = []
        self.labels = []
        if pad_mode == "random":
            self.pad = pad_random
        else:
            self.pad = pad
        self.max_len = max_len
        self.parse_protocol()

    def parse_protocol(self):
        bonafide_list = glob.glob(str(self.base_dir / "wavs16" / "*.wav"))
        fake_list = glob.glob(str(self.base_dir / "generated_audio" / "*" / "*.wav"))
        for path in bonafide_list:
            self.wav_paths.append(path)
            self.labels.append(1)
        for path in fake_list:
            self.wav_paths.append(path)
            self.labels.append(0)

    def __len__(self):
        return len(self.wav_paths)

    def __getitem__(self, index):
        path = self.wav_paths[index]
        x, _ = librosa.load(path, sr=16000)
        x = torch.Tensor(self.pad(x, self.max_len))
        y = torch.LongTensor([1]) if self.labels[index] == 1 else torch.LongTensor([0])
        y = y.squeeze()
        return x, y


class WaveFake(L.LightningDataModule):
    def __init__(self, base_dir, max_len=64000, **dataloaderArgs):
        super().__init__()
        self.base_dir = Path(base_dir)
        self.max_len = max_len
        self.dataloaderArgs = dataloaderArgs

    def setup(self, stage: str):
        self.trainset = WaveFakeDataset(
            self.base_dir, pad_mode="random", max_len=self.max_len
        )
        self.valset = WaveFakeDataset(
            self.base_dir, pad_mode="random", max_len=self.max_len
        )
        self.testset = WaveFakeDataset(
            self.base_dir, pad_mode="normal", max_len=self.max_len
        )

    def train_dataloader(self):
        return DataLoader(self.trainset, shuffle=True, **self.dataloaderArgs)

    def val_dataloader(self):
        return DataLoader(self.valset, shuffle=False, **self.dataloaderArgs)

    def test_dataloader(self):
        return DataLoader(self.testset, shuffle=False, **self.dataloaderArgs)
