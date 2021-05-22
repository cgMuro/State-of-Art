import torch
import numpy as np
from scipy.io import wavfile
from utils import mu_law_encoding


class WaveNet_Dataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            track_list, 
            x_len, 
            y_len=1, 
            bitrate=16, 
            twos_comp=True, 
            num_classes=256, 
            store_tracks=False, 
            encoder=None
        ) -> None:
            super().__init__()

            self.data = []
            self.tracks = []
            self.x_len = x_len
            self.y_len = y_len
            self.num_channels = 1
            self.num_classes = num_classes
            self.bitrate = bitrate
            self.datarange = (-2**(bitrate - 1), 2**(bitrate - 1) - 1)
            self.twos_comp = twos_comp
            self.bins = np.linspace(-1, 1, num_classes)


            if encoder is None:
                self.encoder = mu_law_encoding

            for track in track_list:
                audio, dtype, sample_rate = self._load_audio_from_wav(track)

                if store_tracks:
                    self.tracks.append({
                        'name': track,
                        'audio': audio,
                        'sample_rate': sample_rate
                    })

                for i in range(0, len(audio) - x_len, x_len + y_len):
                    x, y = self._extract_segment(audio, x_len, y_len, start_idx=i)
                    x, y = self.preprocess(x, y)
                    self.data.append({ 'x': x, 'y': y })

            self.dtype = dtype
            self.sample_rate = sample_rate

        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self._to_tensor(self.data[idx]['x'], self.data[idx]['y'])

    def _load_audio_from_wav(self, filename):
        # Read audio
        sample_rate, audio = wavfile.read(filename)
        assert(audio.dtype == 'int16')
        dtype = audio.dtype

        # Combine channels
        audio = np.array(audio)
        if len(audio.shape) > 1:
            audio = np.mean(audio, 1)

        return audio, dtype, sample_rate

    def _extract_segment(self, audio, x_len, y_len, start_idx=None):
        num_samples = audio.shape[0]
        num_points = x_len + y_len

        if start_idx is None:
            # Select random index in range from 0 to num_samples - num_points
            start_idx = np.random.randint(0, num_samples - num_points, 1)[0]

        # Extract segment
        x = audio[start_idx:start_idx + x_len]
        y = audio[start_idx + x_len:start_idx + x_len + y_len]

        return x, y

    def _to_tensor(self, x, y=None):
        x = torch.tensor(x, dtype=torch.float32)

        if len(x.shape) < 2:
            x = torch.unsqueeze(x, 0)

        if y is not None:
            y = torch.tensor(y, dtype=torch.long)
            out = (x, y)
        else:
            out = x

        return out

    def _quantize(self, x, label=False):
        out = np.digitize(x, self.bins, right=False) - 1

        if not label:
            out = self.bins[out]

        return out

    def save_wav(self, filename, data, sample_rate=None, dtype=None):
        if sample_rate is None:
            sample_rate = self.sample_rate

        if dtype is None:
            dtype = self.dtype

        data = data.astype(dtype)

        return wavfile.write(filename, self.sample_rate, data)

    def label2value(self, label):
        return self.bins[label.astype(int)]

    def preprocess(self, x, y=None, mu=255):
        x = self.encoder(x, mu)
        x = self._quantize(x)

        if y is not None:
            y = self.encoder(x, mu)
            y = self._quantize(y, label=True)
            out = (x, y)
        else:
            out = x

        return out


class AudioBatchSampler(torch.utils.data.sampler.BatchSampler):
    def __init__(self, dataset, batch_size, drop_last=True) -> None:
        super().__init__(
            torch.utils.data.sampler.RandomSampler(dataset), 
            batch_size, 
            drop_last
        )

class AudioLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=8, drop_last=True, num_workers=1):
        sampler = AudioBatchSampler(dataset, batch_size, drop_last)
        super().__init__(dataset, batch_sampler=sampler, num_workers=num_workers)
