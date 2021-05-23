import torch
import numpy as np
from scipy.io import wavfile
from utils import mu_law_encoding


class WaveNet_Dataset(torch.utils.data.Dataset):
    """ Creates a dataset that can be used to train and validate WaveNet. """
    def __init__(
            self, 
            track_list: str, 
            x_len: int, 
            y_len: int = 1, 
            bitrate: int = 16, 
            twos_comp: bool = True, 
            num_classes: int = 256, 
            store_tracks: bool = False, 
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

            # Check for encoder and define it if is none
            if encoder is None:
                self.encoder = mu_law_encoding

            # Load each track in list
            for track in track_list:
                audio, dtype, sample_rate = self._load_audio_from_wav(track)

                # Store track if requested
                if store_tracks:
                    self.tracks.append({
                        'name': track,
                        'audio': audio,
                        'sample_rate': sample_rate
                    })

                # Process data
                for i in range(0, len(audio) - x_len, x_len + y_len):
                    x, y = self._extract_segment(audio, x_len, y_len, start_idx=i)
                    x, y = self.preprocess(x, y)
                    self.data.append({ 'x': x, 'y': y })

            self.dtype = dtype
            self.sample_rate = sample_rate

        
    def __len__(self):
        """ Redefine __len__ special method. """
        return len(self.data)

    def __getitem__(self, idx: int):
        """ Redefine __getitem__ special method. """
        return self._to_tensor(self.data[idx]['x'], self.data[idx]['y'])

    def _load_audio_from_wav(self, filename: str):
        """ Loads the audio files. """
        # Read audio
        sample_rate, audio = wavfile.read(filename)
        # Check for audio type
        assert(audio.dtype == 'int16')
        dtype = audio.dtype

        # Combine channels
        audio = np.array(audio)

        # Get mean if audio size greater than 1
        if len(audio.shape) > 1:
            audio = np.mean(audio, 1)

        return audio, dtype, sample_rate

    def _extract_segment(
        self, 
        audio: np.array, 
        x_len: int, 
        y_len: int, 
        start_idx=None
    ):
        """ Extract a segment from the given audio. """
        # Get number of samples
        num_samples = audio.shape[0]
        # Get number of points
        num_points = x_len + y_len
        
        # Init initial index if needed
        if start_idx is None:
            # Select random index in range from 0 to num_samples - num_points
            start_idx = np.random.randint(0, num_samples - num_points, 1)[0]

        # Extract segment
        x = audio[start_idx:start_idx + x_len]
        y = audio[start_idx + x_len:start_idx + x_len + y_len]

        return x, y

    def _to_tensor(
        self, 
        x: np.array, 
        y=None
    ) -> torch.Tensor:
        """ Returns the given arrays(s) into torch.tensor """
        # Create tensor from input parameter
        x = torch.tensor(x, dtype=torch.float32)

        # Unsqueeze tensor if needed
        if len(x.shape) < 2:
            x = torch.unsqueeze(x, 0)

        # If second array is given then transform it into tensor and create tuple with first tensor
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
        """ Save audio to given file. """
        if sample_rate is None:
            sample_rate = self.sample_rate

        if dtype is None:
            dtype = self.dtype

        data = data.astype(dtype)

        return wavfile.write(filename, self.sample_rate, data)

    def label2value(self, label):
        return self.bins[label.astype(int)]

    def preprocess(self, x, y=None, num_classes=256):
        """ Preprocess the given input using Mu-law or another encoder"""
        # Encode passed input and quantize
        x = self.encoder(x, num_classes)
        x = self._quantize(x)

        # If second input is given then encode and quantize it, and create tuple with first tensor
        if y is not None:
            y = self.encoder(x, num_classes)
            y = self._quantize(y, label=True)
            out = (x, y)
        else:
            out = x

        return out


class AudioBatchSampler(torch.utils.data.sampler.BatchSampler):
    def __init__(self, dataset, batch_size, drop_last=True) -> None:
        """ Randomly samples the given dataset. """
        super().__init__(
            torch.utils.data.sampler.RandomSampler(dataset), 
            batch_size, 
            drop_last
        )

class AudioLoader(torch.utils.data.DataLoader):
    """ Returns a dataloader given a dataset. """
    def __init__(self, dataset, batch_size=8, drop_last=True, num_workers=1):
        sampler = AudioBatchSampler(dataset, batch_size, drop_last)
        super().__init__(dataset, batch_sampler=sampler, num_workers=num_workers)
