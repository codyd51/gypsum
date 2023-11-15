import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

_logger = logging.getLogger(__name__)


class AntennaSampleProvider(ABC):
    @abstractmethod
    def get_samples(self, sample_count: int) -> np.ndarray:
        ...

    @abstractmethod
    def peek_samples(self, sample_count: int) -> np.ndarray:
        ...


class AntennaSampleProviderBackedByBytes(AntennaSampleProvider):
    def __init__(self, data: np.ndarray) -> None:
        self.data = data
        self.cursor = 0

    def peek_samples(self, sample_count: int) -> np.ndarray:
        return self.data[self.cursor : self.cursor + sample_count]

    def get_samples(self, sample_count: int) -> np.ndarray:
        data = self.peek_samples(sample_count)
        self.cursor += sample_count
        return data


class AntennaSampleProviderBackedByFile(AntennaSampleProvider):
    def __init__(self, path: Path) -> None:
        self.path = path
        # self.cursor = 795296568
        self.cursor = 0

    def peek_samples(self, sample_count: int) -> np.ndarray:
        words = np.fromfile(
            self.path.as_posix(),
            dtype=np.float32,
            # We have interleaved IQ samples, so the number of words to read will be the sample count * 2
            count=sample_count * 2,
            # Note the change in units: `count` is specified in terms of `dtype`, while `offset` is in bytes.
            offset=self.cursor * 2 * np.dtype(np.float32).itemsize,
        )
        # Recombine the inline IQ samples into complex values
        return (words[0::2]) + (1j * words[1::2])

    def get_samples(self, sample_count: int) -> np.ndarray:
        file_data = self.peek_samples(sample_count)
        self.cursor += sample_count
        return file_data
