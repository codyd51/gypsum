import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple

import numpy as np

from gypsum.constants import SAMPLES_PER_SECOND

ReceiverTimestampSeconds = float

_logger = logging.getLogger(__name__)


class AntennaSampleProvider(ABC):
    @abstractmethod
    def get_samples(self, sample_count: int) -> Tuple[ReceiverTimestampSeconds, np.ndarray]:
        ...

    @abstractmethod
    def peek_samples(self, sample_count: int) -> Tuple[ReceiverTimestampSeconds, np.ndarray]:
        ...


class AntennaSampleProviderBackedByBytes(AntennaSampleProvider):
    def __init__(self, data: np.ndarray) -> None:
        raise NotImplementedError(f'This provider must be reworked to track the passage of time.')
        self.data = data
        self.cursor = 0

    def peek_samples(self, sample_count: int) -> Tuple[ReceiverTimestampSeconds, np.ndarray]:
        return self.data[self.cursor : self.cursor + sample_count]

    def get_samples(self, sample_count: int) -> Tuple[ReceiverTimestampSeconds, np.ndarray]:
        data = self.peek_samples(sample_count)
        self.cursor += sample_count
        return data


class AntennaSampleProviderBackedByFile(AntennaSampleProvider):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.cursor = 0

    def peek_samples(self, sample_count: int) -> Tuple[ReceiverTimestampSeconds, np.ndarray]:
        # The timestamp is always taken at the start of this set of samples
        # TODO(PT): SAMPLES_PER_SECOND should be an instance attribute
        receiver_timestamp_seconds = self.cursor / SAMPLES_PER_SECOND

        words = np.fromfile(
            self.path.as_posix(),
            dtype=np.float32,
            # We have interleaved IQ samples, so the number of words to read will be the sample count * 2
            count=sample_count * 2,
            # Note the change in units: `count` is specified in terms of `dtype`, while `offset` is in bytes.
            offset=self.cursor * 2 * np.dtype(np.float32).itemsize,
        )
        # Recombine the inline IQ samples into complex values
        iq_samples = (words[0::2]) + (1j * words[1::2])
        return (receiver_timestamp_seconds, iq_samples)

    def get_samples(self, sample_count: int) -> Tuple[ReceiverTimestampSeconds, np.ndarray]:
        receiver_timestamp, file_data = self.peek_samples(sample_count)
        self.cursor += sample_count
        return receiver_timestamp, file_data
