import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from gypsum.constants import PRN_REPETITIONS_PER_SECOND
from gypsum.radio_input import InputFileInfo
from gypsum.units import ReceiverDataSeconds
from gypsum.units import SampleCount, Seconds

# Expressed as seconds since the radio started delivering samples,
# as measured by the local clock (i.e. including the receiver clock bias)
ReceiverTimestampSeconds = Seconds

_logger = logging.getLogger(__name__)


class NoMoreSamplesError(Exception):
    pass


@dataclass
class SampleProviderAttributes:
    samples_per_second: SampleCount
    # Note that since this is an integer, the sample rate must be an integer multiple of the PRN chips per second count
    samples_per_prn_transmission: SampleCount


@dataclass
class AntennaSampleChunk:
    start_time: ReceiverTimestampSeconds
    end_time: ReceiverTimestampSeconds
    samples: np.ndarray


class AntennaSampleProvider(ABC):
    @abstractmethod
    def get_samples(self, sample_count: SampleCount) -> AntennaSampleChunk:
        ...

    @abstractmethod
    def peek_samples(self, sample_count: SampleCount) -> AntennaSampleChunk:
        ...

    @abstractmethod
    def seconds_since_start(self) -> ReceiverDataSeconds:
        ...

    @abstractmethod
    def get_attributes(self) -> SampleProviderAttributes:
        ...


class AntennaSampleProviderBackedByBytes(AntennaSampleProvider):
    def __init__(self, data: np.ndarray) -> None:
        self.data = data
        self.cursor = 0
        raise NotImplementedError(f"This provider must be reworked to track the passage of time.")

    def peek_samples(self, sample_count: SampleCount) -> Tuple[ReceiverTimestampSeconds, np.ndarray]:
        raise NotImplementedError(f"This provider must be reworked to track the passage of time.")
        return self.data[self.cursor : self.cursor + sample_count]

    def get_samples(self, sample_count: SampleCount) -> Tuple[ReceiverTimestampSeconds, np.ndarray]:
        raise NotImplementedError(f"This provider must be reworked to track the passage of time.")
        data = self.peek_samples(sample_count)
        self.cursor += sample_count
        return data

    def get_attributes(self) -> SampleProviderAttributes:
        raise NotImplementedError()

    def seconds_since_start(self) -> ReceiverDataSeconds:
        raise NotImplementedError()


class AntennaSampleProviderBackedByFile(AntennaSampleProvider):
    def __init__(self, file_info: InputFileInfo) -> None:
        self.path = file_info.path
        self.cursor = 0
        self.sample_rate = file_info.sdr_sample_rate
        self.utc_start_time = file_info.utc_start_time.timestamp()
        self.sample_component_data_type = file_info.sample_component_data_type
        self.file_size_in_bytes = self.path.stat().st_size

    def _get_elapsed_seconds_at_cursor(self, cursor: int) -> ReceiverTimestampSeconds:
        return round(cursor / self.sample_rate, 6)

    def seconds_since_start(self) -> Seconds:
        return self._get_elapsed_seconds_at_cursor(self.cursor)

    def peek_samples(self, sample_count: SampleCount) -> AntennaSampleChunk:
        # receiver_utc_timestamp = self.utc_start_time + self.seconds_since_start()
        # The timestamp is always taken at the start of this set of samples
        # Note GPS differs from UTC by an integer number of leap seconds (and the epoch slide).
        start_timestamp = self.seconds_since_start()

        # We have interleaved IQ samples, so the number of words to read will be the sample count * 2
        word_count_to_read = sample_count * 2
        # Note the change in units: `count` is specified in terms of `dtype`, while `offset` is in bytes.
        bytes_per_word = np.dtype(self.sample_component_data_type).itemsize
        file_offset_start = self.cursor * 2 * bytes_per_word
        file_offset_end = file_offset_start + (word_count_to_read * bytes_per_word)

        if file_offset_end >= self.file_size_in_bytes:
            raise NoMoreSamplesError(
                f'Ran out of samples at {self.file_size_in_bytes/1024/1024:.2f}MB ({self.seconds_since_start():.2f}s)'
            )

        words = np.fromfile(
            self.path.as_posix(),
            dtype=self.sample_component_data_type,
            count=word_count_to_read,
            offset=file_offset_start,
        )
        # Recombine the inline IQ samples into complex values
        iq_samples = (words[0::2]) + (1j * words[1::2])
        return AntennaSampleChunk(
            start_time=start_timestamp,
            end_time=self._get_elapsed_seconds_at_cursor(self.cursor + sample_count),
            samples=iq_samples,
        )

    def get_samples(self, sample_count: SampleCount) -> AntennaSampleChunk:
        chunk = self.peek_samples(sample_count)
        self.cursor += sample_count
        return chunk

    def get_attributes(self) -> SampleProviderAttributes:
        return SampleProviderAttributes(
            samples_per_second=int(self.sample_rate),
            # PT: Note this must be an integer factor (though if we resample the PRN this limitation might go away)
            samples_per_prn_transmission=int(self.sample_rate // PRN_REPETITIONS_PER_SECOND),
        )
