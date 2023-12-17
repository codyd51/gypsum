import datetime
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

from gypsum.config import UTC_LEAP_SECONDS_COUNT
from gypsum.constants import PRN_REPETITIONS_PER_SECOND
from gypsum.radio_input import InputFileInfo
from gypsum.units import SampleCount
from gypsum.units import Seconds

# Expressed as seconds since the UTC epoch, as measured by the local clock (i.e. including the receiver clock bias)
ReceiverTimestampSeconds = Seconds

_logger = logging.getLogger(__name__)


@dataclass
class SampleProviderAttributes:
    samples_per_second: SampleCount
    # Note that since this is an integer, the sample rate must be an integer multiple of the PRN chips per second count
    samples_per_prn_transmission: SampleCount


class AntennaSampleProvider(ABC):
    @abstractmethod
    def get_samples(self, sample_count: SampleCount) -> Tuple[ReceiverTimestampSeconds, np.ndarray]:
        ...

    @abstractmethod
    def peek_samples(self, sample_count: SampleCount) -> Tuple[ReceiverTimestampSeconds, np.ndarray]:
        ...

    @abstractmethod
    def seconds_since_start(self) -> Seconds:
        ...

    @abstractmethod
    def get_attributes(self) -> SampleProviderAttributes:
        ...


class AntennaSampleProviderBackedByBytes(AntennaSampleProvider):
    def __init__(self, data: np.ndarray) -> None:
        self.data = data
        self.cursor = 0
        raise NotImplementedError(f'This provider must be reworked to track the passage of time.')

    def peek_samples(self, sample_count: SampleCount) -> Tuple[ReceiverTimestampSeconds, np.ndarray]:
        raise NotImplementedError(f'This provider must be reworked to track the passage of time.')
        return self.data[self.cursor : self.cursor + sample_count]

    def get_samples(self, sample_count: SampleCount) -> Tuple[ReceiverTimestampSeconds, np.ndarray]:
        raise NotImplementedError(f'This provider must be reworked to track the passage of time.')
        data = self.peek_samples(sample_count)
        self.cursor += sample_count
        return data

    def get_attributes(self) -> SampleProviderAttributes:
        raise NotImplementedError()


class AntennaSampleProviderBackedByFile(AntennaSampleProvider):
    def __init__(self, file_info: InputFileInfo) -> None:
        self.path = file_info.path
        self.cursor = 0
        self.sample_rate = file_info.sdr_sample_rate
        self.utc_start_time = file_info.utc_start_time.timestamp()
        self.sample_component_data_type = file_info.sample_component_data_type

    def seconds_since_start(self) -> Seconds:
        timestamp_in_seconds_since_start = self.cursor / self.sample_rate
        return timestamp_in_seconds_since_start

    def peek_samples(self, sample_count: SampleCount) -> Tuple[ReceiverTimestampSeconds, np.ndarray]:
        # The timestamp is always taken at the start of this set of samples
        # TODO(PT): SAMPLES_PER_SECOND should be an instance attribute
        receiver_utc_timestamp = self.utc_start_time + self.seconds_since_start()
        # GPS differs from UTC by an integer number of leap seconds.
        receiver_gps_timestamp = receiver_utc_timestamp + UTC_LEAP_SECONDS_COUNT

        words = np.fromfile(
            self.path.as_posix(),
            dtype=self.sample_component_data_type,
            # We have interleaved IQ samples, so the number of words to read will be the sample count * 2
            count=sample_count * 2,
            # Note the change in units: `count` is specified in terms of `dtype`, while `offset` is in bytes.
            offset=self.cursor * 2 * np.dtype(self.sample_component_data_type).itemsize,
        )
        # Recombine the inline IQ samples into complex values
        iq_samples = (words[0::2]) + (1j * words[1::2])
        return receiver_gps_timestamp, iq_samples

    def get_samples(self, sample_count: SampleCount) -> Tuple[ReceiverTimestampSeconds, np.ndarray]:
        receiver_timestamp, file_data = self.peek_samples(sample_count)
        self.cursor += sample_count
        return receiver_timestamp, file_data

    def samples_per_prn_transmission(self) -> SampleCount:
        return self.sample_rate // PRN_REPETITIONS_PER_SECOND

    def get_attributes(self) -> SampleProviderAttributes:
        return SampleProviderAttributes(
            samples_per_second=self.sample_rate,
            # PT: Note this must be an integer factor (though if we resample the PRN this limitation might go away)
            samples_per_prn_transmission=self.sample_rate // PRN_REPETITIONS_PER_SECOND,
        )
