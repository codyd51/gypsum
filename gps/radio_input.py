from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

import numpy as np


class InputFileType(Enum):
    Raw = auto()
    Wav = auto()


@dataclass
class InputFileInfo:
    path: Path
    format: InputFileType
    sdr_sample_rate: float


# TODO(PT): In the future, this can be extended to provide an input representing a live radio
# This can also be extended to include live recordings from my radio
# TODO(PT): Rename the associated symbols to 'input sources'
INPUT_SOURCES = [
    InputFileInfo(
        path=Path(__file__).parents[1] / "vendored_signals" / "2013_04_04_GNSS_SIGNAL_at_CTTC_SPAIN.dat",
        format=InputFileType.Raw,
        sdr_sample_rate=4e6,
    ),
]


def get_samples_from_radio_input_source(input_info: InputFileInfo) -> np.ndarray:
    if input_info.format == InputFileType.Wav:
        byte_length = np.fromfile(input_info.path.as_posix(), dtype=np.int32, count=1, offset=44)[0]
        print(byte_length)
        data = np.fromfile(input_info.path.as_posix(), dtype=np.int16, count=byte_length//2, offset=48)
        if len(data) % 2 != 0:
            data = data[:-1]
    elif input_info.format == InputFileType.Raw:
        sample_rate = input_info.sdr_sample_rate
        # Samples per PRN repeat (PRNs are transmitted by GPS satellites every 1ms)
        samples_per_prn = int(sample_rate // 1000)
        offset = 0
        # For now, just read two PRN transmissions worth of data
        prn_repeats_count = 2
        count = prn_repeats_count * samples_per_prn

        data = np.fromfile(input_info.path.as_posix(), dtype=np.int16, count=int(offset+count)*2, offset=44)
        # Quantise to 2 bits I and 2 bits Q per sample
        data = np.clip(np.floor_divide(data, 150), -2, 1) + 0.5
        # Convert to complex numpy array
        data = np.reshape(data, (data.shape[0]//2, 2))
        data = data[offset:, 0] + 1j * data[offset:, 1]
    else:
        raise ValueError(f'Unrecognized format')

    return data