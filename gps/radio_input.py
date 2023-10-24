from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

import numpy as np

from gps_project_name.gps.constants import PRN_CHIP_COUNT, PRN_REPETITIONS_PER_SECOND


class InputFileType(Enum):
    Raw = auto()
    Wav = auto()
    GnuRadioRecording = auto()


@dataclass
class InputFileInfo:
    path: Path
    format: InputFileType
    sdr_sample_rate: float

    @property
    def samples_in_prn_period(self) -> int:
        """The PRN period is 1ms, and is retransmitted by GPS satellites continuously"""
        return int(self.sdr_sample_rate // 1000)


# TODO(PT): In the future, this can be extended to provide an input representing a live radio
# This can also be extended to include live recordings from my radio
# TODO(PT): Rename the associated symbols to 'input sources'
INPUT_SOURCES = [
    InputFileInfo(
        path=Path(__file__).parents[1] / "vendored_signals" / "2013_04_04_GNSS_SIGNAL_at_CTTC_SPAIN.dat",
        format=InputFileType.Raw,
        sdr_sample_rate=4e6,
    ),
    InputFileInfo(
        path=Path(__file__).parents[1] / "vendored_signals" / "baseband_1575420000Hz_16-12-07_07-10-2023.wav",
        format=InputFileType.Wav,
        sdr_sample_rate=2.4e6,
    ),
    InputFileInfo(
        path=Path(__file__).parents[1] / "vendored_signals" / "phillip.wav",
        format=InputFileType.Wav,
        sdr_sample_rate=2.4e6,
    ),
    InputFileInfo(
        path=Path(__file__).parents[1] / "vendored_signals" / "baseband_1575136000Hz_22-52-15_10-10-2023_Int16_BiasT_TunerAGC_IQCorrection_100PPMCorrection.wav",
        format=InputFileType.Wav,
        sdr_sample_rate=2.4e6,
    ),
    InputFileInfo(
        path=Path(__file__).parents[1] / "vendored_signals" / "baseband_1575420000Hz_23-21-28_10-10-2023_2048mhz_sample_rate.wav",
        format=InputFileType.Wav,
        sdr_sample_rate=2.048e6,
    ),
    InputFileInfo(
        path=Path(__file__).parents[1] / "vendored_signals" / "output_at_seven_wives",
        format=InputFileType.GnuRadioRecording,
        # 2 * C/A PRN chip rate * 1k PRN repetitions per second
        sdr_sample_rate=2*1023*1000,
    ),
]


def get_samples_from_radio_input_source(input_info: InputFileInfo, sample_count: int) -> np.ndarray:
    samples_per_prn = input_info.samples_in_prn_period

    if input_info.format == InputFileType.Wav:
        byte_length = np.fromfile(input_info.path.as_posix(), dtype=np.int32, count=1, offset=44)[0]
        print(byte_length)
        data = np.fromfile(input_info.path.as_posix(), dtype=np.int16, count=byte_length//2, offset=48)
        if len(data) % 2 != 0:
            data = data[:-1]

        data = data[:samples_per_prn*4]

    elif input_info.format == InputFileType.Raw:
        offset = 0
        # For now, just read two PRN transmissions worth of data
        prn_repeats_count = 2
        count = prn_repeats_count * samples_per_prn

        data = np.fromfile(input_info.path.as_posix(), dtype=np.int16, count=int(offset+count)*2)
        # Quantise to 2 bits I and 2 bits Q per sample
        data = np.clip(np.floor_divide(data, 150), -2, 1) + 0.5
        # Convert to complex numpy array
        data = np.reshape(data, (data.shape[0]//2, 2))
        data = data[offset:, 0] + 1j * data[offset:, 1]

    elif input_info.format == InputFileType.GnuRadioRecording:
        # For now, read 1 second of data / 1000 repetitions of the PRN:
        # 1023 chips per PRN
        # Multiplied by 2 for Nyquist sample rate
        # Multiplied by 1000 as the PRN is retransmitted 1000 times per second
        # Multiplied by 2 as the file stores 1 word for the I component, and 1 word for the Q component
        seconds_count = 10
        words_per_iq_sample = 2
        nyquist_multiple = 2
        data = np.fromfile(
            input_info.path.as_posix(),
            dtype=np.float32,
            count=(
                words_per_iq_sample * PRN_CHIP_COUNT * PRN_REPETITIONS_PER_SECOND * nyquist_multiple * seconds_count
            )
        )
        # Recombine the inline IQ samples into complex numbers
        data = (data[0::2] + (1j * data[1::2]))
    else:
        raise ValueError(f'Unrecognized format')

    return data