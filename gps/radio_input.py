from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

import numpy as np

from gps_project_name.gps.constants import PRN_CHIP_COUNT, PRN_REPETITIONS_PER_SECOND, SAMPLES_PER_SECOND


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
    InputFileInfo(
        # Satellites that should be up now (distances taken a few minutes after recording):
        # 2  (Mag 16.9, dist 23,759km)
        # 3  (Mag 13.1, dist 20,377km)
        # 6  (Mag 11.7, dist 23,088km)
        # 9  (Mag 13.1, dist 23,036km)
        # 12 (Mag 12.0, dist 24,807km)
        # 17 (Mag 12.2, dist 21,848km)
        # 19 (Mag 11.8, dist 21,378km)
        # 21 (Mag 17.8, dist 24,931km)
        # 31 (Mag 15.0, dist 23,214km)
        #path=Path(__file__).parents[1] / "vendored_signals" / "test_output_in_office_gnu_radio",

        # 15:21pm
        # 5
        # 13
        # 14    # *** Identified satellite GpsSatelliteId(id=14) at doppler shift -2500, correlation magnitude of 8.334292029948841e-08 at 405, time offset of 0.0008020527859237536, chip offset of 820.5
        # 15
        # 18    # *** Identified satellite GpsSatelliteId(id=18) at doppler shift -1000, correlation magnitude of 9.180577185072234e-08 at 401, time offset of 0.0008040078201368523, chip offset of 822.5
        # 20    # *** Identified satellite GpsSatelliteId(id=20) at doppler shift -3500, correlation magnitude of 5.6041227612533935e-08 at 1571, time offset of 0.0002321603128054741, chip offset of 237.5
        # 23
        # 24    # *** Identified satellite GpsSatelliteId(id=24) at doppler shift -5500, correlation magnitude of 5.057564040211816e-08 at 1281, time offset of 0.00037390029325513196, chip offset of 382.5
        # 30

        # Oct 31, 6:43pm
        # 3
        # 6
        # 11?
        # 12
        # 19
        # 24
        # 25
        # 28?
        # 29
        # 31
        # 32
        path=Path(__file__).parents[1] / "vendored_signals" / "halloween",
        format=InputFileType.GnuRadioRecording,
        # 2 * C/A PRN chip rate * 1k PRN repetitions per second
        sdr_sample_rate=2*1023*1000,
    ),
    InputFileInfo(
        path=Path("/Volumes/Seagate Backup/GPS/GNURadio Recordings/nov_3_time_18_48_roof"),
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
        data = np.fromfile(
            input_info.path.as_posix(),
            dtype=np.float32,
            # We have interleaved IQ samples, so the actual number of bytes to read will be the sample count * 2
            count=sample_count * 2,
        )
        # Recombine the inline IQ samples into complex numbers
        data = (data[0::2] + (1j * data[1::2]))
    else:
        raise ValueError(f'Unrecognized format')

    return data