import datetime
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Type

import dateutil.parser
import numpy as np

from gypsum.units import SampleCount


class InputFileType(Enum):
    Raw = auto()
    Wav = auto()
    GnuRadioRecording = auto()


@dataclass
class InputFileInfo:
    path: Path
    format: InputFileType
    sdr_sample_rate: SampleCount
    utc_start_time: datetime.datetime
    sample_component_data_type: Type[np.number]


# TODO(PT): In the future, this can be extended to provide an input representing a live radio
# This can also be extended to include live recordings from my radio
# TODO(PT): Rename the associated symbols to 'input sources'
INPUT_SOURCES = [
    InputFileInfo(
        path=Path(__file__).parents[1] / "vendored_signals" / "2013_04_04_GNSS_SIGNAL_at_CTTC_SPAIN.dat",
        format=InputFileType.Raw,
        sdr_sample_rate=int(4e6),
        utc_start_time=datetime.datetime.utcnow(),
        sample_component_data_type=np.float32,
    ),
    InputFileInfo(
        path=Path(__file__).parents[1] / "vendored_signals" / "baseband_1575420000Hz_16-12-07_07-10-2023.wav",
        format=InputFileType.Wav,
        sdr_sample_rate=int(2.4e6),
        utc_start_time=datetime.datetime.utcnow(),
        sample_component_data_type=np.float32,
    ),
    InputFileInfo(
        path=Path(__file__).parents[1] / "vendored_signals" / "phillip.wav",
        format=InputFileType.Wav,
        sdr_sample_rate=int(2.4e6),
        utc_start_time=datetime.datetime.utcnow(),
        sample_component_data_type=np.float32,
    ),
    InputFileInfo(
        path=Path(__file__).parents[1]
        / "vendored_signals"
        / "baseband_1575136000Hz_22-52-15_10-10-2023_Int16_BiasT_TunerAGC_IQCorrection_100PPMCorrection.wav",
        format=InputFileType.Wav,
        sdr_sample_rate=int(2.4e6),
        utc_start_time=datetime.datetime.utcnow(),
        sample_component_data_type=np.float32,
    ),
    InputFileInfo(
        path=Path(__file__).parents[1]
        / "vendored_signals"
        / "baseband_1575420000Hz_23-21-28_10-10-2023_2048mhz_sample_rate.wav",
        format=InputFileType.Wav,
        sdr_sample_rate=int(2.048e6),
        utc_start_time=datetime.datetime.utcnow(),
        sample_component_data_type=np.float32,
    ),
    InputFileInfo(
        path=Path(__file__).parents[1] / "vendored_signals" / "output_at_seven_wives",
        format=InputFileType.GnuRadioRecording,
        # 2 * C/A PRN chip rate * 1k PRN repetitions per second
        sdr_sample_rate=2 * 1023 * 1000,
        utc_start_time=datetime.datetime.utcnow(),
        sample_component_data_type=np.float32,
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
        # path=Path(__file__).parents[1] / "vendored_signals" / "test_output_in_office_gnu_radio",
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
        sdr_sample_rate=2 * 1023 * 1000,
        utc_start_time=datetime.datetime.utcnow(),
        sample_component_data_type=np.float32,
    ),
    InputFileInfo(
        path=Path(__file__).parents[1] / "vendored_signals" / "nov_3_time_18_48_roof",
        format=InputFileType.GnuRadioRecording,
        # 2 * C/A PRN chip rate * 1k PRN repetitions per second
        sdr_sample_rate=2 * 1023 * 1000,
        utc_start_time=dateutil.parser.parse("2023-11-03T18:48:00+00:00"),
        sample_component_data_type=np.float32,
    ),
    InputFileInfo(
        path=Path(__file__).parents[1] / "vendored_signals" / "nov_30_15_00_30",
        format=InputFileType.GnuRadioRecording,
        # 2 * C/A PRN chip rate * 1k PRN repetitions per second
        sdr_sample_rate=2 * 1023 * 1000,
        utc_start_time=datetime.datetime.fromtimestamp(1701356430.648076714),
        # 10 @ 1224Hz
        # 12 @ 4569Hz
        # 15 @ -1159Hz
        # 19 @ 2376Hz
        # 24 @ 1198Hz
        # 25 @ 6305Hz
        # 32 @ 5740Hz
        sample_component_data_type=np.float32,
    ),
    InputFileInfo(
        path=Path(__file__).parents[1] / "vendored_signals" / "nov_30_15_04_10",
        format=InputFileType.GnuRadioRecording,
        # 2 * C/A PRN chip rate * 1k PRN repetitions per second
        sdr_sample_rate=2 * 1023 * 1000,
        utc_start_time=datetime.datetime.fromtimestamp(1701356651.195427381),
        sample_component_data_type=np.float32,
        # Visible sats appear to be:
        # 10 @ 1087Hz?
        # 12 @ 4478Hz
        # 15 @ -1101Hz
        # 19 @ 2271Hz
        # 24 @ 1134Hz
        # 25 @ 6261Hz
        # 32 @ 5667Hz
    ),
    InputFileInfo(
        path=Path(__file__).parents[1] / "vendored_signals" / "nov_30_15_34_10",
        format=InputFileType.GnuRadioRecording,
        # 2 * C/A PRN chip rate * 1k PRN repetitions per second
        sdr_sample_rate=2 * 1023 * 1000,
        utc_start_time=datetime.datetime.fromtimestamp(1701358453.379840235),
        sample_component_data_type=np.float32,
        # Visible sats appear to be:
        # 10 @ 287Hz
        # 12 @ 3386Hz
        # 25 @ 5613Hz
        # 28 @ 6242Hz
        # 32 @ 4943Hz
    ),
    InputFileInfo(
        path=Path(__file__).parents[1] / "vendored_signals" / "dec_01_09_12_30",
        format=InputFileType.GnuRadioRecording,
        sdr_sample_rate=2 * 1023 * 1000,
        utc_start_time=datetime.datetime.fromtimestamp(1701421949.998035396),
        sample_component_data_type=np.float32,
        # Visible sats appear to be:
        # 29 @ 973Hz, 97 -- maybe too low? Appears to be lost in short order
        # 7 @ 2248Hz, 121
        # 11 @ -37Hz, 159
        # 20 @ 2959Hz, 181
        # 30 @ 4387Hz, 210
        # 5 @ 5313Hz, 224
        # 13 @ 5943Hz, 294
    ),
]
