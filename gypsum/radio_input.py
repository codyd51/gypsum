import datetime
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Type

import dateutil.parser
import numpy as np

from gypsum.constants import PRN_CHIP_COUNT
from gypsum.constants import PRN_REPETITIONS_PER_SECOND
from gypsum.units import SampleCount
from gypsum.units import SampleRateHz


class InputFileType(Enum):
    Raw = auto()
    Wav = auto()
    GnuRadioRecording = auto()


@dataclass
class InputFileInfo:
    path: Path
    format: InputFileType
    sdr_sample_rate: SampleRateHz
    utc_start_time: datetime.datetime
    sample_component_data_type: Type[np.number]

    @classmethod
    def gnu_radio_recording(
        cls,
        path: Path,
        sample_rate: SampleRateHz,
        utc_start_time: datetime.datetime,
    ):
        return cls(
            path=path,
            format=InputFileType.GnuRadioRecording,
            sdr_sample_rate=sample_rate,
            # GNU Radio recordings are always stored as a pair of 32-bit interleaved floats per IQ sample.
            sample_component_data_type=np.float32,
            utc_start_time=utc_start_time,
        )

    @classmethod
    def gnu_radio_recording_2x(
        cls,
        path: Path,
        utc_start_time: datetime.datetime | None = None,
    ):
        """GNU Radio recording at exactly 2.046MHz.
        Double the PRN chipping rate: exactly Nyquist frequency, and the highest integer multiple of PRN chipping
        rate that the RTL-SDR can pull off.
        """
        return cls.gnu_radio_recording(
            path,
            sample_rate=PRN_CHIP_COUNT * PRN_REPETITIONS_PER_SECOND * 2,
            # TODO(PT): Implement based on the filesystem metadata?
            utc_start_time=datetime.datetime.utcfromtimestamp(0),
        )

    @classmethod
    def gnu_radio_recording_8x(
        cls,
        path: Path,
        utc_start_time: datetime.datetime | None = None,
    ):
        """GNU Radio recording at exactly 8.184MHz.
        8x the PRN chipping rate. Achievable by the HackRF One.
        """
        return cls.gnu_radio_recording(
            path,
            sample_rate=PRN_CHIP_COUNT * PRN_REPETITIONS_PER_SECOND * 8,
            # TODO(PT): Implement based on the filesystem metadata?
            utc_start_time=datetime.datetime.utcfromtimestamp(0),
        )

    @classmethod
    def gnu_radio_recording_16x(
        cls,
        path: Path,
        utc_start_time: datetime.datetime | None = None,
    ):
        """GNU Radio recording at exactly 16.386MHz.
        16x the PRN chipping rate. Achievable by the HackRF One.
        """
        return cls.gnu_radio_recording(
            path,
            sample_rate=PRN_CHIP_COUNT * PRN_REPETITIONS_PER_SECOND * 16,
            # TODO(PT): Implement based on the filesystem metadata?
            utc_start_time=datetime.datetime.utcfromtimestamp(0),
        )


_VENDORED_SIGNALS_ROOT = Path(__file__).parents[1] / "vendored_signals"


# TODO(PT): In the future, this can be extended to provide an input representing a live radio
# This can also be extended to include live recordings from my radio
# TODO(PT): Rename the associated symbols to 'input sources'
INPUT_SOURCES = [
    InputFileInfo(
        path=_VENDORED_SIGNALS_ROOT / "2013_04_04_GNSS_SIGNAL_at_CTTC_SPAIN.dat",
        format=InputFileType.Raw,
        sdr_sample_rate=int(4e6),
        utc_start_time=datetime.datetime.utcnow(),
        sample_component_data_type=np.float32,
    ),
    InputFileInfo(
        path=_VENDORED_SIGNALS_ROOT / "baseband_1575420000Hz_16-12-07_07-10-2023.wav",
        format=InputFileType.Wav,
        sdr_sample_rate=int(2.4e6),
        utc_start_time=datetime.datetime.utcnow(),
        sample_component_data_type=np.float32,
    ),
    InputFileInfo(
        path=_VENDORED_SIGNALS_ROOT / "vendored_signals" / "phillip.wav",
        format=InputFileType.Wav,
        sdr_sample_rate=int(2.4e6),
        utc_start_time=datetime.datetime.utcnow(),
        sample_component_data_type=np.float32,
    ),
    InputFileInfo(
        path=(
            _VENDORED_SIGNALS_ROOT
            / "baseband_1575136000Hz_22-52-15_10-10-2023_Int16_BiasT_TunerAGC_IQCorrection_100PPMCorrection.wav"
        ),
        format=InputFileType.Wav,
        sdr_sample_rate=int(2.4e6),
        utc_start_time=datetime.datetime.utcnow(),
        sample_component_data_type=np.float32,
    ),
    InputFileInfo(
        path=(
            _VENDORED_SIGNALS_ROOT
            / "baseband_1575420000Hz_23-21-28_10-10-2023_2048mhz_sample_rate.wav"
        ),
        format=InputFileType.Wav,
        sdr_sample_rate=int(2.048e6),
        utc_start_time=datetime.datetime.utcnow(),
        sample_component_data_type=np.float32,
    ),
    InputFileInfo.gnu_radio_recording_2x(_VENDORED_SIGNALS_ROOT / "output_at_seven_wives"),
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
    InputFileInfo.gnu_radio_recording_2x(_VENDORED_SIGNALS_ROOT / "halloween"),
    InputFileInfo.gnu_radio_recording_2x(
        _VENDORED_SIGNALS_ROOT / "nov_3_time_18_48_roof",
        utc_start_time=dateutil.parser.parse("2023-11-03T18:48:00+00:00"),
    ),
    # 10 @ 1224Hz
    # 12 @ 4569Hz
    # 15 @ -1159Hz
    # 19 @ 2376Hz
    # 24 @ 1198Hz
    # 25 @ 6305Hz
    # 32 @ 5740Hz
    InputFileInfo.gnu_radio_recording_2x(
        _VENDORED_SIGNALS_ROOT / "nov_30_15_00_30",
        utc_start_time=datetime.datetime.fromtimestamp(1701356430.648076714),
    ),
    # Visible sats appear to be:
    # 10 @ 1087Hz?
    # 12 @ 4478Hz
    # 15 @ -1101Hz
    # 19 @ 2271Hz
    # 24 @ 1134Hz
    # 25 @ 6261Hz
    # 32 @ 5667Hz
    InputFileInfo.gnu_radio_recording_2x(
        _VENDORED_SIGNALS_ROOT / "nov_30_15_04_10",
        utc_start_time=datetime.datetime.fromtimestamp(1701356651.195427381),
    ),
    # Visible sats appear to be:
    # 10 @ 287Hz
    # 12 @ 3386Hz
    # 25 @ 5613Hz
    # 28 @ 6242Hz
    # 32 @ 4943Hz
    InputFileInfo.gnu_radio_recording_2x(
        _VENDORED_SIGNALS_ROOT / "nov_30_15_34_10",
        utc_start_time=datetime.datetime.fromtimestamp(1701358453.379840235),
    ),
    # Visible sats appear to be:
    # 29 @ 973Hz, 97 -- maybe too low? Appears to be lost in short order
    # 7 @ 2248Hz, 121
    # 11 @ -37Hz, 159
    # 20 @ 2959Hz, 181
    # 30 @ 4387Hz, 210
    # 5 @ 5313Hz, 224
    # 13 @ 5943Hz, 294
    InputFileInfo.gnu_radio_recording_2x(
        _VENDORED_SIGNALS_ROOT / "dec_01_09_12_30",
        utc_start_time=datetime.datetime.fromtimestamp(1701421949.998035396),
    ),
    InputFileInfo.gnu_radio_recording(
        path=Path("/Users/philliptennen/interject-ios/test_recv"),
        sample_rate=6 * 1000 * 1000,
        utc_start_time=datetime.datetime.utcfromtimestamp(0),
    ),
    InputFileInfo.gnu_radio_recording_8x(Path("/Users/philliptennen/interject-ios/test_recv2")),
    InputFileInfo.gnu_radio_recording_8x(Path("/Users/philliptennen/interject-ios/test_recv3")),
    InputFileInfo.gnu_radio_recording_2x(
        path=Path("/Users/philliptennen/interject-ios/test_recv5"),
    ),
    InputFileInfo.gnu_radio_recording_2x(path=_VENDORED_SIGNALS_ROOT / "testing2"),
    InputFileInfo.gnu_radio_recording_2x(path=_VENDORED_SIGNALS_ROOT / "testing3"),
    InputFileInfo.gnu_radio_recording_2x(path=_VENDORED_SIGNALS_ROOT / "testing4"),
    InputFileInfo.gnu_radio_recording_16x(path=_VENDORED_SIGNALS_ROOT / "test_15_dec_10am"),
    InputFileInfo.gnu_radio_recording_2x(path=_VENDORED_SIGNALS_ROOT / "test_15_dec_12pm_3"),
    InputFileInfo.gnu_radio_recording_2x(path=_VENDORED_SIGNALS_ROOT / "test_15_dec_12pm_3"),
    InputFileInfo.gnu_radio_recording_2x(path=Path("/Users/philliptennen/Documents/GPS/rx_tools/test_rtlsdr")),
    InputFileInfo.gnu_radio_recording_2x(path=_VENDORED_SIGNALS_ROOT / "test_rtlsdr"),
    InputFileInfo.gnu_radio_recording_8x(path=_VENDORED_SIGNALS_ROOT / "test_hackrf"),
    InputFileInfo.gnu_radio_recording_8x(path=_VENDORED_SIGNALS_ROOT / "test_hackrf3"),
    InputFileInfo.gnu_radio_recording_2x(path=_VENDORED_SIGNALS_ROOT / "test_hackrf5"),
    InputFileInfo.gnu_radio_recording_2x(path=_VENDORED_SIGNALS_ROOT / "test_rtl6"),
    InputFileInfo.gnu_radio_recording_16x(path=_VENDORED_SIGNALS_ROOT / "test_hack6"),
    InputFileInfo.gnu_radio_recording_16x(path=_VENDORED_SIGNALS_ROOT / "test_rtl_new"),
    InputFileInfo.gnu_radio_recording_16x(path=_VENDORED_SIGNALS_ROOT / "test_rtl_new"),
    InputFileInfo.gnu_radio_recording_2x(path=_VENDORED_SIGNALS_ROOT / "test_rtl_new2"),
    InputFileInfo.gnu_radio_recording_16x(path=_VENDORED_SIGNALS_ROOT / "test_hackrf_new"),
    InputFileInfo.gnu_radio_recording_16x(path=_VENDORED_SIGNALS_ROOT / "test_hackrf_new2"),
    # [(5, 3108.3248994290457),
    #  (25, 3139.186118860456),
    #  (3, 3147.3998410937224),
    #  (19, 3148.639646966868),
    #  (30, 3200.440176057415),
    #  (4, 3215.5997460474196),
    #  (12, 3242.329519242233),
    #  (18, 3265.5346650495408),
    #  (7, 3285.467698761179),
    #  (16, 3304.0207198992616),
    #  (26, 3326.7762836126276),
    #  (11, 3334.3861058136044),
    #  (10, 3336.9231844287797),
    #  (21, 3345.7044135098677),
    #  (1, 3346.6130532707557),
    #  (29, 3347.276038506539),
    #  (24, 3372.2056733025206),
    #  (8, 3374.0763808857655),
    #  (9, 3394.195514670206),
    #  (17, 3401.6134254185345),
    #  (31, 3403.7169369737403),
    #  (32, 3430.623800936277),
    #  (2, 3432.2167820326813),
    #  (14, 3444.2854768928887),
    #  (6, 3505.200645947617),
    #  (28, 3541.9792597614155),
    #  (27, 3634.844212222258),
    #  (20, 3747.7139688671177),
    #  (13, 7218.957396982364),
    #  (23, 8207.270579526188),
    #  (22, 9043.37046816484),
    #  (15, 12433.944236015503)]
    InputFileInfo.gnu_radio_recording_16x(path=_VENDORED_SIGNALS_ROOT / "test_hackrf_new3"),
    InputFileInfo.gnu_radio_recording_2x(path=_VENDORED_SIGNALS_ROOT / "test_rtl_night", utc_start_time=datetime.datetime.fromtimestamp(1702851337.874580792)),
]


def get_input_source_by_file_name(name: str) -> InputFileInfo:
    input_files_matching_name = [x for x in INPUT_SOURCES if x.path.name == name]
    if len(input_files_matching_name) == 0:
        raise FileNotFoundError(f'No input file named "{name}" found.')

    if len(input_files_matching_name) > 1:
        raise RuntimeError(f'Found more than one file named "{name}".')

    return input_files_matching_name[0]
