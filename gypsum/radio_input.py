import datetime
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Type

import dateutil.parser
import numpy as np

from gypsum.constants import PRN_CHIP_COUNT
from gypsum.constants import PRN_REPETITIONS_PER_SECOND
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
    InputFileInfo.gnu_radio_recording_2x(
        _VENDORED_SIGNALS_ROOT / "nov_3_time_18_48_st_ives",
        utc_start_time=dateutil.parser.parse("2023-11-03T18:48:00+00:00"),
    ),
    InputFileInfo.gnu_radio_recording_2x(
        path=_VENDORED_SIGNALS_ROOT / "test_rtl_night",
        utc_start_time=datetime.datetime.fromtimestamp(1702851337.874580792),
    ),
    InputFileInfo.gnu_radio_recording_16x(path=_VENDORED_SIGNALS_ROOT / "test_hackrf_new3"),
]


def get_input_source_by_file_name(name: str) -> InputFileInfo:
    input_files_matching_name = [x for x in INPUT_SOURCES if x.path.name == name]
    if len(input_files_matching_name) == 0:
        raise FileNotFoundError(f'No input file named "{name}" found.')

    if len(input_files_matching_name) > 1:
        raise RuntimeError(f'Found more than one file named "{name}".')

    return input_files_matching_name[0]
