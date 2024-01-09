import logging
import argparse

import numpy as np

from gypsum.antenna_sample_provider import AntennaSampleProviderBackedByFile
from gypsum.gps_ca_prn_codes import GpsSatelliteId
from gypsum.radio_input import INPUT_SOURCES, get_input_source_by_file_name
from gypsum.receiver import GpsReceiver

_AntennaSamplesSpanningOneMs = np.ndarray
_AntennaSamplesSpanningAcquisitionIntegrationPeriodMs = np.ndarray


_logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--only_acquire_satellite_ids", nargs="*")
    parser.add_argument("--present_matplotlib_sat_tracker", action=argparse.BooleanOptionalAction)
    parser.add_argument("--present_web_ui", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    input_source = INPUT_SOURCES[7]
    #input_source = get_input_source_by_file_name("nov_30_15_34_10")
    antenna_samples_provider = AntennaSampleProviderBackedByFile(input_source)
    _logger.info(f"Set up antenna sample stream backed by file: {input_source.path.as_posix()}")

    if args.only_acquire_satellite_ids is not None:
        only_acquire_satellite_ids = [GpsSatelliteId(id=int(x)) for x in args.only_acquire_satellite_ids]
    else:
        only_acquire_satellite_ids = None
    # argparse will give us None if the user didn't specify the flag
    present_matplotlib_sat_tracker = args.present_matplotlib_sat_tracker is True
    present_web_ui = args.present_web_ui is True
    receiver = GpsReceiver(
        antenna_samples_provider,
        only_acquire_satellite_ids=only_acquire_satellite_ids,
        present_matplotlib_satellite_tracker=present_matplotlib_sat_tracker,
        present_web_ui=present_web_ui,
    )
    while True:
        receiver.step()

    # Lock decider: look at the average messiness of the IQ constellation over the last 3 seconds?


if __name__ == "__main__":
    main()
