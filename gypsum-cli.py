import logging

import matplotlib.pyplot as plt
import numpy as np

from gypsum.antenna_sample_provider import AntennaSampleProviderBackedByFile
from gypsum.radio_input import INPUT_SOURCES, get_input_source_by_file_name
from gypsum.receiver import GpsReceiver

_AntennaSamplesSpanningOneMs = np.ndarray
_AntennaSamplesSpanningAcquisitionIntegrationPeriodMs = np.ndarray


_logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO)

    input_source = INPUT_SOURCES[7]
    #input_source = get_input_source_by_file_name("test_rtl_night")
    antenna_samples_provider = AntennaSampleProviderBackedByFile(input_source)
    _logger.info(f"Set up antenna sample stream backed by file: {input_source.path.as_posix()}")

    receiver = GpsReceiver(antenna_samples_provider)
    while True:
        receiver.step()

    # Lock decider: look at the average messiness of the IQ constellation over the last 3 seconds?


if __name__ == "__main__":
    main()
