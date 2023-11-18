import logging

import matplotlib.pyplot as plt
import numpy as np

from gypsum.antenna_sample_provider import AntennaSampleProviderBackedByFile
from gypsum.radio_input import INPUT_SOURCES
from gypsum.receiver import GpsReceiver

_AntennaSamplesSpanningOneMs = np.ndarray
_AntennaSamplesSpanningAcquisitionIntegrationPeriodMs = np.ndarray


_logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO)

    input_source = INPUT_SOURCES[7]
    antenna_samples_provider = AntennaSampleProviderBackedByFile(input_source.path)
    _logger.info(f"Set up antenna sample stream backed by file: {input_source.path.as_posix()}")

    receiver = GpsReceiver(antenna_samples_provider)
    while True:
        receiver.step()


if __name__ == "__main__":
    main()
