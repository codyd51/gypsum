import numpy as np
import matplotlib.pyplot as plt
from rtlsdr import RtlSdr
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

from scipy.signal import max_len_seq, resample

from gps_project_name.gps.gps_ca_prn_codes import generate_replica_prn_signals
from gps_project_name.gps.radio_input import INPUT_SOURCES, get_samples_from_radio_input_source

# PT: The SDR must be set to this center frequency
_GPS_L1_FREQUENCY = 1575.42e6


def main():
    input_source = INPUT_SOURCES[0]
    sample_rate = input_source.sdr_sample_rate
    data = get_samples_from_radio_input_source(input_source)
    # Generate PRN signals for each satellite
    satellites_to_replica_prn_signals = generate_replica_prn_signals(int(sample_rate))
    print(satellites_to_replica_prn_signals)

    return


if __name__ == '__main__':
    main()
