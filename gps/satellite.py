import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from gps.config import PRN_CORRELATION_CYCLE_COUNT
from gps.gps_ca_prn_codes import GpsSatelliteId, GpsReplicaPrnSignal


ALL_SATELLITE_IDS = [GpsSatelliteId(i + 1) for i in range(32)]


@dataclass
class GpsSatellite:
    satellite_id: GpsSatelliteId
    prn_code: GpsReplicaPrnSignal

    def __hash__(self) -> int:
        return hash(self.satellite_id)

    @lru_cache(maxsize=PRN_CORRELATION_CYCLE_COUNT)
    def fft_of_prn_of_length(self, vector_size: int) -> np.ndarray:
        print(f'Calculating {self.satellite_id} PRN FFT of length {vector_size}...')
        needed_repetitions_to_match_vector_size = math.ceil(vector_size / len(self.prn_as_complex))
        prn_of_correct_length = np.tile(self.prn_as_complex, needed_repetitions_to_match_vector_size)[:vector_size]
        return np.fft.fft(prn_of_correct_length)

    @property
    @lru_cache
    def prn_as_complex(self) -> list[complex]:
        # Repeat each chip data point twice
        # This is because we'll be sampling at Nyquist frequency (2 * signal frequency, which is 1023 data points)
        prn_with_repeated_data_points = np.repeat(self.prn_code.inner, 2)
        # Adjust domain from [0 - 1] to [-1, 1] to match the IQ samples we'll receive
        prn_with_adjusted_domain = np.array([-1 if chip == 0 else 1 for chip in prn_with_repeated_data_points])
        # Convert to complex with a zero imaginary part
        prn_as_complex = [x+0j for x in prn_with_adjusted_domain]
        return prn_as_complex
