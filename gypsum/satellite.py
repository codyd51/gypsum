from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from gypsum.gps_ca_prn_codes import GpsReplicaPrnSignal, GpsSatelliteId

ALL_SATELLITE_IDS = [GpsSatelliteId(i + 1) for i in range(32)]


@dataclass
class GpsSatellite:
    satellite_id: GpsSatelliteId
    prn_code: GpsReplicaPrnSignal
    scale_factor: int

    def __hash__(self) -> int:
        return hash(self.satellite_id)

    @property
    @lru_cache
    def prn_as_complex(self) -> np.ndarray:
        # We're going to try to correlate our generated PRN with the samples coming in from our radio, so we need to
        # resample our generated PRN to match however many samples we expect the received PRN to occupy.
        # TODO(PT): Perhaps this should own the calculation for the scale factor, once we can resample fractionally?
        prn_with_repeated_data_points = np.repeat(self.prn_code.inner, self.scale_factor)
        # Adjust domain from [0 - 1] to [-1, 1] to match the IQ samples we'll receive
        prn_with_adjusted_domain = np.array([-1 if chip == 0 else 1 for chip in prn_with_repeated_data_points])
        # Convert to complex with a zero imaginary part
        prn_as_complex = prn_with_adjusted_domain.astype(complex)
        return prn_as_complex
