import collections
import logging
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from enum import auto
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft

from gypsum.constants import SAMPLES_PER_SECOND
from gypsum.gps_ca_prn_codes import generate_replica_prn_signals, GpsSatelliteId
from gypsum.constants import SAMPLES_PER_PRN_TRANSMISSION, MINIMUM_TRACKED_SATELLITES_FOR_POSITION_FIX
from gypsum.satellite import GpsSatellite, ALL_SATELLITE_IDS
from gypsum.antenna_sample_provider import AntennaSampleProvider
from gypsum.config import ACQUISITION_INTEGRATION_PERIOD_MS
from gypsum.utils import chunks


_logger = logging.getLogger(__name__)


_AntennaSamplesSpanningOneMs = np.ndarray
_AntennaSamplesSpanningAcquisitionIntegrationPeriodMs = np.ndarray
_PrnReplicaCodeSamplesSpanningOneMs = np.ndarray

_CorrelationProfile = np.ndarray
_CorrelationStrength = float

_DopplerShiftHz = float
_CarrierWavePhaseInRadians = float
_PrnCodePhaseInSamples = int


class IntegrationType(Enum):
    Coherent = auto()
    NonCoherent = auto()


@dataclass
class BestNonCoherentCorrelationProfile:
    doppler_shift: _DopplerShiftHz
    non_coherent_correlation_profile: _CorrelationProfile
    # Just convenience accessors that can be derived from the correlation profile
    sample_offset_of_correlation_peak: int
    correlation_strength: float


@dataclass
class SatelliteAcquisitionAttemptResult:
    satellite_id: GpsSatelliteId
    doppler_shift: _DopplerShiftHz
    carrier_wave_phase_shift: _CarrierWavePhaseInRadians
    prn_phase_shift: _PrnCodePhaseInSamples
    correlation_strength: float


def frequency_domain_correlation(
    antenna_samples: _AntennaSamplesSpanningOneMs,
    prn_replica: _PrnReplicaCodeSamplesSpanningOneMs
) -> _CorrelationProfile:
    # Perform correlation in the frequency domain.
    # This is much more efficient than attempting to perform correlation in the time domain, as we don't need to try
    # every possible phase shift of the PRN to identify the correlation peak.
    antenna_samples_fft = np.fft.fft(antenna_samples)
    prn_replica_fft = np.fft.fft(prn_replica)
    # Multiply by the complex conjugate of the PRN replica.
    # This aligns the phases of the antenna data and replica, and performs the cross-correlation.
    correlation_in_frequency_domain = antenna_samples_fft * np.conj(prn_replica_fft)
    # Convert the correlation result back to the time domain.
    # Each value gives the correlation of the antenna data with the PRN at different phase offsets.
    # Therefore, the offset of the peak will give the phase shift of the PRN that gives maximum correlation.
    return np.fft.ifft(correlation_in_frequency_domain)


def integrate_correlation_with_doppler_shifted_prn(
    integration_type: IntegrationType,
    antenna_data: _AntennaSamplesSpanningAcquisitionIntegrationPeriodMs,
    doppler_shift: _DopplerShiftHz,
    prn_as_complex: _PrnReplicaCodeSamplesSpanningOneMs,
) -> _CorrelationProfile:
    sample_count = len(antenna_data)
    integration_time_domain = np.arange(sample_count) / SAMPLES_PER_SECOND
    doppler_shift_carrier = np.exp(-1j * 2 * np.pi * doppler_shift * integration_time_domain)
    doppler_shifted_antenna_data_snippet = antenna_data * doppler_shift_carrier

    correlation_data_type = {
        IntegrationType.Coherent: complex,
        IntegrationType.NonCoherent: np.float64,
    }[integration_type]
    coherent_integration_result = np.zeros(SAMPLES_PER_PRN_TRANSMISSION, dtype=correlation_data_type)
    for i, chunk_that_may_contain_one_prn in enumerate(
        chunks(doppler_shifted_antenna_data_snippet, SAMPLES_PER_PRN_TRANSMISSION)
    ):
        correlation_result = frequency_domain_correlation(chunk_that_may_contain_one_prn, prn_as_complex)

        if integration_type == IntegrationType.Coherent:
            coherent_integration_result += correlation_result
        elif integration_type == IntegrationType.NonCoherent:
            coherent_integration_result += np.abs(correlation_result)
        else:
            raise ValueError("Unexpected integration type")

    return coherent_integration_result


class GpsSatelliteDetector:
    def __init__(self, satellites_by_id: dict[GpsSatelliteId, GpsSatellite]) -> None:
        self.satellites_by_id = satellites_by_id
        self._cached_correlation_profiles: dict[Any, _CorrelationProfile] = {}

    def detect_satellites_in_antenna_data(
        self,
        satellites_to_search_for: list[GpsSatelliteId],
        antenna_data: _AntennaSamplesSpanningAcquisitionIntegrationPeriodMs,
    ) -> None:
        for satellite_id in satellites_to_search_for:
            result = self._attempt_acquisition_for_satellite_id(
                satellite_id,
                antenna_data,
            )
            # TODO(PT): How to decide whether the correlation strength is strong enough?
            if result.correlation_strength > 70.0:
                _logger.info(f"Correlation strength above threshold, successfully detected satellite {satellite_id}!")

    def _attempt_acquisition_for_satellite_id(
        self,
        satellite_id: GpsSatelliteId,
        samples_for_integration_period: _AntennaSamplesSpanningAcquisitionIntegrationPeriodMs,
    ) -> SatelliteAcquisitionAttemptResult:
        _logger.info(f"Attempting acquisition of {satellite_id}...")
        best_non_coherent_correlation_profile_across_all_search_space = None
        center_doppler_shift_estimation = 0
        doppler_frequency_estimation_spread = 7000
        # This must be 10 as the search factor divides the spread by 10
        while doppler_frequency_estimation_spread >= 10:
            best_non_coherent_correlation_profile_in_this_search_space = self.get_best_doppler_shift_estimation(
                center_doppler_shift_estimation,
                doppler_frequency_estimation_spread,
                samples_for_integration_period,
                satellite_id,
            )
            doppler_frequency_estimation_spread /= 2

            if (
                # Base case
                not best_non_coherent_correlation_profile_across_all_search_space
                # Found a better candidate
                or best_non_coherent_correlation_profile_in_this_search_space.correlation_strength
                > best_non_coherent_correlation_profile_across_all_search_space.correlation_strength
            ):
                best_non_coherent_correlation_profile_across_all_search_space = (
                    best_non_coherent_correlation_profile_in_this_search_space
                )
                doppler_estimation = best_non_coherent_correlation_profile_across_all_search_space.doppler_shift
                _logger.info(
                    f"Found a better candidate Doppler for SV({satellite_id}): "
                    f"(Found in [{doppler_estimation - doppler_frequency_estimation_spread:.2f} | "
                    f"{doppler_estimation:.2f} | {doppler_estimation + doppler_frequency_estimation_spread:.2f}], "
                    f"Strength: "
                    f"{best_non_coherent_correlation_profile_across_all_search_space.correlation_strength:.2f}"
                )

        _logger.info(
            f"Best correlation for SV({satellite_id}) at "
            f"Doppler {center_doppler_shift_estimation:.2f} "
            f"corr {best_non_coherent_correlation_profile_across_all_search_space.correlation_strength:.2f}"
        )

        best_doppler_shift = best_non_coherent_correlation_profile_across_all_search_space.doppler_shift
        plt.plot(best_non_coherent_correlation_profile_across_all_search_space.non_coherent_correlation_profile)
        plt.title(f"SV {satellite_id.id} doppler {best_doppler_shift}")
        plt.show(block=True)
        # Now, compute the coherent correlation so that we can determine (an estimate) of the phase of the carrier wave
        coherent_correlation_profile = self.get_integrated_correlation_with_doppler_shifted_prn(
            IntegrationType.Coherent,
            samples_for_integration_period,
            best_doppler_shift,
            self.satellites_by_id[satellite_id].prn_as_complex,
        )

        # Rely on the correlation peak index that comes from non-coherent integration, since it'll be stronger and
        # therefore has less chance of being overridden by noise. Coherent integration may have selected a noise
        # peak.
        sample_offset_of_correlation_peak = (
            best_non_coherent_correlation_profile_across_all_search_space.sample_offset_of_correlation_peak
        )
        carrier_wave_phase_shift = np.angle(coherent_correlation_profile[sample_offset_of_correlation_peak])
        # The sample offset where the best correlation occurs gives us (an estimate) of the phase shift of the PRN
        prn_phase_shift = sample_offset_of_correlation_peak
        correlation_strength = best_non_coherent_correlation_profile_across_all_search_space.correlation_strength
        _logger.info(f"Acquisition attempt result for SV({satellite_id}):")
        _logger.info(f"\tCorrelation strength {correlation_strength:.2f}")
        _logger.info(f"\tDoppler {best_doppler_shift:.2f}")
        _logger.info(f"\tCarrier phase {carrier_wave_phase_shift}")
        _logger.info(f"\tPRN phase {prn_phase_shift:.2f}")
        return SatelliteAcquisitionAttemptResult(
            satellite_id=satellite_id,
            doppler_shift=best_doppler_shift,
            carrier_wave_phase_shift=carrier_wave_phase_shift,
            prn_phase_shift=prn_phase_shift,
            correlation_strength=correlation_strength,
        )

    def get_best_doppler_shift_estimation(
        self,
        center_doppler_shift: float,
        doppler_shift_spread: float,
        antenna_data: _AntennaSamplesSpanningAcquisitionIntegrationPeriodMs,
        satellite_id: GpsSatelliteId,
    ) -> BestNonCoherentCorrelationProfile:
        doppler_shift_to_correlation_profile = {}
        for doppler_shift in range(
            int(center_doppler_shift - doppler_shift_spread),
            int(center_doppler_shift + doppler_shift_spread),
            int(doppler_shift_spread / 10),
        ):
            correlation_profile = self.get_integrated_correlation_with_doppler_shifted_prn(
                # Always use non-coherent integration when searching for the best Doppler peaks.
                # This will give us the strongest SNR possible to detect peaks.
                IntegrationType.NonCoherent,
                antenna_data,
                doppler_shift,
                self.satellites_by_id[satellite_id].prn_as_complex,
            )
            doppler_shift_to_correlation_profile[doppler_shift] = correlation_profile

        # Find the best correlation result
        best_doppler_shift = max(
            doppler_shift_to_correlation_profile, key=lambda key: np.max(doppler_shift_to_correlation_profile[key])
        )
        best_correlation_profile = doppler_shift_to_correlation_profile[best_doppler_shift]
        sample_offset_of_correlation_peak = np.argmax(best_correlation_profile)
        correlation_strength = best_correlation_profile[sample_offset_of_correlation_peak]
        return BestNonCoherentCorrelationProfile(
            doppler_shift=best_doppler_shift,
            non_coherent_correlation_profile=best_correlation_profile,
            sample_offset_of_correlation_peak=int(sample_offset_of_correlation_peak),
            correlation_strength=correlation_strength,
        )

    def get_integrated_correlation_with_doppler_shifted_prn(
        self,
        integration_type: IntegrationType,
        antenna_data: _AntennaSamplesSpanningAcquisitionIntegrationPeriodMs,
        doppler_shift: _DopplerShiftHz,
        prn_as_complex: _PrnReplicaCodeSamplesSpanningOneMs,
    ) -> _CorrelationProfile:
        # Ref: https://stackoverflow.com/questions/16589791/most-efficient-property-to-hash-for-numpy-array
        # antenna_data.sum() will have a higher chance of collisions than .tostring(), but it's faster,
        # and I'm willing to take the chance.
        key = hash((integration_type, hash(antenna_data.sum()), doppler_shift, hash(prn_as_complex.tostring())))
        if key in self._cached_correlation_profiles:
            _logger.debug(f'Did hit cache for PRN correlation result')
            cached_correlation_profile = self._cached_correlation_profiles[key]
            return cached_correlation_profile

        _logger.debug(f'Did not hit cache for PRN correlation result')
        correlation_profile = integrate_correlation_with_doppler_shifted_prn(
            integration_type,
            antenna_data,
            doppler_shift,
            prn_as_complex,
        )
        self._cached_correlation_profiles[key] = correlation_profile
        return correlation_profile


class GpsReceiver:
    def __init__(self, antenna_samples_provider: AntennaSampleProvider) -> None:
        self.antenna_samples_provider = antenna_samples_provider

        # Generate the replica signals that we'll use to correlate against the received antenna signals upfront
        satellites_to_replica_prn_signals = generate_replica_prn_signals()
        self.satellites_by_id = {
            satellite_id: GpsSatellite(satellite_id=satellite_id, prn_code=code)
            for satellite_id, code in satellites_to_replica_prn_signals.items()
        }

        self.acquired_satellites = []
        self.satellite_ids_eligible_for_acquisition = deepcopy(ALL_SATELLITE_IDS)

        self.satellite_detector = GpsSatelliteDetector(self.satellites_by_id)
        # Used during acquisition to integrate correlation over a longer period than a millisecond.
        self.rolling_samples_buffer = collections.deque(maxlen=ACQUISITION_INTEGRATION_PERIOD_MS)

    def step(self):
        """Run one 'iteration' of the GPS receiver. This consumes one millisecond of antenna data."""
        samples: _AntennaSamplesSpanningOneMs = self.antenna_samples_provider.get_samples(SAMPLES_PER_PRN_TRANSMISSION)
        # Firstly, record this sample in our rolling buffer
        self.rolling_samples_buffer.append(samples)

        # If we need to perform acquisition, do so now
        if len(self.acquired_satellites) < MINIMUM_TRACKED_SATELLITES_FOR_POSITION_FIX:
            _logger.info(
                f"Will perform acquisition search because we're only "
                f"tracking {len(self.acquired_satellites)} satellites"
            )
            self._perform_acquisition()

    def _perform_acquisition(self) -> None:
        # To improve signal-to-noise ratio during acquisition, we integrate antenna data over 20ms.
        # Therefore, we keep a rolling buffer of the last few samples.
        # If this buffer isn't primed yet, we can't do any work yet.
        if len(self.rolling_samples_buffer) < ACQUISITION_INTEGRATION_PERIOD_MS:
            _logger.info(f"Skipping acquisition attempt because the history buffer isn't primed yet.")
            return

        _logger.info(
            f"Performing acquisition search over {len(self.satellite_ids_eligible_for_acquisition)} satellites."
        )

        samples_for_integration_period = np.concatenate(self.rolling_samples_buffer)
        self.satellite_detector.detect_satellites_in_antenna_data(
            self.satellite_ids_eligible_for_acquisition,
            samples_for_integration_period,
        )
