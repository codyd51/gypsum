import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from gypsum.config import ACQUISITION_INTEGRATED_CORRELATION_STRENGTH_DETECTION_THRESHOLD
from gypsum.gps_ca_prn_codes import GpsSatelliteId
from gypsum.satellite import GpsSatellite
from gypsum.units import (
    CarrierWavePhaseInRadians,
    PrnCodePhaseInSamples,
)
from gypsum.utils import (
    AntennaSamplesSpanningAcquisitionIntegrationPeriodMs,
    CorrelationProfile,
    DopplerShiftHz,
    IntegrationType,
    PrnReplicaCodeSamplesSpanningOneMs,
    integrate_correlation_with_doppler_shifted_prn,
)

_logger = logging.getLogger(__name__)


@dataclass
class BestNonCoherentCorrelationProfile:
    doppler_shift: DopplerShiftHz
    non_coherent_correlation_profile: CorrelationProfile
    # Just convenience accessors that can be derived from the correlation profile
    sample_offset_of_correlation_peak: int
    correlation_strength: float


@dataclass
class SatelliteAcquisitionAttemptResult:
    satellite_id: GpsSatelliteId
    doppler_shift: DopplerShiftHz
    carrier_wave_phase_shift: CarrierWavePhaseInRadians
    prn_phase_shift: PrnCodePhaseInSamples
    correlation_strength: float


class GpsSatelliteDetector:
    def __init__(self, satellites_by_id: dict[GpsSatelliteId, GpsSatellite]) -> None:
        self.satellites_by_id = satellites_by_id
        self._cached_correlation_profiles: dict[Any, CorrelationProfile] = {}

    def detect_satellites_in_antenna_data(
        self,
        satellites_to_search_for: list[GpsSatelliteId],
        antenna_data: AntennaSamplesSpanningAcquisitionIntegrationPeriodMs,
    ) -> list[SatelliteAcquisitionAttemptResult]:
        detected_satellites = []
        for satellite_id in satellites_to_search_for:
            result = self._attempt_acquisition_for_satellite_id(
                satellite_id,
                antenna_data,
            )
            if result.correlation_strength > ACQUISITION_INTEGRATED_CORRELATION_STRENGTH_DETECTION_THRESHOLD:
                _logger.info(f"Correlation strength above threshold, successfully detected satellite {satellite_id}!")
                detected_satellites.append(result)
        return detected_satellites

    def _attempt_acquisition_for_satellite_id(
        self,
        satellite_id: GpsSatelliteId,
        samples_for_integration_period: AntennaSamplesSpanningAcquisitionIntegrationPeriodMs,
    ) -> SatelliteAcquisitionAttemptResult:
        _logger.info(f"Attempting acquisition of {satellite_id}...")
        best_non_coherent_correlation_profile_across_all_search_space = None
        center_doppler_shift_estimation = 0.0
        doppler_frequency_estimation_spread = 7000.0
        # This must be 10 as the search factor divides the spread by 10
        while doppler_frequency_estimation_spread >= 10:
            best_non_coherent_correlation_profile_in_this_search_space = self.get_best_doppler_shift_estimation(
                center_doppler_shift_estimation,
                doppler_frequency_estimation_spread,
                samples_for_integration_period,
                satellite_id,
            )
            doppler_frequency_estimation_spread /= 2
            center_doppler_shift_estimation = best_non_coherent_correlation_profile_in_this_search_space.doppler_shift

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
                # center_doppler_shift_estimation = best_non_coherent_correlation_profile_across_all_search_space.doppler_shift
                _logger.info(
                    f"Found a better candidate Doppler for SV({satellite_id}): "
                    f"(Found in [{center_doppler_shift_estimation - doppler_frequency_estimation_spread:.2f} | "
                    f"{center_doppler_shift_estimation:.2f} | {center_doppler_shift_estimation + doppler_frequency_estimation_spread:.2f}], "
                    f"Strength: "
                    f"{best_non_coherent_correlation_profile_across_all_search_space.correlation_strength:.2f}"
                )

        # PT: For typing
        if not best_non_coherent_correlation_profile_across_all_search_space:
            raise RuntimeError(f'Should never happen: Expected at least one correlation profile')

        _logger.info(
            f"Best correlation for SV({satellite_id}) at "
            f"Doppler {center_doppler_shift_estimation:.2f} "
            f"corr {best_non_coherent_correlation_profile_across_all_search_space.correlation_strength:.2f}"
        )

        best_doppler_shift = best_non_coherent_correlation_profile_across_all_search_space.doppler_shift
        # Now, compute the coherent correlation so that we can determine (an estimate) of the phase of the carrier wave
        coherent_correlation_profile = self.get_integrated_correlation_with_doppler_shifted_prn(
            IntegrationType.Coherent,
            samples_for_integration_period,
            best_doppler_shift,
            self.satellites_by_id[satellite_id].prn_as_complex, # type: ignore
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
        antenna_data: AntennaSamplesSpanningAcquisitionIntegrationPeriodMs,
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
                self.satellites_by_id[satellite_id].prn_as_complex, # type: ignore
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
        antenna_data: AntennaSamplesSpanningAcquisitionIntegrationPeriodMs,
        doppler_shift: DopplerShiftHz,
        prn_as_complex: PrnReplicaCodeSamplesSpanningOneMs,
    ) -> CorrelationProfile:
        # Ref: https://stackoverflow.com/questions/16589791/most-efficient-property-to-hash-for-numpy-array
        # antenna_data.sum() will have a higher chance of collisions than .tostring(), but it's faster,
        # and I'm willing to take the chance.
        key = hash((integration_type, hash(antenna_data.sum()), doppler_shift, hash(prn_as_complex.tostring())))    # type: ignore
        if False and key in self._cached_correlation_profiles:
            _logger.debug(f"Did hit cache for PRN correlation result")
            cached_correlation_profile = self._cached_correlation_profiles[key]
            return cached_correlation_profile

        _logger.debug(f"Did not hit cache for PRN correlation result")
        correlation_profile = integrate_correlation_with_doppler_shifted_prn(
            integration_type,
            antenna_data,
            doppler_shift,
            prn_as_complex,
        )
        self._cached_correlation_profiles[key] = correlation_profile
        return correlation_profile
