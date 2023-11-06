import collections
import logging
from copy import deepcopy

import math
from dataclasses import dataclass
from enum import Enum
from enum import auto
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft

from gypsum.config import ACQUISITION_INTEGRATED_CORRELATION_STRENGTH_DETECTION_THRESHOLD
from gypsum.constants import SAMPLES_PER_SECOND
from gypsum.gps_ca_prn_codes import generate_replica_prn_signals, GpsSatelliteId
from gypsum.constants import SAMPLES_PER_PRN_TRANSMISSION, MINIMUM_TRACKED_SATELLITES_FOR_POSITION_FIX
from gypsum.satellite import ALL_SATELLITE_IDS
from gypsum.satellite import GpsSatellite
from gypsum.antenna_sample_provider import AntennaSampleProvider
from gypsum.config import ACQUISITION_INTEGRATION_PERIOD_MS
from gypsum.utils import chunks
from gypsum.utils import does_list_contain_sublist

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
    correlation_data_type = {
        IntegrationType.Coherent: complex,
        IntegrationType.NonCoherent: np.float64,
    }[integration_type]
    integrated_correlation_result = np.zeros(SAMPLES_PER_PRN_TRANSMISSION, dtype=correlation_data_type)
    for i, chunk_that_may_contain_one_prn in enumerate(chunks(antenna_data, SAMPLES_PER_PRN_TRANSMISSION)):
        sample_index = i * SAMPLES_PER_PRN_TRANSMISSION
        integration_time_domain = (np.arange(SAMPLES_PER_PRN_TRANSMISSION) / SAMPLES_PER_SECOND) + (sample_index / SAMPLES_PER_SECOND)
        doppler_shift_carrier = np.exp(-1j * math.tau * doppler_shift * integration_time_domain)
        doppler_shifted_antenna_data_chunk = chunk_that_may_contain_one_prn * doppler_shift_carrier

        correlation_result = frequency_domain_correlation(doppler_shifted_antenna_data_chunk, prn_as_complex)

        if integration_type == IntegrationType.Coherent:
            integrated_correlation_result += correlation_result
        elif integration_type == IntegrationType.NonCoherent:
            integrated_correlation_result += np.abs(correlation_result)
        else:
            raise ValueError("Unexpected integration type")

    return integrated_correlation_result


class GpsSatelliteDetector:
    def __init__(self, satellites_by_id: dict[GpsSatelliteId, GpsSatellite]) -> None:
        self.satellites_by_id = satellites_by_id
        self._cached_correlation_profiles: dict[Any, _CorrelationProfile] = {}

    def detect_satellites_in_antenna_data(
        self,
        satellites_to_search_for: list[GpsSatelliteId],
        antenna_data: _AntennaSamplesSpanningAcquisitionIntegrationPeriodMs,
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
                #center_doppler_shift_estimation = best_non_coherent_correlation_profile_across_all_search_space.doppler_shift
                _logger.info(
                    f"Found a better candidate Doppler for SV({satellite_id}): "
                    f"(Found in [{center_doppler_shift_estimation - doppler_frequency_estimation_spread:.2f} | "
                    f"{center_doppler_shift_estimation:.2f} | {center_doppler_shift_estimation + doppler_frequency_estimation_spread:.2f}], "
                    f"Strength: "
                    f"{best_non_coherent_correlation_profile_across_all_search_space.correlation_strength:.2f}"
                )

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
        if False and key in self._cached_correlation_profiles:
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


@dataclass
class GpsSatelliteTrackingParameters:
    satellite: GpsSatellite
    current_doppler_shift: _DopplerShiftHz
    current_carrier_wave_phase_shift: _CarrierWavePhaseInRadians
    current_prn_code_phase_shift: _PrnCodePhaseInSamples

    doppler_shifts: list[_DopplerShiftHz]
    carrier_wave_phases: list[_CarrierWavePhaseInRadians]
    carrier_wave_phase_errors: list[float]
    navigation_bit_pseudosymbols: list[int]


class GpsReceiver:
    def __init__(self, antenna_samples_provider: AntennaSampleProvider) -> None:
        self.antenna_samples_provider = antenna_samples_provider

        # Generate the replica signals that we'll use to correlate against the received antenna signals upfront
        satellites_to_replica_prn_signals = generate_replica_prn_signals()
        self.satellites_by_id = {
            satellite_id: GpsSatellite(satellite_id=satellite_id, prn_code=code)
            for satellite_id, code in satellites_to_replica_prn_signals.items()
        }

        self.tracked_satellites = [
            GpsSatelliteTrackingParameters(
                satellite=self.satellites_by_id[GpsSatelliteId(id=3)],
                #current_doppler_shift=2800,
                current_doppler_shift=2450,
                #current_carrier_wave_phase_shift=3.0287323328394664,
                current_prn_code_phase_shift=476,
                #correlation_strength=101.1720157896715
                #current_carrier_wave_phase_shift=2.533448909741912,
                current_carrier_wave_phase_shift=-1.413635681996738,
                #correlation_strength=102.15,
                doppler_shifts=[],
                carrier_wave_phases=[],
                carrier_wave_phase_errors=[],
                navigation_bit_pseudosymbols=[],
            ),
            GpsSatelliteTrackingParameters(
                satellite=self.satellites_by_id[GpsSatelliteId(id=12)],
                current_doppler_shift=778,
                current_carrier_wave_phase_shift=-0.10151988803983533,
                current_prn_code_phase_shift=720,
                #correlation_strength=101.60051260802132
                doppler_shifts=[],
                carrier_wave_phases=[],
                carrier_wave_phase_errors=[],
                navigation_bit_pseudosymbols=[],
            ),
            GpsSatelliteTrackingParameters(
                satellite=self.satellites_by_id[GpsSatelliteId(id=25)],
                current_doppler_shift=3150,
                current_carrier_wave_phase_shift=2.3901717091201107,
                current_prn_code_phase_shift=178,
                #correlation_strength=197.00508614060738
                doppler_shifts=[],
                carrier_wave_phases=[],
                carrier_wave_phase_errors=[],
                navigation_bit_pseudosymbols=[],
            ),
            GpsSatelliteTrackingParameters(
                satellite=self.satellites_by_id[GpsSatelliteId(id=28)],
                current_doppler_shift=5600,
                current_carrier_wave_phase_shift=-0.5132165873027408,
                current_prn_code_phase_shift=374,
                #correlation_strength=184.6157516415438
                doppler_shifts=[],
                carrier_wave_phases=[],
                carrier_wave_phase_errors=[],
                navigation_bit_pseudosymbols=[],
            ),
            GpsSatelliteTrackingParameters(
                satellite=self.satellites_by_id[GpsSatelliteId(id=29)],
                current_doppler_shift=6300,
                current_carrier_wave_phase_shift=0.3169472758230659,
                current_prn_code_phase_shift=639,
                #correlation_strength=122.33878610414894
                doppler_shifts=[],
                carrier_wave_phases=[],
                carrier_wave_phase_errors=[],
                navigation_bit_pseudosymbols=[],
            ),
            GpsSatelliteTrackingParameters(
                satellite=self.satellites_by_id[GpsSatelliteId(id=31)],
                current_doppler_shift=6300,
                current_carrier_wave_phase_shift=2.612177767316696,
                current_prn_code_phase_shift=236,
                #correlation_strength=234.71669484893457
                doppler_shifts=[],
                carrier_wave_phases=[],
                carrier_wave_phase_errors=[],
                navigation_bit_pseudosymbols=[],
            ),
            GpsSatelliteTrackingParameters(
                satellite=self.satellites_by_id[GpsSatelliteId(id=32)],
                current_doppler_shift=1750,
                #current_doppler_shift=1800,
                current_carrier_wave_phase_shift=0.8955068896739946,
                current_prn_code_phase_shift=468,
                #correlation_strength=302.7964454627832
                doppler_shifts=[],
                carrier_wave_phases=[],
                carrier_wave_phase_errors=[],
                navigation_bit_pseudosymbols=[],
            )
        ]
        #self.tracked_satellites = self.tracked_satellites[:1]
        self.tracked_satellites = []
        self.satellite_ids_eligible_for_acquisition = deepcopy(ALL_SATELLITE_IDS)
        #self.satellite_ids_eligible_for_acquisition = [GpsSatelliteId(id=3), GpsSatelliteId(id=12), GpsSatelliteId(id=25)]
        #self.satellite_ids_eligible_for_acquisition = [GpsSatelliteId(id=32), GpsSatelliteId(id=3)]

        self.satellite_detector = GpsSatelliteDetector(self.satellites_by_id)
        # Used during acquisition to integrate correlation over a longer period than a millisecond.
        self.rolling_samples_buffer = collections.deque(maxlen=ACQUISITION_INTEGRATION_PERIOD_MS)

    def step(self):
        """Run one 'iteration' of the GPS receiver. This consumes one millisecond of antenna data."""
        sample_index = self.antenna_samples_provider.cursor
        samples: _AntennaSamplesSpanningOneMs = self.antenna_samples_provider.get_samples(SAMPLES_PER_PRN_TRANSMISSION)
        # PT: Instead of trying to find a roll that works across all time, dynamically adjust where we consider the start to be?

        if (len(self.tracked_satellites) and len(self.tracked_satellites[-1].carrier_wave_phase_errors) > 6000) or len(samples) < SAMPLES_PER_PRN_TRANSMISSION:
            for sat in reversed(self.tracked_satellites):
                self.decode_nav_bits(sat)
                plt.plot(sat.doppler_shifts[::50])
                plt.title(f"Doppler shift for {sat.satellite.satellite_id.id}")
                plt.show(block=True)
                plt.plot(sat.carrier_wave_phases[::50])
                plt.title(f"Carrier phase for {sat.satellite.satellite_id.id}")
                plt.show(block=True)
                plt.plot(sat.carrier_wave_phase_errors[::50])
                plt.title(f"Carrier phase errors for {sat.satellite.satellite_id.id}")
                plt.show(block=True)
            import sys
            sys.exit(0)

        # Firstly, record this sample in our rolling buffer
        self.rolling_samples_buffer.append(samples)

        # If we need to perform acquisition, do so now
        if len(self.tracked_satellites) < MINIMUM_TRACKED_SATELLITES_FOR_POSITION_FIX:
            _logger.info(
                f"Will perform acquisition search because we're only "
                f"tracking {len(self.tracked_satellites)} satellites"
            )
            self._perform_acquisition()

        # Continue tracking each acquired satellite
        self._track_acquired_satellites(samples, sample_index)

    def decode_nav_bits(self, sat: GpsSatelliteTrackingParameters):
        navigation_bit_pseudosymbols = sat.navigation_bit_pseudosymbols
        confidence_scores = []
        for roll in range(0, 20):
            phase_shifted_bits = navigation_bit_pseudosymbols[roll:]
            confidences = []
            for twenty_pseudosymbols in chunks(phase_shifted_bits, 20):
                integrated_value = sum(twenty_pseudosymbols)
                confidences.append(abs(integrated_value))
            # Compute an overall confidence score for this offset
            confidence_scores.append(np.mean(confidences))

        #print(f"Confidence scores: {confidence_scores}")
        best_offset = np.argmax(confidence_scores)
        print(f"Best Offset: {best_offset} ({confidence_scores[best_offset]})")

        bit_phase = best_offset
        phase_shifted_bits = navigation_bit_pseudosymbols[bit_phase:]
        bits = []
        for twenty_pseudosymbols in chunks(phase_shifted_bits, 20):
            integrated_value = sum(twenty_pseudosymbols)
            bit_value = np.sign(integrated_value)
            bits.append(bit_value)

        digital_bits = [1 if b == 1.0 else 0 for b in bits]
        inverted_bits = [0 if b == 1.0 else 1 for b in bits]
        print(f"Bit count: {len(digital_bits)}")
        print(f"Bits:          {digital_bits}")
        print(f"Inverted bits: {inverted_bits}")

        preamble = [1, 0, 0, 0, 1, 0, 1, 1]
        print(f"Preamble {preamble} found in bits? {does_list_contain_sublist(digital_bits, preamble)}")
        print(f"Preamble {preamble} found in inverted bits? {does_list_contain_sublist(inverted_bits, preamble)}")

        def get_matches(l, sub):
            return [l[pos : pos + len(sub)] == sub for pos in range(0, len(l) - len(sub) + 1)]

        preamble_starts_in_digital_bits = [
            x[0] for x in (np.argwhere(np.array(get_matches(digital_bits, preamble)) == True))
        ]
        print(f"Preamble starts in bits:          {preamble_starts_in_digital_bits}")
        preamble_starts_in_inverted_bits = [
            x[0] for x in (np.argwhere(np.array(get_matches(inverted_bits, preamble)) == True))
        ]
        print(f"Preamble starts in inverted bits: {preamble_starts_in_inverted_bits}")

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
        newly_acquired_satellites = self.satellite_detector.detect_satellites_in_antenna_data(
            self.satellite_ids_eligible_for_acquisition,
            samples_for_integration_period,
        )
        for satellite_acquisition_result in newly_acquired_satellites:
            self.tracked_satellites.append(
                GpsSatelliteTrackingParameters(
                    satellite=self.satellites_by_id[satellite_acquisition_result.satellite_id],
                    current_doppler_shift=satellite_acquisition_result.doppler_shift,
                    current_carrier_wave_phase_shift=satellite_acquisition_result.carrier_wave_phase_shift,
                    current_prn_code_phase_shift=satellite_acquisition_result.prn_phase_shift,
                    doppler_shifts=[],
                    carrier_wave_phases=[],
                    carrier_wave_phase_errors=[],
                    navigation_bit_pseudosymbols=[],
                )
            )

    def _track_acquired_satellites(self, samples: _AntennaSamplesSpanningOneMs, sample_index: int):
        for satellite in self.tracked_satellites:
            self._track_satellite(satellite, samples, sample_index)

    def _track_satellite(
        self,
        satellite_tracking_params: GpsSatelliteTrackingParameters,
        samples: _AntennaSamplesSpanningOneMs,
        sample_index: int
    ) -> None:
        loop_bandwidth = 2.046 / 1000
        # Common choice for zeta, considered optimal
        damping_factor = math.sqrt(2) / 2.0
        # Natural frequency
        natural_freq = loop_bandwidth / (damping_factor * (1 + damping_factor ** 2) ** 0.5)
        # This represents the gain of *instantaneous* error correction,
        # which applies to the estimate of the carrier wave phase.
        # Also called 'alpha'.
        loop_gain_phase = (4 * damping_factor * natural_freq) / (1 + ((2 * damping_factor * natural_freq) + (natural_freq ** 2)))
        # This represents the *integrated* error correction,
        # which applies to the estimate of the Doppler shifted frequency.
        # Also called 'beta'.
        loop_gain_freq = (4 * (natural_freq ** 2)) / (1 + ((2 * damping_factor * natural_freq) + (natural_freq ** 2)))

        # Generate Doppler-shifted and phase-shifted carrier wave
        time_domain = (np.arange(SAMPLES_PER_PRN_TRANSMISSION) / SAMPLES_PER_SECOND) + (sample_index / SAMPLES_PER_SECOND)
        doppler_shift_carrier = np.exp(-1j * ((2 * np.pi * satellite_tracking_params.current_doppler_shift * time_domain) + satellite_tracking_params.current_carrier_wave_phase_shift))
        doppler_shifted_samples = samples * doppler_shift_carrier

        # Correlate early, prompt, and late phase versions of the PRN
        unslid_prn = satellite_tracking_params.satellite.prn_as_complex
        prompt_prn = np.roll(unslid_prn, satellite_tracking_params.current_prn_code_phase_shift)

        coherent_prompt_correlation = frequency_domain_correlation(doppler_shifted_samples, prompt_prn)
        non_coherent_prompt_correlation = np.abs(coherent_prompt_correlation)
        non_coherent_prompt_peak_offset = np.argmax(non_coherent_prompt_correlation)
        non_coherent_prompt_peak = non_coherent_prompt_correlation[non_coherent_prompt_peak_offset]

        # Recenter the code phase offset so that it looks positive or negative, depending on where the offset sits
        # in the period of the PRN.
        if non_coherent_prompt_peak_offset <= SAMPLES_PER_PRN_TRANSMISSION / 2:
            centered_non_coherent_prompt_peak_offset = non_coherent_prompt_peak_offset
        else:
            centered_non_coherent_prompt_peak_offset = non_coherent_prompt_peak_offset - SAMPLES_PER_PRN_TRANSMISSION + 1

        logging.info(
            f"Peak offset {non_coherent_prompt_peak_offset}, centered offset {centered_non_coherent_prompt_peak_offset}"
        )
        if centered_non_coherent_prompt_peak_offset > 0:
            satellite_tracking_params.current_prn_code_phase_shift += centered_non_coherent_prompt_peak_offset
        else:
            satellite_tracking_params.current_prn_code_phase_shift -= centered_non_coherent_prompt_peak_offset

        # Finally, ensure we're always sliding within one PRN transmission
        satellite_tracking_params.current_prn_code_phase_shift = int(satellite_tracking_params.current_prn_code_phase_shift) % SAMPLES_PER_PRN_TRANSMISSION

        # Ensure carrier wave alignment
        new_prompt_prn = np.roll(unslid_prn, satellite_tracking_params.current_prn_code_phase_shift)
        new_coherent_prompt_correlation = frequency_domain_correlation(doppler_shifted_samples, new_prompt_prn)
        new_non_coherent_prompt_correlation = np.abs(new_coherent_prompt_correlation)
        new_non_coherent_prompt_peak_offset = np.argmax(new_non_coherent_prompt_correlation)
        new_coherent_prompt_prn_correlation_peak = new_coherent_prompt_correlation[new_non_coherent_prompt_peak_offset]

        I = np.real(new_coherent_prompt_prn_correlation_peak)
        Q = np.imag(new_coherent_prompt_prn_correlation_peak)
        carrier_wave_phase_error = I * Q

        satellite_tracking_params.current_doppler_shift += loop_gain_freq * carrier_wave_phase_error
        satellite_tracking_params.current_carrier_wave_phase_shift += loop_gain_phase * carrier_wave_phase_error
        satellite_tracking_params.current_carrier_wave_phase_shift %= math.tau

        navigation_bit_pseudosymbol_value = int(np.sign(new_coherent_prompt_prn_correlation_peak))
        satellite_tracking_params.navigation_bit_pseudosymbols.append(navigation_bit_pseudosymbol_value)

        logging.info(f"Doppler shift {satellite_tracking_params.current_doppler_shift:.2f}")
        logging.info(f"Carrier phase {satellite_tracking_params.current_carrier_wave_phase_shift:.8f}")
        logging.info(f"Code phase {satellite_tracking_params.current_prn_code_phase_shift}")

        satellite_tracking_params.doppler_shifts.append(satellite_tracking_params.current_doppler_shift)
        satellite_tracking_params.carrier_wave_phases.append(satellite_tracking_params.current_carrier_wave_phase_shift)
        satellite_tracking_params.carrier_wave_phase_errors.append(carrier_wave_phase_error)
