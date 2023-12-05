import collections
import functools
import logging
from typing import Tuple

import math
from dataclasses import dataclass
from enum import Enum, auto
import matplotlib.pyplot as plt

import numpy as np

from gypsum.antenna_sample_provider import Seconds
from gypsum.constants import SAMPLES_PER_PRN_TRANSMISSION, SAMPLES_PER_SECOND
from gypsum.satellite import GpsSatellite
from gypsum.utils import (
    AntennaSamplesSpanningOneMs,
    CarrierWavePhaseInRadians,
    DopplerShiftHz,
    PrnCodePhaseInSamples,
    frequency_domain_correlation,
)
from gypsum.utils import CoherentCorrelationPeak

_logger = logging.getLogger(__name__)


class BitValue(Enum):
    UNKNOWN = auto()
    ZERO = auto()
    ONE = auto()

    @classmethod
    def from_val(cls, val: int) -> 'BitValue':
        return {
            0: BitValue.ZERO,
            1: BitValue.ONE,
        }[val]

    def as_val(self) -> int:
        if self == BitValue.UNKNOWN:
            raise ValueError(f'Cannot convert an unknown bit value into an integer')

        return {
            BitValue.ZERO: 0,
            BitValue.ONE: 1,
        }[self]

    def inverted(self) -> 'BitValue':
        if self == BitValue.UNKNOWN:
            raise ValueError(f'Cannot invert an unknown bit value')

        return {
            BitValue.ZERO: BitValue.ONE,
            BitValue.ONE: BitValue.ZERO,
        }[self]

    def __eq__(self, other) -> bool:
        if not isinstance(other, BitValue):
            return False
        return self.value == other.value

    def __hash__(self) -> int:
        return hash(self.value)


class NavigationBitPseudosymbol(Enum):
    MINUS_ONE = auto()
    ONE = auto()

    @classmethod
    def from_val(cls, val: int) -> 'NavigationBitPseudosymbol':
        return {
            -1: NavigationBitPseudosymbol.MINUS_ONE,
            1: NavigationBitPseudosymbol.ONE,
        }[val]

    def as_val(self) -> int:
        return {
            NavigationBitPseudosymbol.MINUS_ONE: -1,
            NavigationBitPseudosymbol.ONE: 1,
        }[self]


# The tracker runs at 1000Hz by definition, since it always operates on the output of a 1ms-long PRN correlation.
_TRACKER_ITERATIONS_PER_SECOND = 1000


@dataclass
class GpsSatelliteTrackingParameters:
    """This also maintains state about the tracking history / various tracking metrics.
    This is used both as part of the tracker's fundamental work, and for data visualization."""
    satellite: GpsSatellite
    current_doppler_shift: DopplerShiftHz
    current_carrier_wave_phase_shift: CarrierWavePhaseInRadians
    current_prn_code_phase_shift: PrnCodePhaseInSamples

    doppler_shifts: list[DopplerShiftHz]
    carrier_wave_phases: list[CarrierWavePhaseInRadians]
    carrier_wave_phase_errors: list[float]
    navigation_bit_pseudosymbols: list[NavigationBitPseudosymbol]

    # The following arguments are handled automatically by this implementation
    correlation_peaks_rolling_buffer: collections.deque = None
    correlation_peak_angles: list = None

    def __post_init__(self) -> None:
        for field in [
            self.correlation_peaks_rolling_buffer,
            self.correlation_peak_angles,
        ]:
            if field is not None:
                raise RuntimeError(f'This field is not intended to be initialized at a call site.')
        # Maintain a rolling buffer of the last few correlation peaks we've seen. Integrating these peaks over time
        # allows us to track the signal modulation (i.e. in a constellation plot).
        # The tracker runs at 1000Hz, so this represents the last n seconds of tracking.
        # TODO(PT): Pull this out into a constant?
        self.correlation_peaks_rolling_buffer = collections.deque(maxlen=_TRACKER_ITERATIONS_PER_SECOND * 1)
        self.correlation_peak_angles = []

    def is_locked(self) -> bool:
        """Apply heuristics to the recorded tracking metrics history to give an answer whether the tracker is 'locked'.
        'Locked' means we feel confident we're accurately tracking the carrier wave frequency and phase.
        """
        # TODO(PT): This should be cached for each loop iteration somehow...
        # The PLL currently runs at 1000Hz, so each error entry is spaced at 1ms.
        # TODO(PT): Pull this out into a constant.
        previous_milliseconds_to_consider = 250
        if len(self.carrier_wave_phase_errors) < previous_milliseconds_to_consider:
            # We haven't run our PLL for long enough to determine lock
            _logger.info(f'Not enough errors to determine variance')
            return False

        last_few_phase_errors = self.carrier_wave_phase_errors[-previous_milliseconds_to_consider:]
        phase_error_variance = np.var(last_few_phase_errors)
        # TODO(PT): Pull this out into a constant?
        is_phase_error_variance_under_threshold = phase_error_variance < 900

        # Default to claiming the I channel is fine if we don't have enough samples to make a proper decision
        does_i_channel_look_locked = True
        # Same with the constellation rotation
        is_constellation_rotation_acceptable = True

        last_few_peaks = np.array(list(self.correlation_peaks_rolling_buffer)[-previous_milliseconds_to_consider:])
        if len(self.correlation_peaks_rolling_buffer) > 2:
            # A locked `I` channel should output values strongly centered around a positive pole and a negative pole.
            # We don't know the exact values of these poles, as they'll depend on the exact signal, but we can split
            # our `I` channel into positive and negative components and try to see how strongly values are clustered
            # around each pole.
            peaks_on_negative_pole = last_few_peaks[last_few_peaks.real < 0]
            peaks_on_positive_pole = last_few_peaks[last_few_peaks.real >= 0]
            mean_negative_peak = np.mean(peaks_on_negative_pole)
            # mean_positive_peak = np.mean(peaks_on_positive_pole)

            negative_i_peak_variance = np.var(peaks_on_negative_pole.real)
            positive_i_peak_variance = np.var(peaks_on_positive_pole.real)
            mean_i_peak_variance = (negative_i_peak_variance + positive_i_peak_variance) / 2.0
            # PT: Chosen through experimentation
            does_i_channel_look_locked = mean_i_peak_variance < 2

            # Interrogate the constellation rotation
            angle = 180 - (((np.arctan2(mean_negative_peak.imag, mean_negative_peak.real) / math.tau) * 360) % 180)
            centered_angle = angle if angle < 90 else 180 - angle
            is_constellation_rotation_acceptable = abs(centered_angle < 6)

        return (
            is_phase_error_variance_under_threshold
            and does_i_channel_look_locked
            and is_constellation_rotation_acceptable
        )


class GpsSatelliteTracker:
    def __init__(self, tracking_params: GpsSatelliteTrackingParameters) -> None:
        self.tracking_params = tracking_params
        # PT: Small optimization here. Each time we process a millisecond of samples, we need to generate a time
        # domain representing the time offset of the samples from when we began tracking. This involves creating the
        # range below, plus a phase offset representing the current offset from when we started tracking. We save work
        # by generating the correctly-spaced range just once upfront, then applying the phase offset for the current
        # time each iteration.
        self.time_domain_for_1ms = np.arange(SAMPLES_PER_PRN_TRANSMISSION) / SAMPLES_PER_SECOND

    # We only have two PLL modes, so an LRU cache with 2 entries should be sufficient.
    @functools.lru_cache(maxsize=2)
    def _calculate_loop_filter_alpha_and_beta(self, loop_bandwidth: float) -> Tuple[float, float]:
        time_per_sample = 1.0 / SAMPLES_PER_SECOND
        # Common choice for zeta, considered optimal
        damping_factor = 1.0 / math.sqrt(2)

        # This represents the gain of *instantaneous* error correction,
        # which applies to the estimate of the carrier wave phase.
        # Also called 'alpha'.
        # This is the 'first order' of the loop.
        loop_gain_phase = 4 * damping_factor * loop_bandwidth * time_per_sample
        # This represents the *integrated* error correction,
        # which applies to the estimate of the Doppler shifted frequency.
        # Also called 'beta'.
        # This is the 'second order' of the loop (as frequency is the derivative of phase).
        loop_gain_freq = 4 * (loop_bandwidth ** 2) * time_per_sample

        return loop_gain_phase, loop_gain_freq

    def _run_carrier_wave_tracking_loop_iteration(self, correlation_peak: CoherentCorrelationPeak) -> None:
        # Classic error discriminator for a Costas-style PLL loop,
        # since it's not sensitive to a 180 degree phase rotation.
        error = correlation_peak.real * correlation_peak.imag

        if self.tracking_params.is_locked():
            # When we detect our PLL is locked, use a 'fine-grained'/'track' mode with a low loop bandwidth.
            alpha, beta = self._calculate_loop_filter_alpha_and_beta(3)
        else:
            # When unlocked, use a wider 'pull-in' bandwidth to try to get back on track.
            alpha, beta = self._calculate_loop_filter_alpha_and_beta(6)

        self.tracking_params.current_carrier_wave_phase_shift += error * alpha
        self.tracking_params.current_carrier_wave_phase_shift %= math.tau
        self.tracking_params.current_doppler_shift += error * beta
        self.tracking_params.carrier_wave_phase_errors.append(error)
        self.tracking_params.correlation_peak_angles.append(np.angle(correlation_peak)) # type: ignore

    def _run_prn_code_tracking_loop_iteration(
        self,
        seconds_since_start: Seconds,
        samples: AntennaSamplesSpanningOneMs
    ) -> Tuple[CoherentCorrelationPeak, NavigationBitPseudosymbol]:
        params = self.tracking_params
        # Adjust the time domain based on our current time
        time_domain = self.time_domain_for_1ms + seconds_since_start

        # Generate Doppler-shifted and phase-shifted carrier wave, based on our current carrier wave estimation.
        # (Note that there's a circular dependency between the carrier wave tracker and the PRN code tracker.
        # This loop will update the PRN code loop tracker by first demodulating with the current estimate of the carrier
        # wave. The carrier wave tracker will similarly demodulate with the current estimation of the PRN code tracker,
        # and so on).
        doppler_shift_carrier = np.exp(
            -1j * ((2 * np.pi * params.current_doppler_shift * time_domain) + params.current_carrier_wave_phase_shift)
        )
        doppler_shifted_samples = samples * doppler_shift_carrier

        # Correlate early, prompt, and late phase versions of the PRN
        unslid_prn = params.satellite.prn_as_complex
        prompt_prn = np.roll(unslid_prn, params.current_prn_code_phase_shift)   # type: ignore

        coherent_prompt_correlation = frequency_domain_correlation(doppler_shifted_samples, prompt_prn)
        non_coherent_prompt_correlation = np.abs(coherent_prompt_correlation)
        non_coherent_prompt_peak_offset = np.argmax(non_coherent_prompt_correlation)

        # Recenter the code phase offset so that it looks positive or negative, depending on where the offset sits
        # in the period of the PRN.
        if non_coherent_prompt_peak_offset <= SAMPLES_PER_PRN_TRANSMISSION / 2:
            centered_non_coherent_prompt_peak_offset = non_coherent_prompt_peak_offset
        else:
            centered_non_coherent_prompt_peak_offset = non_coherent_prompt_peak_offset - SAMPLES_PER_PRN_TRANSMISSION

        if centered_non_coherent_prompt_peak_offset > 0:
            params.current_prn_code_phase_shift += 1
        elif centered_non_coherent_prompt_peak_offset < 0:
            params.current_prn_code_phase_shift -= 1

        # Finally, ensure we're always sliding within one PRN transmission
        params.current_prn_code_phase_shift = int(params.current_prn_code_phase_shift) % SAMPLES_PER_PRN_TRANSMISSION

        coherent_prompt_prn_correlation_peak = coherent_prompt_correlation[non_coherent_prompt_peak_offset]

        # The sign of the correlation peak *is* the transmitted pseudosymbol value, since the signal is BPSK-modulated.
        navigation_bit_pseudosymbol_value = int(np.sign(coherent_prompt_prn_correlation_peak.real))
        navigation_bit_pseudosymbol = NavigationBitPseudosymbol.from_val(navigation_bit_pseudosymbol_value)

        return coherent_prompt_prn_correlation_peak, navigation_bit_pseudosymbol

    def process_samples(self, seconds_since_start: Seconds, samples: AntennaSamplesSpanningOneMs) -> NavigationBitPseudosymbol:
        params = self.tracking_params

        # First, run an iteration of the PRN code tracking loop
        coherent_prompt_prn_correlation_peak, navigation_bit_pseudosymbol = self._run_prn_code_tracking_loop_iteration(
            seconds_since_start,
            samples,
        )
        params.navigation_bit_pseudosymbols.append(navigation_bit_pseudosymbol)

        self.tracking_params.correlation_peaks_rolling_buffer.append(coherent_prompt_prn_correlation_peak)

        # Next, run an iteration of the carrier wave tracking loop
        self._run_carrier_wave_tracking_loop_iteration(
            coherent_prompt_prn_correlation_peak,
        )

        params.doppler_shifts.append(params.current_doppler_shift)
        params.carrier_wave_phases.append(params.current_carrier_wave_phase_shift)

        points = self.tracking_params.correlation_peaks_rolling_buffer
        points_on_left_pole = [p for p in points if p.real < 0]
        if len(points_on_left_pole) > 2:
            left_point = np.mean(points_on_left_pole)
            angle = 180 - (((np.arctan2(left_point.imag, left_point.real) / math.tau) * 360) % 180)
            rotation = angle
            if angle > 90:
                rotation = angle - 180
            if not params.is_locked():
                filtered_rotation = rotation * 0.00005
                self.tracking_params.current_doppler_shift -= filtered_rotation

        return navigation_bit_pseudosymbol
