import functools
import logging
import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Self

import numpy as np

from gypsum.antenna_sample_provider import ReceiverTimestampSeconds
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


@dataclass
class GpsSatelliteTrackingParameters:
    satellite: GpsSatellite
    current_doppler_shift: DopplerShiftHz
    current_carrier_wave_phase_shift: CarrierWavePhaseInRadians
    current_prn_code_phase_shift: PrnCodePhaseInSamples

    doppler_shifts: list[DopplerShiftHz]
    carrier_wave_phases: list[CarrierWavePhaseInRadians]
    carrier_wave_phase_errors: list[float]
    navigation_bit_pseudosymbols: list[int]


class GpsSatelliteTracker:
    def __init__(self, tracking_params: GpsSatelliteTrackingParameters, loop_bandwidth: float) -> None:
        self.tracking_params = tracking_params
        self.loop_bandwidth = loop_bandwidth
        # Common choice for zeta, considered optimal
        damping_factor = math.sqrt(2) / 2.0
        # Natural frequency
        natural_freq = loop_bandwidth / (damping_factor * (1 + damping_factor**2) ** 0.5)
        # This represents the gain of *instantaneous* error correction,
        # which applies to the estimate of the carrier wave phase.
        # Also called 'alpha'.
        self.loop_gain_phase = (4 * damping_factor * natural_freq) / (
            1 + ((2 * damping_factor * natural_freq) + (natural_freq**2))
        )
        # This represents the *integrated* error correction,
        # which applies to the estimate of the Doppler shifted frequency.
        # Also called 'beta'.
        self.loop_gain_freq = (4 * (natural_freq**2)) / (
            1 + ((2 * damping_factor * natural_freq) + (natural_freq**2))
        )

    def _is_locked(self) -> bool:
        # The PLL currently runs at 1000Hz, so each error entry is spaced at 1ms.
        # TODO(PT): Pull this out into a constant.
        previous_milliseconds_to_consider = 250
        if len(self.tracking_params.carrier_wave_phase_errors) < previous_milliseconds_to_consider:
            # We haven't run our PLL for long enough to determine lock
            return False
        last_few_phase_errors = self.tracking_params.carrier_wave_phase_errors[-previous_milliseconds_to_consider:]
        phase_error_variance = np.var(last_few_phase_errors)
        # TODO(PT): Pull this out into a constant?
        is_phase_error_variance_under_threshold = phase_error_variance < 900

        # Default to claiming the I channel is fine if we don't have enough samples to make a proper decision
        does_i_channel_look_locked = True
        if len(self._is) > 2:
            last_few_i_values = self._is[-previous_milliseconds_to_consider:]
            #import statistics
            #s = statistics.stdev(last_few_i_values)
            s = np.var(last_few_i_values)
            # A locked `I` channel should output values strongly centered around a positive pole and a negative pole.
            # We don't know the exact values of these poles, as they'll depend on the exact signal, but we can split
            # our `I` channel into positive and negative components and try to see how strongly values are clustered
            # around each pole.
            positive_i_values = [x for x in last_few_i_values if x >= 0]
            positive_var = np.var(positive_i_values)
            negative_i_values = [x for x in last_few_i_values if x < 0]
            negative_var = np.var(negative_i_values)
            s = (positive_var + negative_var) / 2.0
            #print(f'stdev: {s:.2f}')
            # PT: Chosen through experimentation
            does_i_channel_look_locked = s < 2
            # Prev 900, 2, 6

        points = list(complex(i, q) for i, q in zip(self._is, self._qs))
        is_constellation_rotation_acceptable = True
        if len(points) > 2:
            points_on_left_pole = [p for p in points if p.real < 0]
            points_on_right_pole = [p for p in points if p.real >= 0]
            left_point = np.mean(points_on_left_pole)
            # right_point = np.mean(points_on_right_pole)
            angle = 180 - (((np.arctan2(left_point.imag, left_point.real) / math.tau) * 360) % 180)
            centered_angle = angle if angle < 90 else 180 - angle
            is_constellation_rotation_acceptable = abs(centered_angle < 6)

        #return is_phase_error_variance_under_threshold

        return (
            is_phase_error_variance_under_threshold
            and does_i_channel_look_locked
            and is_constellation_rotation_acceptable
        )

    @functools.lru_cache
    def _calculate_loop_filter_alpha_and_beta(self, loop_bandwidth: float) -> Tuple[float, float]:
        # Common choice for zeta, considered optimal
        damping_factor = math.sqrt(2) / 2.0
        # Natural frequency
        natural_freq = loop_bandwidth / (damping_factor * (1 + damping_factor**2) ** 0.5)
        # This represents the *integrated* error correction,
        # which applies to the estimate of the Doppler shifted frequency.
        # Also called 'beta'.
        loop_gain_freq = (4 * damping_factor * natural_freq) / (
                1 + ((2 * damping_factor * natural_freq) + (natural_freq**2))
        )
        # This represents the gain of *instantaneous* error correction,
        # which applies to the estimate of the carrier wave phase.
        # Also called 'alpha'.
        loop_gain_phase = (4 * (natural_freq**2)) / (
                1 + ((2 * damping_factor * natural_freq) + (natural_freq**2))
        )

        time_per_sample = 1.0 / SAMPLES_PER_SECOND
        zeta = 1.0 / math.sqrt(2)
        loop_gain_phase = 4 * zeta * loop_bandwidth * time_per_sample
        loop_gain_freq = 4 * (loop_bandwidth ** 2) * time_per_sample

        #factor = 1.0
        #return loop_gain_phase / factor, loop_gain_freq / factor
        return loop_gain_phase, loop_gain_freq

    def process_samples(self, seconds_since_start: Seconds, samples: AntennaSamplesSpanningOneMs) -> NavigationBitPseudosymbol:
        params = self.tracking_params

        # Generate Doppler-shifted and phase-shifted carrier wave
        time_domain = (np.arange(SAMPLES_PER_PRN_TRANSMISSION) / SAMPLES_PER_SECOND) + seconds_since_start
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
        # non_coherent_prompt_peak = non_coherent_prompt_correlation[non_coherent_prompt_peak_offset]

        # Recenter the code phase offset so that it looks positive or negative, depending on where the offset sits
        # in the period of the PRN.
        if non_coherent_prompt_peak_offset <= SAMPLES_PER_PRN_TRANSMISSION / 2:
            centered_non_coherent_prompt_peak_offset = non_coherent_prompt_peak_offset
        else:
            centered_non_coherent_prompt_peak_offset = non_coherent_prompt_peak_offset - SAMPLES_PER_PRN_TRANSMISSION

        # logging.info(
        #    f"Peak offset {non_coherent_prompt_peak_offset}, centered offset {centered_non_coherent_prompt_peak_offset}"
        # )
        if centered_non_coherent_prompt_peak_offset > 0:
            params.current_prn_code_phase_shift += 1
        elif centered_non_coherent_prompt_peak_offset < 0:
            params.current_prn_code_phase_shift -= 1

        # Finally, ensure we're always sliding within one PRN transmission
        params.current_prn_code_phase_shift = int(params.current_prn_code_phase_shift) % SAMPLES_PER_PRN_TRANSMISSION

        coherent_prompt_prn_correlation_peak = coherent_prompt_correlation[non_coherent_prompt_peak_offset]

        I = np.real(coherent_prompt_prn_correlation_peak)
        Q = np.imag(coherent_prompt_prn_correlation_peak)
        carrier_wave_phase_error = I * Q

        params.current_doppler_shift += self.loop_gain_freq * carrier_wave_phase_error
        params.current_carrier_wave_phase_shift += self.loop_gain_phase * carrier_wave_phase_error
        params.current_carrier_wave_phase_shift %= math.tau

        # for i, s in enumerate(coherent_prompt_correlation):
        #    i2 = np.real(s)
        #    q2 = np.imag(s)
        #    sample_error = (i2 * q2) / 2
        #    params.current_carrier_wave_phase_shift += self.loop_gain_phase * sample_error
        #    #params.current_carrier_wave_phase_shift %= math.tau

        navigation_bit_pseudosymbol_value = int(np.sign(coherent_prompt_prn_correlation_peak))
        params.navigation_bit_pseudosymbols.append(navigation_bit_pseudosymbol_value)

        # logging.info(f"Doppler shift {params.current_doppler_shift:.2f}")
        # logging.info(f"Code phase {params.current_prn_code_phase_shift}")
        # logging.info(f"Carrier phase {params.current_carrier_wave_phase_shift:.8f}")

        params.doppler_shifts.append(params.current_doppler_shift)
        params.carrier_wave_phases.append(params.current_carrier_wave_phase_shift)
        params.carrier_wave_phase_errors.append(carrier_wave_phase_error)

        return NavigationBitPseudosymbol.from_val(navigation_bit_pseudosymbol_value)
