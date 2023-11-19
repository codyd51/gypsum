import logging
import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Self

import numpy as np

from gypsum.antenna_sample_provider import ReceiverTimestampSeconds
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
    ZERO = auto()
    ONE = auto()

    @classmethod
    def from_val(cls, val: int) -> Self:
        return {
            0: BitValue.ZERO,
            1: BitValue.ONE,
        }[val]

    def as_val(self) -> int:
        return {
            BitValue.ZERO: 0,
            BitValue.ONE: 1,
        }[self]

    def inverted(self) -> Self:
        return {
            BitValue.ZERO: BitValue.ONE,
            BitValue.ONE: BitValue.ZERO,
        }[self]

    def __eq__(self, other) -> bool:
        if not isinstance(other, BitValue):
            return False
        return self.as_val() == other.as_val()

    def __hash__(self) -> int:
        return hash(self.value)


class NavigationBitPseudosymbol(Enum):
    MINUS_ONE = auto()
    ONE = auto()

    @classmethod
    def from_val(cls, val: int) -> Self:
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

    def process_samples(self, receiver_timestamp: ReceiverTimestampSeconds, samples: AntennaSamplesSpanningOneMs) -> NavigationBitPseudosymbol:
        params = self.tracking_params

        # Generate Doppler-shifted and phase-shifted carrier wave
        time_domain = (np.arange(SAMPLES_PER_PRN_TRANSMISSION) / SAMPLES_PER_SECOND) + receiver_timestamp
        doppler_shift_carrier = np.exp(
            -1j * ((2 * np.pi * params.current_doppler_shift * time_domain) + params.current_carrier_wave_phase_shift)
        )
        doppler_shifted_samples = samples * doppler_shift_carrier

        # Correlate early, prompt, and late phase versions of the PRN
        unslid_prn = params.satellite.prn_as_complex
        prompt_prn = np.roll(unslid_prn, params.current_prn_code_phase_shift)

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
