import logging
from dataclasses import dataclass

import math
import numpy as np

from gypsum.constants import SAMPLES_PER_PRN_TRANSMISSION
from gypsum.constants import SAMPLES_PER_SECOND
from gypsum.satellite import GpsSatellite
from gypsum.utils import AntennaSamplesSpanningOneMs
from gypsum.utils import CarrierWavePhaseInRadians
from gypsum.utils import DopplerShiftHz
from gypsum.utils import PrnCodePhaseInSamples
from gypsum.utils import frequency_domain_correlation

_logger = logging.getLogger(__name__)


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
    def __init__(self, tracking_params: GpsSatelliteTrackingParameters) -> None:
        self.tracking_params = tracking_params

    def process_samples(self, samples: AntennaSamplesSpanningOneMs, sample_index: int) -> None:
        params = self.tracking_params

        loop_bandwidth = 2.046 / 1000
        # Common choice for zeta, considered optimal
        damping_factor = math.sqrt(2) / 2.0
        # Natural frequency
        natural_freq = loop_bandwidth / (damping_factor * (1 + damping_factor**2) ** 0.5)
        # This represents the gain of *instantaneous* error correction,
        # which applies to the estimate of the carrier wave phase.
        # Also called 'alpha'.
        loop_gain_phase = (4 * damping_factor * natural_freq) / (
            1 + ((2 * damping_factor * natural_freq) + (natural_freq**2))
        )
        # This represents the *integrated* error correction,
        # which applies to the estimate of the Doppler shifted frequency.
        # Also called 'beta'.
        loop_gain_freq = (4 * (natural_freq**2)) / (1 + ((2 * damping_factor * natural_freq) + (natural_freq**2)))

        # Generate Doppler-shifted and phase-shifted carrier wave
        time_domain = (np.arange(SAMPLES_PER_PRN_TRANSMISSION) / SAMPLES_PER_SECOND) + (
            sample_index / SAMPLES_PER_SECOND
        )
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
            centered_non_coherent_prompt_peak_offset = (
                non_coherent_prompt_peak_offset - SAMPLES_PER_PRN_TRANSMISSION + 1
            )

        logging.info(
            f"Peak offset {non_coherent_prompt_peak_offset}, centered offset {centered_non_coherent_prompt_peak_offset}"
        )
        if centered_non_coherent_prompt_peak_offset > 0:
            params.current_prn_code_phase_shift += centered_non_coherent_prompt_peak_offset
        else:
            params.current_prn_code_phase_shift -= centered_non_coherent_prompt_peak_offset

        # Finally, ensure we're always sliding within one PRN transmission
        params.current_prn_code_phase_shift = int(params.current_prn_code_phase_shift) % SAMPLES_PER_PRN_TRANSMISSION

        coherent_prompt_prn_correlation_peak = coherent_prompt_correlation[non_coherent_prompt_peak_offset]

        I = np.real(coherent_prompt_prn_correlation_peak)
        Q = np.imag(coherent_prompt_prn_correlation_peak)
        carrier_wave_phase_error = I * Q

        params.current_doppler_shift += loop_gain_freq * carrier_wave_phase_error
        params.current_carrier_wave_phase_shift += loop_gain_phase * carrier_wave_phase_error
        params.current_carrier_wave_phase_shift %= math.tau

        navigation_bit_pseudosymbol_value = int(np.sign(coherent_prompt_prn_correlation_peak))
        params.navigation_bit_pseudosymbols.append(navigation_bit_pseudosymbol_value)

        logging.info(f"Doppler shift {params.current_doppler_shift:.2f}")
        logging.info(f"Carrier phase {params.current_carrier_wave_phase_shift:.8f}")
        logging.info(f"Code phase {params.current_prn_code_phase_shift}")

        params.doppler_shifts.append(params.current_doppler_shift)
        params.carrier_wave_phases.append(params.current_carrier_wave_phase_shift)
        params.carrier_wave_phase_errors.append(carrier_wave_phase_error)
