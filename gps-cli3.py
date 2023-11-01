from collections import defaultdict
from enum import Enum, auto
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from dataclasses import dataclass

from scipy.signal import resample_poly

from gps.gps_ca_prn_codes import generate_replica_prn_signals, GpsSatelliteId
from gps.radio_input import INPUT_SOURCES, get_samples_from_radio_input_source
from gps.constants import SAMPLES_PER_SECOND, SAMPLES_PER_PRN_TRANSMISSION
from gps.satellite import GpsSatellite, ALL_SATELLITE_IDS
from gps.utils import chunks


class Samples:
    def __init__(self, data: np.ndarray) -> None:
        self.data = data
        self.i = 0

    def peek_samples(self, n: int) -> np.ndarray:
        return self.data[self.i:self.i + n]

    def get_samples(self, n: int) -> np.ndarray:
        out = self.peek_samples(n)
        self.i += n
        return out


def get_satellites_info_and_antenna_samples() -> Tuple[dict[GpsSatelliteId, GpsSatellite], Samples]:
    satellites_to_replica_prn_signals = generate_replica_prn_signals()
    satellites_by_id = {
        satellite_id: GpsSatellite(
            satellite_id=satellite_id,
            prn_code=code
        )
        for satellite_id, code in satellites_to_replica_prn_signals.items()
    }
    input_source = INPUT_SOURCES[6]
    print(input_source.path.as_posix())
    sample_rate = input_source.sdr_sample_rate
    antenna_data = Samples(get_samples_from_radio_input_source(input_source, sample_rate))
    return satellites_by_id, antenna_data


def frequency_domain_correlation(segment, prn_replica):
    R = np.fft.fft(segment)
    L = np.fft.fft(prn_replica)
    product = R * np.conj(L)
    return np.fft.ifft(product)


_CorrelationProfile = np.ndarray
_CorrelationStrength = float

_DopplerShiftHz = float
_CarrierWavePhaseInRadians = float
_PrnCodePhaseInSamples = float


class ResampledPrnProvider:
    def __init__(self, satellites: dict[GpsSatelliteId, GpsSatellite]):
        self.satellites = satellites
        self.sv_id_to_resampled_prn_cache = defaultdict(dict)

    def get_resampled_prn(self, sv_id: GpsSatelliteId, sample_count: int) -> np.ndarray:
        if sample_count not in self.sv_id_to_resampled_prn_cache[sv_id]:
            prn = self.satellites[sv_id].prn_as_complex
            resampled_prn = resample_poly(prn, sample_count, len(prn))
            resampled_prn = np.array([complex(1, 0) if x.real >= 0.5 else complex(-1, 0) for x in resampled_prn][:sample_count])
            self.sv_id_to_resampled_prn_cache[sv_id][sample_count] = resampled_prn
        return self.sv_id_to_resampled_prn_cache[sv_id][sample_count]


@dataclass
class BestNonCoherentCorrelationProfile:
    doppler_shift: _DopplerShiftHz
    non_coherent_correlation_profile: _CorrelationProfile
    # Just convenience accessors that can be derived from the correlation profile
    sample_offset_of_correlation_peak: int
    correlation_strength: float


@dataclass
class DetectedSatelliteInfo:
    satellite_id: GpsSatelliteId
    doppler_shift: _DopplerShiftHz
    carrier_wave_phase_shift: _CarrierWavePhaseInRadians
    prn_phase_shift: _PrnCodePhaseInSamples


class IntegrationType(Enum):
    Coherent = auto()
    NonCoherent = auto()


def integrate_correlation_with_doppler_shifted_prn(
    integration_type: IntegrationType,
    antenna_data: Samples,
    sv_id: GpsSatelliteId,
    doppler_shift: _DopplerShiftHz,
    prn_provider: ResampledPrnProvider,
) -> _CorrelationProfile:
    integration_period_ms = 20
    prn_chip_rate = 1.023e6

    # Calculate the PRN length, accounting for this Doppler shift
    shifted_prn_chip_rate = prn_chip_rate + doppler_shift
    prn_resampling_ratio = shifted_prn_chip_rate / prn_chip_rate
    shifted_prn_sample_count = int(SAMPLES_PER_PRN_TRANSMISSION * prn_resampling_ratio)
    resampled_prn = prn_provider.get_resampled_prn(sv_id, shifted_prn_sample_count)

    samples_in_window = integration_period_ms * shifted_prn_sample_count
    antenna_data_snippet = antenna_data.data[:samples_in_window]
    integration_time_domain = np.arange(integration_period_ms * shifted_prn_sample_count) / SAMPLES_PER_SECOND
    doppler_shift_carrier = np.exp(-1j * 2 * np.pi * doppler_shift * integration_time_domain)
    doppler_shifted_antenna_data_snippet = antenna_data_snippet * doppler_shift_carrier

    correlation_data_type = {
        IntegrationType.Coherent: complex,
        IntegrationType.NonCoherent: np.float64,
    }[integration_type]
    coherent_integration_result = np.zeros(shifted_prn_sample_count, dtype=correlation_data_type)
    for i, chunk_that_may_contain_one_prn in enumerate(chunks(doppler_shifted_antenna_data_snippet, shifted_prn_sample_count)):
        correlation_result = frequency_domain_correlation(chunk_that_may_contain_one_prn, resampled_prn)

        if integration_type == IntegrationType.Coherent:
            coherent_integration_result += correlation_result
        elif integration_type == IntegrationType.NonCoherent:
            coherent_integration_result += np.abs(correlation_result)
        else:
            raise ValueError('Unexpected integration type')

    return coherent_integration_result


def compute_best_doppler_shift_estimation(
    center: float,
    spread: float,
    antenna_data: Samples,
    sv_id: GpsSatelliteId,
    prn_provider: ResampledPrnProvider,
) -> BestNonCoherentCorrelationProfile:
    doppler_shift_to_correlation_profile = {}
    for doppler_shift in range(
        int(center - spread),
        int(center + spread),
        int(spread / 10),
    ):
        correlation_profile = integrate_correlation_with_doppler_shifted_prn(
            # Always use non-coherent integration when searching for the best Doppler peaks.
            # This will give us the strongest SNR possible to detect peaks.
            IntegrationType.NonCoherent,
            antenna_data,
            sv_id,
            doppler_shift,
            prn_provider,
        )
        doppler_shift_to_correlation_profile[doppler_shift] = correlation_profile

    # Find the best correlation result
    best_doppler_shift = max(doppler_shift_to_correlation_profile, key=lambda key: np.max(doppler_shift_to_correlation_profile[key]))
    best_correlation_profile = doppler_shift_to_correlation_profile[best_doppler_shift]
    sample_offset_of_correlation_peak = np.argmax(best_correlation_profile)
    correlation_strength = best_correlation_profile[sample_offset_of_correlation_peak]
    return BestNonCoherentCorrelationProfile(
        doppler_shift=best_doppler_shift,
        non_coherent_correlation_profile=best_correlation_profile,
        sample_offset_of_correlation_peak=int(sample_offset_of_correlation_peak),
        correlation_strength=correlation_strength,
    )


def test_acquire():
    satellites_by_id, antenna_data = get_satellites_info_and_antenna_samples()
    prn_provider = ResampledPrnProvider(satellites_by_id)
    detected_satellites_by_id = {}
    for sv_id in ALL_SATELLITE_IDS:
        # Detection
        doppler_frequency_estimation_spread = 5000
        # This must be 10 as the search factor divides the spread by 10
        best_non_coherent_correlation_profile_across_all_search_space = None
        doppler_estimation = 0
        while doppler_frequency_estimation_spread >= 10:
            best_non_coherent_correlation_profile_in_this_search_space = compute_best_doppler_shift_estimation(
                doppler_estimation,
                doppler_frequency_estimation_spread,
                antenna_data,
                sv_id,
                prn_provider,
            )
            doppler_frequency_estimation_spread /= 2
            if (
                # Base case
                not best_non_coherent_correlation_profile_across_all_search_space
                # Found a better candidate
                or best_non_coherent_correlation_profile_in_this_search_space.correlation_strength > best_non_coherent_correlation_profile_across_all_search_space.correlation_strength
            ):
                best_non_coherent_correlation_profile_across_all_search_space = best_non_coherent_correlation_profile_in_this_search_space
                doppler_estimation = best_non_coherent_correlation_profile_across_all_search_space.doppler_shift
                print(
                    f'Found a better candidate Doppler for SV({sv_id}): '
                    f'(Found in [{doppler_estimation - doppler_frequency_estimation_spread:.2f} | '
                    f'{doppler_estimation:.2f} | {doppler_estimation + doppler_frequency_estimation_spread:.2f}], '
                    f'Strength: '
                    f'{best_non_coherent_correlation_profile_across_all_search_space.correlation_strength:.2f}'
                )

        print(f'Best correlation for SV({sv_id}) at Doppler {doppler_estimation:.2f} corr {best_non_coherent_correlation_profile_across_all_search_space.correlation_strength:.2f}')
        if best_non_coherent_correlation_profile_across_all_search_space.correlation_strength > 37.0:
            # Detection, Doppler frequency over threshold
            print('Non-coherent correlation strength was above threshold, continuing to coherent integration...')
            # Now, compute the coherent correlation so that we can determine (an estimate) of the phase of the carrier wave
            best_doppler_shift = best_non_coherent_correlation_profile_across_all_search_space.doppler_shift
            coherent_correlation_profile = integrate_correlation_with_doppler_shifted_prn(
                IntegrationType.Coherent,
                antenna_data,
                sv_id,
                best_doppler_shift,
                prn_provider,
            )

            # Rely on the correlation peak index that comes from non-coherent integration, since it'll be stronger and
            # therefore has less chance of being overridden by noise. Coherent integration may have selected a noise
            # peak.
            sample_offset_of_correlation_peak = best_non_coherent_correlation_profile_across_all_search_space.sample_offset_of_correlation_peak
            carrier_wave_phase_shift = np.angle(coherent_correlation_profile[sample_offset_of_correlation_peak])
            # The sample offset where the best correlation occurs gives us (an estimate) of the phase shift of the PRN
            prn_phase_shift = sample_offset_of_correlation_peak
            print(f'Detected SV({sv_id}):')
            print(f'\tDoppler {best_doppler_shift:.2f}')
            print(f'\tCarrier phase {carrier_wave_phase_shift}')
            print(f'\tPRN phase {prn_phase_shift:.2f}')
            detected_satellites_by_id[sv_id] = DetectedSatelliteInfo(
                satellite_id=sv_id,
                doppler_shift=best_doppler_shift,
                carrier_wave_phase_shift=carrier_wave_phase_shift,
                prn_phase_shift=prn_phase_shift,
            )

    return detected_satellites_by_id
    # Detected sats with coherent integration:
    # SV(GpsSatelliteId(id=5)) at Doppler 0.00 phase(samp) 1023.00 corr 116.46+0.00j
    # SV(GpsSatelliteId(id=23)) at Doppler 3142.00 phase(samp) 1512.00 corr 37.18+0.00j
    # SV(GpsSatelliteId(id=26)) at Doppler -6446.00 phase(samp) 1429.00 corr 37.22+0.00j
    # SV(GpsSatelliteId(id=30)) at Doppler 0.00 phase(samp) 708.00 corr 40.09+0.00j
    # Detected sats with non-coherent integration:
    # SV(GpsSatelliteId(id=5)) at Doppler -369.00 corr 24.26


def main():
    satellites_to_replica_prn_signals = generate_replica_prn_signals()
    satellites_by_id = {
        satellite_id: GpsSatellite(
            satellite_id=satellite_id,
            prn_code=code
        )
        for satellite_id, code in satellites_to_replica_prn_signals.items()
    }
    input_source = INPUT_SOURCES[6]
    sample_rate = input_source.sdr_sample_rate
    antenna_data = Samples(get_samples_from_radio_input_source(input_source, sample_rate))

    satellite_prn = satellites_by_id[GpsSatelliteId(5)].prn_as_complex
    satellite_prn_fft = np.fft.fft(satellite_prn)
    acquired_doppler_shift = 362.1
    acquired_prn_phase_shift = 1022
    rolled_prn = np.roll(satellite_prn, -acquired_prn_phase_shift)

    integration_period_ms = 10
    integration_period_sample_count = int(SAMPLES_PER_SECOND / 1000 * integration_period_ms)
    print(f'Integration period (in samples): {integration_period_sample_count}')

    frequency_shift = acquired_doppler_shift
    carrier_phase = 0
    prn_chip_shift = -acquired_prn_phase_shift
    angular_doppler_shift = 0
    alpha = 0.001
    beta = 0.001
    gamma = 0.7

    estimated_samples_per_prn = SAMPLES_PER_PRN_TRANSMISSION
    while True:
        samples_that_should_contain_one_prn_transmission = antenna_data.get_samples(estimated_samples_per_prn)
        time_domain = np.arange(SAMPLES_PER_SECOND)[:estimated_samples_per_prn]
        print(len(time_domain))

        # Generate carrier wave replica
        print(f'Frequency shift {frequency_shift:.2f} Carrier phase {carrier_phase:.2f}')
        carrier_i = np.cos((2. * np.pi * time_domain * frequency_shift) + carrier_phase)
        carrier_q = np.sin((2. * np.pi * time_domain * frequency_shift) + carrier_phase)
        carrier = np.array([complex(i, q) for i, q in zip(carrier_i, carrier_q)])

        mixed_samples = samples_that_should_contain_one_prn_transmission * carrier
        data_i_with_carrier_removed = mixed_samples.real
        data_q_with_carrier_removed = mixed_samples.imag

        # Generate I arms of early, prompt, late PRN correlations
        #prn_early = np.tile(np.roll(satellite_prn, int(prn_chip_shift - 1)).real, integration_period_ms)
        #prn_prompt = np.tile(np.roll(satellite_prn, int(prn_chip_shift)).real, integration_period_ms)
        #prn_late = np.tile(np.roll(satellite_prn, int(prn_chip_shift + 1)).real, integration_period_ms)
        # TODO(PT): We do need to be able to resample the PRN to the correct length!
        prn_early = np.roll(satellite_prn, int(prn_chip_shift - 1)).real
        prn_prompt = np.roll(satellite_prn, int(prn_chip_shift)).real
        prn_late = np.roll(satellite_prn, int(prn_chip_shift + 1)).real

        i_prn_early = np.sum(data_i_with_carrier_removed * prn_early)
        i_prn_prompt = np.sum(data_i_with_carrier_removed * prn_prompt)
        i_prn_late = np.sum(data_i_with_carrier_removed * prn_late)

        q_prn_early = np.sum(data_q_with_carrier_removed * prn_early)
        q_prn_prompt = np.sum(data_q_with_carrier_removed * prn_prompt)
        q_prn_late = np.sum(data_q_with_carrier_removed * prn_late)

        early_power = (i_prn_early**2) + (q_prn_early**2)
        late_power = (i_prn_late**2) + (q_prn_late**2)
        code_delay_discriminator = (early_power - late_power) / (early_power + late_power)

        if False:
            plt.plot(i_prn_early)
            plt.plot(i_prn_prompt)
            plt.plot(i_prn_late)
            plt.show()

        # Compute Costas loop error from prompt values
        error = np.arctan(q_prn_prompt / i_prn_prompt)
        angular_doppler_shift = beta * error
        carrier_phase += (2. * np.pi * frequency_shift / SAMPLES_PER_SECOND) + angular_doppler_shift + alpha * error
        print(carrier_phase)
        carrier_phase %= 2. * np.pi

        samples_per_prn_chip = 2
        prn_chip_shift += (samples_per_prn_chip + (gamma * code_delay_discriminator))
        prn_chip_shift %= len(satellite_prn)

        print(f'PRN chip shift {prn_chip_shift}, doppler={angular_doppler_shift:.2f}, carrier phase={carrier_phase:.2f}')
        plt.plot(data_i_with_carrier_removed * prn_prompt)
        plt.plot(data_q_with_carrier_removed * prn_prompt)
        plt.show()

    fig, axes = plt.subplots(nrows=2)
    fig.subplots_adjust(bottom=0.4)


if __name__ == '__main__':
    test_acquire()
    #main()
