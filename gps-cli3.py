from typing import Tuple

from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
import numpy as np

from dataclasses import dataclass

from scipy.signal import butter, filtfilt, find_peaks, resample, resample_poly
from tqdm import tqdm

from gps.gps_ca_prn_codes import generate_replica_prn_signals, GpsSatelliteId
from gps.radio_input import INPUT_SOURCES, get_samples_from_radio_input_source
from gps.config import PRN_CORRELATION_CYCLE_COUNT, DOPPLER_SHIFT_FREQUENCY_LOWER_BOUND, \
    DOPPLER_SHIFT_FREQUENCY_UPPER_BOUND, DOPPLER_SHIFT_SEARCH_INTERVAL, PRN_CORRELATION_MAGNITUDE_THRESHOLD
from gps.constants import SAMPLES_PER_SECOND, SAMPLES_PER_PRN_TRANSMISSION, PRN_REPETITIONS_PER_SECOND, PRN_CHIP_COUNT
from gps.satellite import GpsSatellite, ALL_SATELLITE_IDS
from gps.utils import chunks, round_to_previous_multiple_of


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
    sample_rate = input_source.sdr_sample_rate
    antenna_data = Samples(get_samples_from_radio_input_source(input_source, sample_rate))
    return satellites_by_id, antenna_data


def frequency_domain_correlation(segment, prn_replica):
    R = np.fft.fft(segment)
    L = np.fft.fft(prn_replica)
    product = R * np.conj(L)
    return np.fft.ifft(product)


def get_ratio_of_outlier_signal(a: np.ndarray) -> float:
    """Get the ratio of the strongest component in the signal to the signal mean.
    In other words, how strongly does a signal pop out?
    """
    new = np.copy(a)
    mean = np.mean(a)
    max_value = np.max(a)
    max_peaks = np.argwhere(a >= max_value)
    new[max_peaks] = mean
    snr_ratio = max_value / np.mean(new)
    return snr_ratio


def threshold_detection(data: np.ndarray, k: int = 5) -> np.ndarray:
    """Detects peaks in the correlation results that exceed a threshold
    - k: Multiplier for the standard deviation to set the threshold.
    """
    threshold = np.mean(data) + (k * np.std(data))
    peak_indices = np.where(data > threshold)[0]
    return peak_indices


def estimate_SNR(correlation_result):
    """
    Estimates the Signal-to-Noise Ratio (SNR) of the correlation result.

    Parameters:
    - correlation_result: The absolute values of the correlation result.

    Returns:
    - SNR in decibels (dB).
    """

    # 1. Identify the peak value (signal power)
    signal_power = np.max(correlation_result)

    # 2. Estimate the noise power
    # Exclude the peak value for noise estimation
    without_peak = np.delete(correlation_result, np.argmax(correlation_result))
    noise_power = np.std(without_peak)

    # Avoid division by zero
    if noise_power == 0:
        return float('inf')  # Infinite SNR

    # 3. Compute the SNR
    snr_linear = signal_power / noise_power
    snr_dB = 10 * np.log10(snr_linear)

    return snr_dB


def test_acquire():
    satellites_by_id, antenna_data = get_satellites_info_and_antenna_samples()
    for sv_id in ALL_SATELLITE_IDS:
        satellite_info = satellites_by_id[sv_id]
        prn = satellite_info.prn_as_complex

        integration_period_ms = 20
        prn_chip_rate = 1.023e6
        doppler_shift_to_integrated_coherent_correlation = {}
        for doppler_shift in range(
            -6000,
            6000,
            500,
        ):
            #print(f'Trying Doppler shift {doppler_shift}')
            # Calculate the PRN length, accounting for this Doppler shift
            shifted_prn_chip_rate = prn_chip_rate + doppler_shift
            prn_resampling_ratio = shifted_prn_chip_rate / prn_chip_rate
            shifted_prn_sample_count = int(SAMPLES_PER_PRN_TRANSMISSION * prn_resampling_ratio)
            #resampled_prn = resample(prn, shifted_prn_sample_count)
            resampled_prn = resample_poly(prn, int(shifted_prn_chip_rate), int(prn_chip_rate))
            resampled_prn = np.array([complex(1, 0) if x.real >= 0.5 else complex(-1, 0) for x in resampled_prn][:shifted_prn_sample_count])
            #print(f'Resampled PRN sample count: {shifted_prn_sample_count}')

            samples_in_window = integration_period_ms * shifted_prn_sample_count
            antenna_data_snippet = antenna_data.data[:samples_in_window]
            integration_time_domain = np.arange(integration_period_ms * shifted_prn_sample_count) / SAMPLES_PER_SECOND
            doppler_shift_carrier = np.exp(-1j * 2 * np.pi * doppler_shift * integration_time_domain)
            doppler_shifted_antenna_data_snippet = antenna_data_snippet * doppler_shift_carrier

            coherent_integration_result = np.zeros(shifted_prn_sample_count, dtype=complex)
            for i, chunk_that_may_contain_one_prn in enumerate(chunks(doppler_shifted_antenna_data_snippet, shifted_prn_sample_count)):
                correlation_result = frequency_domain_correlation(chunk_that_may_contain_one_prn, resampled_prn)
                coherent_integration_result += correlation_result
            doppler_shift_to_integrated_coherent_correlation[doppler_shift] = np.abs(coherent_integration_result)

        # Find the best integration result
        best_doppler_shift = max(doppler_shift_to_integrated_coherent_correlation, key=lambda key: np.max(doppler_shift_to_integrated_coherent_correlation[key]))
        best_integrated_coherent_correlation = doppler_shift_to_integrated_coherent_correlation[best_doppler_shift]
        plt.plot(best_integrated_coherent_correlation)
        peaks = threshold_detection(best_integrated_coherent_correlation)
        plt.plot(peaks, best_integrated_coherent_correlation[peaks], 'rp', markersize=10)
        print(f'Best Doppler shift for SV {sv_id.id}: {best_doppler_shift}')
        snr = estimate_SNR(best_integrated_coherent_correlation)
        print(f'\t*** SNR {snr:.2f}')
        plt.title(f'SV {sv_id.id} best Doppler: {best_doppler_shift:.2f}')
        plt.show()




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
        #time_domain = np.arange(estimated_samples_per_prn*1000) / SAMPLES_PER_SECOND
        #time_domain = np.arange(estimated_samples_per_prn)
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

        #prn_phase += prn_step + gamma * code_delay_discriminator
        samples_per_prn_chip = 2
        prn_chip_shift += (samples_per_prn_chip + (gamma * code_delay_discriminator))
        prn_chip_shift %= len(satellite_prn)

        #carrier_replica = generate_prn_rolled_by()
        print(f'PRN chip shift {prn_chip_shift}, doppler={angular_doppler_shift:.2f}, carrier phase={carrier_phase:.2f}')
        plt.plot(data_i_with_carrier_removed * prn_prompt)
        plt.plot(data_q_with_carrier_removed * prn_prompt)
        plt.show()


    fig, axes = plt.subplots(nrows=2)
    fig.subplots_adjust(bottom=0.4)


if __name__ == '__main__':
    test_acquire()
    #main()
