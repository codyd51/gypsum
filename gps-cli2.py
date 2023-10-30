from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from tqdm import tqdm

from gps.gps_ca_prn_codes import generate_replica_prn_signals, GpsSatelliteId
from gps.radio_input import INPUT_SOURCES, get_samples_from_radio_input_source
from gps.config import PRN_CORRELATION_CYCLE_COUNT, DOPPLER_SHIFT_FREQUENCY_LOWER_BOUND, \
    DOPPLER_SHIFT_FREQUENCY_UPPER_BOUND, DOPPLER_SHIFT_SEARCH_INTERVAL, PRN_CORRELATION_MAGNITUDE_THRESHOLD
from gps.constants import SAMPLES_PER_SECOND, SAMPLES_PER_PRN_TRANSMISSION, PRN_REPETITIONS_PER_SECOND, PRN_CHIP_COUNT
from gps.satellite import GpsSatellite, ALL_SATELLITE_IDS
from gps.utils import chunks, round_to_previous_multiple_of

# PT: Newtype for semantic clarity
Frequency = int


def show_and_quit(x, array):
    ax1 = plt.subplot(212)
    ax1.margins(0.02, 0.2)
    ax1.use_sticky_edges = False

    ax1.plot(x, array)
    plt.show()


# No need to generate the same waves over and over, so throw a cache in front
def generate_cosine_with_frequency(num_samples: int, frequency: float) -> list[complex]:
    time_domain = np.arange(num_samples) / SAMPLES_PER_SECOND
    i_components = np.cos(2. * np.pi * time_domain * frequency)
    q_components = np.sin(2. * np.pi * time_domain * frequency)
    cosine = [complex(i, q) for i, q in zip(i_components, q_components)]
    return cosine


@dataclass
class DetectedSatelliteInfo:
    satellite_id: GpsSatelliteId
    doppler_frequency_shift: int
    time_offset: float
    chip_offset: float
    sample_offset: int


class GpsSatelliteDetector:
    def __init__(self):
        # Generate PRN signals for each satellite
        satellites_to_replica_prn_signals = generate_replica_prn_signals()
        self.satellites_by_id = {
            satellite_id: GpsSatellite(
                satellite_id=satellite_id,
                prn_code=code
            )
            for satellite_id, code in satellites_to_replica_prn_signals.items()
        }
        self._cached_doppler_shifted_carriers = {}

    def get_doppler_shifted_carrier(self, chunk_length: int, doppler_shift: float) -> np.ndarray:
        if (chunk_length, doppler_shift) not in self._cached_doppler_shifted_carriers:
            doppler_shifted_carrier = generate_cosine_with_frequency(chunk_length, doppler_shift)
            self._cached_doppler_shifted_carriers[(chunk_length, doppler_shift)] = doppler_shifted_carrier
        return self._cached_doppler_shifted_carriers[(chunk_length, doppler_shift)]

    def detect_satellites_in_data(self, antenna_data: np.ndarray) -> dict[GpsSatelliteId, DetectedSatelliteInfo]:
        # PT: Currently, this expects the provided antenna data to represent exactly one second of sampling
        start_time = 0
        end_time = len(antenna_data) / SAMPLES_PER_SECOND
        time_domain = np.arange(start_time, end_time, 1 / (float(SAMPLES_PER_SECOND)))
        # We're going to read PRN_CORRELATION_CYCLE_COUNT repetitions of the PRN
        correlation_bucket_sample_count = int(PRN_CORRELATION_CYCLE_COUNT * SAMPLES_PER_PRN_TRANSMISSION)
        print(f'Correlation bucket sample count: {correlation_bucket_sample_count}')
        time_domain_for_correlation_bucket = time_domain[:correlation_bucket_sample_count]

        # Throw away extra signal data that doesn't fit in a modulo of our window size
        trimmed_antenna_data = antenna_data[
                               :round_to_previous_multiple_of(len(antenna_data), correlation_bucket_sample_count)]

        # Precompute the Doppler-shifted carrier waves that we'll use to try to demodulate the antenna data, across
        # our search radius.
        print(f'Precomputing Doppler-shifted carrier waves...')
        doppler_shifted_carrier_waves = {
            doppler_shift: generate_cosine_with_frequency(correlation_bucket_sample_count, doppler_shift)
            for doppler_shift in range(
                DOPPLER_SHIFT_FREQUENCY_LOWER_BOUND,
                DOPPLER_SHIFT_FREQUENCY_UPPER_BOUND,
                DOPPLER_SHIFT_SEARCH_INTERVAL,
            )
        }
        print(f'Finished precomputing Doppler-shifted carrier waves.')

        detected_satellites_by_id = {}
        antenna_data_chunks = list(
            chunks(trimmed_antenna_data, correlation_bucket_sample_count, step=SAMPLES_PER_SECOND // 4))
        for i, antenna_data_chunk in tqdm(list(enumerate(antenna_data_chunks))):
            for satellite_id in ALL_SATELLITE_IDS:
                # Only necessary to look for a satellite if we haven't already detected it
                if satellite_id in detected_satellites_by_id:
                    # print(f'Will not search for satellite {satellite_id.id} again because we\'ve already identified it')
                    continue

                if detected_satellite := self._detect_satellite_in_correlation_bucket(
                        satellite_id,
                        correlation_bucket_sample_count,
                        time_domain_for_correlation_bucket,
                        antenna_data_chunk,
                        doppler_shifted_carrier_waves,
                ):
                    detected_satellites_by_id[satellite_id] = detected_satellite

            # Check whether we've detected enough satellites to carry out a position fix
            if len(detected_satellites_by_id) >= 4:
                print(f"We've detected enough satellites to carry out a position fix!")
                break
            else:
                # print(f"Finished searching this correlation bucket, but we've only detected {len(detected_satellites_by_id)} satellites. Moving on to the next bucket...")
                pass
        return detected_satellites_by_id

    @staticmethod
    def _compute_correlation(
            antenna_data: np.ndarray,
            doppler_shifted_carrier_wave: np.ndarray,
            prn_fft: np.ndarray,
    ) -> np.ndarray:
        correlation_sample_count = len(antenna_data)
        antenna_data_multiplied_with_doppler_shifted_carrier = antenna_data * doppler_shifted_carrier_wave
        fft_of_demodulated_doppler_shifted_signal = np.fft.fft(antenna_data_multiplied_with_doppler_shifted_carrier)

        mult = prn_fft * np.conjugate(fft_of_demodulated_doppler_shifted_signal)
        mult_ifft = np.fft.ifft(mult)
        return mult_ifft
        normalized_correlation = mult_ifft / correlation_sample_count
        return normalized_correlation

        # Retain the sign of the correlation result
        # correlation_sign = np.sign(normalized_correlation.real) + 1j * np.sign(normalized_correlation.imag)

        # Compute the magnitude and phase of the correlation result
        # correlation_magnitude = np.abs(correlation_sign)
        # correlation_phase = np.angle(correlation_sign)
        # print(f'Correlation magnitude: {correlation_magnitude}')
        # print(f'Correlation phase: {correlation_phase}')

        # plt.plot(correlation_magnitude)
        # plt.show()
        # plt.plot(correlation_phase)
        # plt.show()

        return normalized_correlation
        correlation = np.abs(normalized_correlation)
        return correlation

    @staticmethod
    def get_ratio_of_outlier_signal(a: np.ndarray) -> (float, np.ndarray):
        """Get the ratio of the strongest component in the signal to the signal mean.
        In other words, how strongly does a signal pop out?
        """
        new = np.copy(a)
        mean = np.mean(a)
        max_value = np.max(a)
        max_peaks = np.argwhere(a >= max_value)
        new[max_peaks] = mean
        snr_ratio = max_value / np.mean(new)
        return snr_ratio, max_peaks

    def _detect_satellite_in_correlation_bucket(
            self,
            satellite_id: GpsSatelliteId,
            correlation_bucket_sample_count: int,
            time_domain: np.ndarray,
            antenna_data_chunk: np.ndarray,
            # These are precomputed and passed in as an optimization
            doppler_shifted_carrier_waves: dict[Frequency, np.ndarray],
    ) -> DetectedSatelliteInfo | None:
        # Sanity check
        if len(time_domain) != len(antenna_data_chunk) != correlation_bucket_sample_count:
            raise ValueError(f'Expected the bucketed antenna data and time domain to be the same length')

        # TODO(PT): We can binary search the Doppler search space? Or not, because we won't find "enough" signal to
        # know where to look.
        # Another strategy for cutting up the search space:
        # We can sample the search space in 10 buckets, then search further in the one that gives the best match?
        satellite_prn_fft = self.satellites_by_id[satellite_id].fft_of_prn_of_length(correlation_bucket_sample_count)
        for doppler_shift in range(
                DOPPLER_SHIFT_FREQUENCY_LOWER_BOUND,
                DOPPLER_SHIFT_FREQUENCY_UPPER_BOUND,
                DOPPLER_SHIFT_SEARCH_INTERVAL
        ):
            correlation = self._compute_correlation(
                antenna_data_chunk,
                doppler_shifted_carrier_waves[doppler_shift],
                satellite_prn_fft,
            )
            # print(f'Sat {satellite_id.id} shift {doppler_shift} corr min {np.min(correlation)} corr max {np.max(correlation)}')

            correlation_abs = np.abs(correlation)
            snr_ratio, correlation_peaks = GpsSatelliteDetector.get_ratio_of_outlier_signal(correlation_abs)
            indexes_of_peaks_above_prn_correlation_threshold = list(sorted(correlation_peaks))
            # TODO(PT): Instead of immediately returning, we should hold out to find the best correlation across the search space
            if snr_ratio >= 5.0:
                # We're matching against many PRN cycles to increase our correlation strength.
                # We now want to figure out the time slide (and therefore distance) for the transmitter.
                # Therefore, we only care about the distance to the first correlation peak (i.e. the phase offset)
                sample_offset_where_we_started_receiving_prn = indexes_of_peaks_above_prn_correlation_threshold[0][0]
                #  Divide by Nyquist frequency
                chip_index_where_we_started_receiving_prn = sample_offset_where_we_started_receiving_prn / 2
                # Convert to time
                time_per_prn_chip = ((1 / PRN_REPETITIONS_PER_SECOND) / PRN_CHIP_COUNT)
                chip_offset_from_satellite = PRN_CHIP_COUNT - chip_index_where_we_started_receiving_prn
                # TODO(PT): Maybe the sign varies depending on whether the Doppler shift is positive or negative?
                time_offset = chip_offset_from_satellite * time_per_prn_chip
                # print(f'We are {chip_offset_from_satellite} chips ahead of satellite {satellite_id}. This represents a time delay of {time_offset}')
                print(
                    f'*** Identified satellite {satellite_id} at doppler shift {doppler_shift}, correlation magnitude of {correlation[sample_offset_where_we_started_receiving_prn]} at {sample_offset_where_we_started_receiving_prn}, time offset of {time_offset}, chip offset of {chip_offset_from_satellite}')
                print(f'*** SNR: {snr_ratio}')
                plt.plot(correlation.real)
                plt.show()

                # PT: It seems as though we don't yet have enough information to determine
                # whether the satellite's clock is ahead of or behind our own (which makes sense).
                # All we can say for now is that the received PRN is out of phase with our PRN
                # by some number of chips/some time delay (which we can choose to be either positive or negative).
                # It seems like the next step now is to use the delay to decode the navigation message, and figure
                # out later our time differential.
                return DetectedSatelliteInfo(
                    satellite_id=satellite_id,
                    doppler_frequency_shift=doppler_shift,
                    time_offset=time_offset,
                    chip_offset=chip_offset_from_satellite,
                    sample_offset=int(sample_offset_where_we_started_receiving_prn),
                )

        # Failed to find a PRN correlation peak strong enough to count as a detection
        return None


def main_run_satellite_detection():
    input_source = INPUT_SOURCES[6]
    sample_rate = input_source.sdr_sample_rate

    antenna_data = get_samples_from_radio_input_source(input_source, sample_rate)

    satellite_detector = GpsSatelliteDetector()
    detected_satellites = satellite_detector.detect_satellites_in_data(antenna_data)
    for satellite_id, detected_satellite_info in detected_satellites.items():
        print(f'{satellite_id}: {detected_satellite_info}')


def main2():
    input_source = INPUT_SOURCES[5]
    sample_rate = input_source.sdr_sample_rate

    # TODO(PT): When I switch back to live SDR processing, we'll need to ensure that we normalize the actual
    # sample rate coming off the SDR to the sample rate we expect (SAMPLES_PER_SECOND).
    sdr_data = get_samples_from_radio_input_source(input_source, sample_rate)
    # Read 1 second worth of antenna data to search through
    satellite_detector = GpsSatelliteDetector()
    subset_of_antenna_data_for_satellite_detection = sdr_data[:SAMPLES_PER_SECOND]
    if False:
        detected_satellites = satellite_detector.detect_satellites_in_data(
            subset_of_antenna_data_for_satellite_detection)
    else:
        # With BPSK demodulation
        detected_satellites = {
            GpsSatelliteId(id=10): DetectedSatelliteInfo(satellite_id=GpsSatelliteId(id=10),
                                                         doppler_frequency_shift=-2500,
                                                         time_offset=0.0002785923753665689, chip_offset=285.0,
                                                         sample_offset=1476),
            GpsSatelliteId(id=24): DetectedSatelliteInfo(satellite_id=GpsSatelliteId(id=24),
                                                         doppler_frequency_shift=-2500,
                                                         time_offset=0.00012170087976539589, chip_offset=124.5,
                                                         sample_offset=1797),
            GpsSatelliteId(id=10): DetectedSatelliteInfo(satellite_id=GpsSatelliteId(id=15),
                                                         doppler_frequency_shift=500, time_offset=0.0009608993157380253,
                                                         chip_offset=983.0, sample_offset=80),
            GpsSatelliteId(id=10): DetectedSatelliteInfo(satellite_id=GpsSatelliteId(id=12),
                                                         doppler_frequency_shift=-5500,
                                                         time_offset=0.0007038123167155425, chip_offset=720.0,
                                                         sample_offset=606),
        }
        # Prior to BPSK demodulation
        detected_satellites = {
            GpsSatelliteId(id=24): DetectedSatelliteInfo(satellite_id=GpsSatelliteId(id=24),
                                                         doppler_frequency_shift=-2500,
                                                         time_offset=0.00012121212121212121, chip_offset=124.0,
                                                         sample_offset=1798),
            GpsSatelliteId(id=10): DetectedSatelliteInfo(satellite_id=GpsSatelliteId(id=10),
                                                         doppler_frequency_shift=-2500,
                                                         time_offset=0.0002785923753665689, chip_offset=285.0,
                                                         sample_offset=1476),
            GpsSatelliteId(id=15): DetectedSatelliteInfo(satellite_id=GpsSatelliteId(id=15),
                                                         doppler_frequency_shift=500, time_offset=0.0009608993157380253,
                                                         chip_offset=983.0, sample_offset=80),
            GpsSatelliteId(id=12): DetectedSatelliteInfo(satellite_id=GpsSatelliteId(id=12),
                                                         doppler_frequency_shift=-5500,
                                                         time_offset=0.0007028347996089931, chip_offset=719.0,
                                                         sample_offset=608),
        }

    for detected_satellite_id, detected_satellite_info in detected_satellites.items():
        print(detected_satellite_id, detected_satellite_info)

        start_time = 0
        end_time = 3
        time_domain = np.arange(start_time, end_time, 1 / (float(SAMPLES_PER_SECOND)))
        # For now just use the same sample count as the other bit of code, just testing
        test_sample_count = len(time_domain)
        test_sample_count = int(PRN_CORRELATION_CYCLE_COUNT * SAMPLES_PER_PRN_TRANSMISSION)
        # 200ms
        # This should give us between 8-10 data bits (depending on whether we were in the middle of a bit transition
        # when we started listening)
        test_sample_count = int(SAMPLES_PER_PRN_TRANSMISSION) * 200
        trimmed_time_domain = time_domain[:test_sample_count]
        # Take a sample of antenna data and multiply it by the computed Doppler and time shift
        # antenna_data = subset_of_antenna_data_for_satellite_detection[:test_sample_count]
        antenna_data = sdr_data[:test_sample_count]

        # Doppler the carrier wave to match the correlation peak
        doppler_shifted_carrier_wave = generate_cosine_with_frequency(trimmed_time_domain,
                                                                      detected_satellite_info.doppler_frequency_shift)
        # antenna_data_multiplied_with_doppler_shifted_carrier = antenna_data * doppler_shifted_carrier_wave

        satellite_prn_fft = satellite_detector.satellites_by_id[detected_satellite_id].fft_of_prn_of_length(
            test_sample_count)

        best_doppler_shift = None
        best_correlation_peak = 0
        best_sample_offset = 0
        for precise_doppler_shift in range(
                detected_satellite_info.doppler_frequency_shift - 25,
                detected_satellite_info.doppler_frequency_shift + 25,
                5,
        ):
            precise_doppler_shifted_carrier = generate_cosine_with_frequency(trimmed_time_domain, precise_doppler_shift)
            correlation = satellite_detector._compute_correlation(
                antenna_data,
                precise_doppler_shifted_carrier,
                satellite_prn_fft,
            )
            highest_peak = np.max(correlation)
            if highest_peak > best_correlation_peak:
                best_correlation_peak = highest_peak
                best_doppler_shift = precise_doppler_shift

                indexes_of_peaks_above_prn_correlation_threshold = list(
                    sorted(np.argwhere(correlation >= PRN_CORRELATION_MAGNITUDE_THRESHOLD)))
                if len(indexes_of_peaks_above_prn_correlation_threshold):
                    best_sample_offset = indexes_of_peaks_above_prn_correlation_threshold[0][0]
                    print(
                        f'Found new highest peak with doppler shift of {precise_doppler_shift}: {highest_peak} (sample offset {best_sample_offset})')

        print(f'Found best Doppler shift: {best_doppler_shift}')
        precise_carrier_wave = generate_cosine_with_frequency(trimmed_time_domain, best_doppler_shift)
        # antenna_data_with_precise_wave = antenna_data * precise_carrier_wave

        # plt.plot(antenna_data_with_precise_wave)
        # plt.show()
        correlation_sample_count = len(antenna_data)
        fft_received_signal = np.fft.fft(antenna_data)
        prn = satellite_detector.satellites_by_id[detected_satellite_id].prn_as_complex
        tiled_prn = np.tile(prn, 6000)
        trimmed_time_shifted_prn = tiled_prn[:test_sample_count]
        fft_satellite_prn_code = np.fft.fft(trimmed_time_shifted_prn)

        cross_correlation = np.fft.ifft(fft_received_signal * np.conj(fft_satellite_prn_code))
        plt.plot([x.real for x in cross_correlation])
        plt.plot([x.imag for x in cross_correlation])
        plt.show()
        return
        phase_offset = np.argmax(np.abs(cross_correlation))

        antenna_data_multiplied_with_doppler_shifted_carrier = antenna_data * doppler_shifted_carrier_wave
        # fft_of_doppler_shifted_signal = np.fft.fft(antenna_data_multiplied_with_doppler_shifted_carrier)
        # Perform BPSK demodulation
        threshold = 0.0  # Adjust the threshold based on the signal characteristics
        demodulated_bits = (np.real(antenna_data_multiplied_with_doppler_shifted_carrier) > threshold).astype(
            int) * 2 - 1
        fft_of_demodulated_doppler_shifted_signal = np.fft.fft(demodulated_bits)

        mult = prn_fft * np.conjugate(fft_of_demodulated_doppler_shifted_signal)
        mult_ifft = np.fft.ifft(mult)
        scaled_ifft = mult_ifft * ((1 / correlation_sample_count) * correlation_sample_count)
        correlation = np.absolute(scaled_ifft)

        compensated_signal = np.exp(
            -1j * 2 * np.pi * best_doppler_shift / sample_rate * np.arange(test_sample_count)) * antenna_data
        plt.plot(compensated_signal)
        plt.show()

        return

        # Perform BPSK demodulation
        threshold = 0.0  # Adjust the threshold based on the signal characteristics
        demodulated_bits = (np.real(antenna_data_with_precise_wave) > threshold).astype(int) * 2 - 1

        # Time-shift the PRN to match the offset we detected
        prn = satellite_detector.satellites_by_id[detected_satellite_id].prn_as_complex
        # Just tile a bunch for now
        tiled_prn = np.tile(prn, 6000)
        # Apply time shift
        time_shifted_prn = tiled_prn[best_sample_offset:]
        trimmed_time_shifted_prn = time_shifted_prn[:test_sample_count]

        correlation_result = np.correlate(antenna_data_with_precise_wave, trimmed_time_shifted_prn, mode='valid')
        # Find the peak of the correlation (code phase)
        code_phase = np.argmax(np.abs(correlation_result))
        print(correlation_result)
        plt.plot(correlation_result)
        print(code_phase)
        plt.show()

        # Multiply them
        demodulated = demodulated_bits * trimmed_time_shifted_prn

        # Output demodulated bits
        print("Demodulated Bits:", "".join(['0' if x == -1 else '1' for x in demodulated]))

        # plt.scatter(trimmed_time_domain, [x.real for x in demodulated])
        # plt.scatter(trimmed_time_domain, [x.imag for x in demodulated])
        plt.scatter(trimmed_time_domain, demodulated)
        plt.show()

        return

        # Multiply them
        demodulated = antenna_data_multiplied_with_doppler_shifted_carrier * trimmed_time_shifted_prn
        plt.plot(trimmed_time_domain, demodulated)
        # plt.plot(antenna_data_multiplied_with_doppler_shifted_carrier)
        # plt.plot(trimmed_time_domain, demodulated)
        plt.show()
        return

        if False:
            satellite_prn_fft = self.satellites_by_id[satellite_id].fft_of_prn_of_length(
                correlation_bucket_sample_count)
            print(f'Calculating {self.satellite_id} PRN FFT of length {vector_size}...')
            needed_repetitions_to_match_vector_size = math.ceil(vector_size / len(self.prn_as_complex))
            prn_of_correct_length = np.tile(self.prn_as_complex, needed_repetitions_to_match_vector_size)[:vector_size]
            return np.fft.fft(prn_of_correct_length)

            doppler_shifted_carrier_waves = {
                doppler_shift: generate_cosine_with_frequency(time_domain_for_correlation_bucket, doppler_shift)
                for doppler_shift in range(
                    DOPPLER_SHIFT_FREQUENCY_LOWER_BOUND,
                    DOPPLER_SHIFT_FREQUENCY_UPPER_BOUND,
                    DOPPLER_SHIFT_SEARCH_INTERVAL,
                )
            }


def main2():
    import numpy as np

    # Sample rate of the received signal (in Hz)
    sample_rate = 1e6  # 1 MHz

    # Doppler frequency shift (in Hz)
    doppler_shift = 5000  # 5 kHz (example value)

    # Symbol rate (baud rate) of the BPSK signal (in Hz)
    symbol_rate = 1000  # 1 kHz (example value)

    # Create a BPSK signal as an example
    num_symbols = 1000
    bpsk_symbols = np.random.randint(0, 2, num_symbols) * 2 - 1  # BPSK symbols (+1 or -1)

    # Modulate BPSK symbols to generate the transmitted signal
    t = np.arange(num_symbols) / symbol_rate
    transmitted_signal = np.sqrt(2) * bpsk_symbols * np.cos(2 * np.pi * symbol_rate * t)

    # Simulate Doppler shift
    t_with_doppler = np.arange(num_symbols) / (sample_rate + doppler_shift)
    received_signal = transmitted_signal * np.exp(2j * np.pi * doppler_shift * t_with_doppler)

    # Timing Synchronization (Early-Late Gate Timing Recovery)
    timing_offset = 0  # Initial timing offset (samples)
    loop_filter_gain = 0.01  # Loop filter gain (adjust as needed)

    # Loop variables
    early_output = 0
    late_output = 0
    timing_errors = []

    # Iterate over the received signal
    for i in range(1, num_symbols):
        # Early and late outputs for timing error calculation
        early_output = np.real(received_signal[i - 1]) * np.real(received_signal[i])
        late_output = np.imag(received_signal[i - 1]) * np.imag(received_signal[i])

        # Calculate timing error using the early-late gate method
        timing_error = early_output - late_output

        # Update timing offset using the loop filter
        timing_offset += loop_filter_gain * timing_error

        # Adjust sampling instants based on timing offset
        sampled_symbol_index = int(i - timing_offset)

        # Ensure the sampled index is within bounds
        sampled_symbol_index = max(0, min(num_symbols - 1, sampled_symbol_index))

        # Store timing errors for analysis (optional)
        timing_errors.append(timing_error)

        # Perform BPSK demodulation at the adjusted sampling instant
        # (Code for BPSK demodulation and symbol decision goes here)
        # ...

    # Print the timing errors (optional)
    print("Timing Errors:", timing_errors)


def main():
    input_source = INPUT_SOURCES[6]
    sample_rate = input_source.sdr_sample_rate

    # TODO(PT): When I switch back to live SDR processing, we'll need to ensure that we normalize the actual
    # sample rate coming off the SDR to the sample rate we expect (SAMPLES_PER_SECOND).
    sdr_data = get_samples_from_radio_input_source(input_source, sample_rate)
    # Read 1 second worth of antenna data to search through
    satellite_detector = GpsSatelliteDetector()
    # detected_satellite = DetectedSatelliteInfo(satellite_id=GpsSatelliteId(id=24), doppler_frequency_shift=-2500, time_offset=0.00012121212121212121, chip_offset=124.0, sample_offset=1798)
    detected_satellite = DetectedSatelliteInfo(satellite_id=GpsSatelliteId(id=18), doppler_frequency_shift=-1000,
                                               time_offset=-0.0021959921798631477, chip_offset=-2246.5,
                                               sample_offset=6539)
    detected_satellite_id = detected_satellite.satellite_id

    start_time = 0
    end_time = 1
    time_domain = np.arange(start_time, end_time, 1 / (float(SAMPLES_PER_SECOND)))

    # 200ms
    # This should give us between 8-10 data bits (depending on whether we were in the middle of a bit transition
    # when we started listening)
    test_sample_count = int(SAMPLES_PER_PRN_TRANSMISSION) * 800
    trimmed_time_domain = time_domain[:test_sample_count]
    antenna_data = sdr_data[:test_sample_count]

    satellite_prn_fft = satellite_detector.satellites_by_id[detected_satellite_id].fft_of_prn_of_length(
        test_sample_count)

    best_doppler_shift = detected_satellite.doppler_frequency_shift
    precise_doppler_shifted_carrier = generate_cosine_with_frequency(trimmed_time_domain, best_doppler_shift)
    correlation = satellite_detector._compute_correlation(
        antenna_data,
        np.array(precise_doppler_shifted_carrier),
        satellite_prn_fft,
    )
    plt.plot(correlation)
    # plt.show()
    phase_offset = np.argmax(np.abs(correlation)) % SAMPLES_PER_PRN_TRANSMISSION

    print(f'Rebasing antenna data phase {phase_offset}, {SAMPLES_PER_PRN_TRANSMISSION - phase_offset}...')
    # aligned_antenna_data = sdr_data[(SAMPLES_PER_PRN_TRANSMISSION - phase_offset):]
    aligned_antenna_data = sdr_data[(SAMPLES_PER_PRN_TRANSMISSION - phase_offset):]
    aligned_antenna_data = aligned_antenna_data[:test_sample_count]
    correlation = satellite_detector._compute_correlation(
        aligned_antenna_data,
        np.array(generate_cosine_with_frequency(trimmed_time_domain, best_doppler_shift)),
        satellite_prn_fft,
    )
    plt.plot(correlation)
    phase_offset = np.argmax(np.abs(correlation))
    print(phase_offset)
    # plt.show()

    antenna_data_multiplied_with_doppler_shifted_carrier = antenna_data * np.array(precise_doppler_shifted_carrier)
    # plt.plot(antenna_data_multiplied_with_doppler_shifted_carrier)
    # plt.show()

    satellite_prn = satellite_detector.satellites_by_id[detected_satellite_id].prn_as_complex
    plt.cla()
    plt.plot([x.real for x in satellite_prn])
    plt.plot([x.imag for x in satellite_prn])
    plt.show()
    tiled_prn = np.tile(satellite_prn, 1000)[:test_sample_count]
    antenna_data_multiplied_with_prn = antenna_data_multiplied_with_doppler_shifted_carrier * tiled_prn
    plt.cla()
    plt.plot(trimmed_time_domain, [x.real for x in antenna_data_multiplied_with_prn])
    plt.plot(trimmed_time_domain, [x.imag for x in antenna_data_multiplied_with_prn])
    plt.show()

    # plt.plot(bit)
    # plt.plot(smoothed_data)
    # plt.plot(normalized_data)
    # plt.plot(cleaned_data)
    # plt.show()

    bit_length = 2 * 1023 * 20  # Length of each BPSK bit in samples
    bits = [binary_signal[i:i + bit_length] for i in range(0, len(binary_signal), bit_length)]
    for i, bit in enumerate(bits):
        smoothed_data = savgol_filter(bit, window_length=100, polyorder=3)
        normalized_data = (smoothed_data - np.mean(smoothed_data)) / np.std(smoothed_data)
        median = np.median(normalized_data)
        std_dev = np.std(normalized_data)
        threshold = median + 2 * std_dev  # Define a threshold (adjust multiplier as needed)
        cleaned_data = np.clip(normalized_data, median - threshold, median + threshold)
        # plt.plot(bit)
        # plt.plot(smoothed_data)
        # plt.plot(normalized_data)
        plt.plot(cleaned_data)
        plt.show()

        print(f'Bit {i}: (len({len(bit)})')
        # print(list(bit))


def frequency_domain_correlation(prn, antenna_data):
    # Zero pad the sequences to double the length for cyclic convolution
    n = len(prn) + len(antenna_data) - 1
    print(n)
    prn_padded = np.pad(prn, (0, n - len(prn)))
    print(prn_padded)
    antenna_data_padded = np.pad(antenna_data, (0, n - len(antenna_data)))
    print(antenna_data_padded)

    # FFT of both sequences
    PRN_fft = np.fft.fft(prn_padded)
    AntennaData_fft = np.fft.fft(antenna_data_padded)

    # Multiply PRN_fft with the conjugate of AntennaData_fft
    result_fft = PRN_fft * np.conjugate(AntennaData_fft)

    # Inverse FFT to get the correlation in time domain
    correlation_result = np.fft.ifft(result_fft)

    # Return the valid portion of the correlation result
    # return correlation_result[:1-len(prn)]
    return correlation_result


def time_domain_correlation(prn, antenna_data):
    return np.correlate(antenna_data, prn, mode='valid')


def main_decode_nav_bits():
    input_source = INPUT_SOURCES[6]
    sample_rate = input_source.sdr_sample_rate

    sdr_data = get_samples_from_radio_input_source(input_source, sample_rate)
    detected_satellite = DetectedSatelliteInfo(satellite_id=GpsSatelliteId(id=18), doppler_frequency_shift=-1000,
                                               time_offset=-0.0021959921798631477, chip_offset=-2246.5,
                                               sample_offset=6539)
    detected_satellite_id = detected_satellite.satellite_id

    start_time = 0
    end_time = 1
    time_domain = np.arange(start_time, end_time, 1 / (float(SAMPLES_PER_SECOND)))

    satellite_detector = GpsSatelliteDetector()
    best_doppler_shift = detected_satellite.doppler_frequency_shift

    ms_count = 20
    twenty_ms_chunk_size = int(SAMPLES_PER_PRN_TRANSMISSION) * ms_count
    satellite_prn_fft = satellite_detector.satellites_by_id[detected_satellite_id].fft_of_prn_of_length(
        twenty_ms_chunk_size)
    twenty_ms_time_chunk = time_domain[:twenty_ms_chunk_size]
    precise_doppler_shifted_carrier = generate_cosine_with_frequency(twenty_ms_chunk_size, best_doppler_shift)
    # plt.plot(precise_doppler_shifted_carrier)
    # plt.show()
    # plt.plot(satellite_prn_fft)
    # plt.show()
    # plt.plot(np.fft.ifft(satellite_prn_fft) * twenty_ms_chunk_size)
    # plt.plot(np.tile(np.fft.ifft(satellite_prn_fft), ms_count))
    # plt.show()

    for twenty_ms_antenna_data_chunk in chunks(sdr_data, twenty_ms_chunk_size):
        print(f'Processing 20ms chunk...')
        prn = satellite_detector.satellites_by_id[detected_satellite_id].prn_as_complex
        corr = frequency_domain_correlation(
            prn, twenty_ms_antenna_data_chunk * precise_doppler_shifted_carrier
        )
        # plt.plot(corr)
        # plt.show()

        correlation = satellite_detector._compute_correlation(
            twenty_ms_antenna_data_chunk,
            np.array(precise_doppler_shifted_carrier),
            satellite_prn_fft,
        )
        plt.plot(correlation)
        plt.show()
        continue

        correlation = satellite_detector._compute_correlation(
            twenty_ms_antenna_data_chunk,
            np.array(precise_doppler_shifted_carrier),
            satellite_prn_fft,
        )
        plt.plot(correlation.real)
        plt.show()
        bit_signs = []
        for i in range(0, len(correlation), SAMPLES_PER_PRN_TRANSMISSION):
            segment = correlation[i:i + SAMPLES_PER_PRN_TRANSMISSION]
            # peak_sign = np.sign(np.max(segment, key=abs))
            peak_value = max(segment, key=abs)
            peak_sign = np.sign(peak_value)
            print(f'i={i} peak={peak_sign}')
            # bit_signs.append(peak_sign)
    return
    # Detect the bit sign for each 1ms segment
    # For simplicity, we'll assume the sign of the peak within each 1ms segment represents the bit sign.
    # In practice, you might want to average over the segment or use other methods to robustly detect the sign.

    bit_signs = []
    for i in range(0, len(correlation), samples_per_code):
        segment = correlation[i:i + samples_per_code]
        peak_sign = np.sign(np.max(segment, key=abs))
        bit_signs.append(peak_sign)

    # Should be 50 bits?
    test_sample_count = int(SAMPLES_PER_PRN_TRANSMISSION) * 1000
    trimmed_time_domain = time_domain[:test_sample_count]
    antenna_data = sdr_data[:test_sample_count]

    # plt.plot(correlation)
    # plt.show()
    phase_offset = np.argmax(np.abs(correlation)) % SAMPLES_PER_PRN_TRANSMISSION

    print(f'Rebasing antenna data phase {phase_offset}, {SAMPLES_PER_PRN_TRANSMISSION - phase_offset}...')
    # aligned_antenna_data = sdr_data[(SAMPLES_PER_PRN_TRANSMISSION - phase_offset):]
    aligned_antenna_data = sdr_data[(SAMPLES_PER_PRN_TRANSMISSION - phase_offset):]
    aligned_antenna_data = aligned_antenna_data[:test_sample_count]
    correlation = satellite_detector._compute_correlation(
        aligned_antenna_data,
        np.array(generate_cosine_with_frequency(trimmed_time_domain, best_doppler_shift)),
        satellite_prn_fft,
    )
    plt.plot(correlation)
    plt.show()
    # phase_offset = np.argmax(np.abs(correlation))
    # print(phase_offset)


def main_test():
    input_source = INPUT_SOURCES[6]
    sample_rate = input_source.sdr_sample_rate

    antenna_data = get_samples_from_radio_input_source(input_source, sample_rate)
    # detected_satellite = DetectedSatelliteInfo(satellite_id=GpsSatelliteId(id=18), doppler_frequency_shift=-1000, time_offset=-0.0021959921798631477, chip_offset=-2246.5, sample_offset=6539)
    detected_satellite = DetectedSatelliteInfo(satellite_id=GpsSatelliteId(id=5), doppler_frequency_shift=0,
                                               time_offset=-0.0325, chip_offset=-33247.5, sample_offset=68541)
    detected_satellite_id = detected_satellite.satellite_id

    satellite_detector = GpsSatelliteDetector()
    best_doppler_shift = detected_satellite.doppler_frequency_shift

    prn = satellite_detector.satellites_by_id[detected_satellite_id].prn_as_complex
    # Zero-pad the PRN to match a chunk length (e.g., 2046 for direct correlation without overlap)
    chunk_length = 2046 * 40
    prn_padded = np.pad(prn, (0, chunk_length - len(prn)))

    # Compute FFT of zero-padded PRN
    prn_fft = np.fft.fft(prn_padded)

    # Initialize an array to store correlation results
    correlation_results = []

    # Iterate over antenna data in chunks
    # 350 = 8.75516325282047
    # 356 = 8.756593613581858
    # 358 = 8.756859904078814
    # 360 = 8.757020976853438
    # 361 = 8.757062063425717
    # 362 = 8.757076854088336
    #  .1 = 8.757076887000503
    # .25 = 8.757076443386925
    #  .5 = 8.757074389440207
    # 363 = 8.757065352018927
    # 364 = 8.757027560556963
    # 367 = 8.756756485637235
    # 375 = 8.754877871521707
    # 400 = 8.738199782243697
    doppler_shift = 362.1
    for i in range(0, len(antenna_data) - chunk_length + 1, chunk_length):
        chunk = antenna_data[i:i + chunk_length]
        precise_doppler_shifted_carrier = generate_cosine_with_frequency(chunk_length, doppler_shift)
        chunk = chunk * precise_doppler_shifted_carrier
        chunk_fft = np.fft.fft(chunk)

        # Frequency domain multiplication
        product_fft = prn_fft * np.conjugate(chunk_fft)

        # Compute inverse FFT to obtain correlation result for this chunk
        correlation_chunk = np.fft.ifft(product_fft)

        # correlation_results.append(np.abs(correlation_chunk))
        non_coherent_integration = np.abs(correlation_chunk)
        print(np.max(non_coherent_integration))

        plt.plot(non_coherent_integration)
        plt.show()
        # continue

        # Get the index of the peak
        switch = 2
        if switch == 0:
            # peak_index = np.argmax(non_coherent_integration)
            mu = np.mean(non_coherent_integration)
            sigma = np.std(non_coherent_integration)
            # Define threshold
            k = 3.3
            threshold = mu + k * sigma
            # Use scipy's find_peaks to identify peaks above the threshold
            significant_peak_indices, _ = find_peaks(non_coherent_integration, height=threshold)
        elif switch == 1:
            plt.plot(non_coherent_integration)
            plt.show()
            snr_ratio, correlation_peaks = GpsSatelliteDetector.get_ratio_of_outlier_signal(non_coherent_integration)
            significant_peak_indices = list(sorted(correlation_peaks))
            print(snr_ratio, correlation_peaks)
        elif switch == 2:
            significant_peak_indices = np.argwhere(non_coherent_integration >= 4.0)
            print(len(significant_peak_indices))
            pass
        print(len(significant_peak_indices))

        peak_sign = np.sign(correlation_chunk[significant_peak_indices].real)

        # Get the sign of the real part at the peak index
        # plt.plot(non_coherent_integration)
        plt.plot(peak_sign)
        print(peak_sign)
        plt.show()

    # Convert the list of arrays into a single numpy array
    correlation_results = np.concatenate(correlation_results)


def main_try_doppler_track():
    input_source = INPUT_SOURCES[6]
    sample_rate = input_source.sdr_sample_rate
    antenna_data = get_samples_from_radio_input_source(input_source, sample_rate)

    # Rough plan:
    # Proc 2 samples at time
    # Try to maximize correlation by going up up up or down down down
    # Save bit values (or 0 if we could identify nothing)
    # Then when we have tons of bits it should be easier to identify transitions/values

    satellite_detector = GpsSatelliteDetector()
    acquired_satellite_info = DetectedSatelliteInfo(satellite_id=GpsSatelliteId(id=5), doppler_frequency_shift=0,
                                                    time_offset=-0.0325, chip_offset=-33247.5, sample_offset=68541)

    # We've now got a 'coarse' fix on the satellite. We've detected some correlation peak, and now it's time to
    # refine our initial Doppler shift.
    #
    # To do so, we're going to select a small set of samples and progressively refine our Doppler shift to -- nvm
    satellite_prn = satellite_detector.satellites_by_id[acquired_satellite_info.satellite_id].prn_as_complex

    # Process one PRN period at a time
    samples_chunk_size = SAMPLES_PER_PRN_TRANSMISSION
    padded_satellite_prn = np.pad(satellite_prn, (0, samples_chunk_size - len(satellite_prn)))
    # Compute FFT of zero-padded PRN
    padded_satellite_prn_fft = np.fft.fft(padded_satellite_prn)

    previous_doppler_shift = acquired_satellite_info.doppler_frequency_shift
    doppler_search_step_size = 100
    previous_phase_shift = 0

    # Look at the very first chunk to identify the initial Doppler shift/phase offset
    # Eventually, this could come from the detector
    first_ms = antenna_data[:samples_chunk_size]
    previous_doppler_shift = binary_search_best_doppler_shift(
        satellite_detector,
        first_ms,
        padded_satellite_prn_fft,
        previous_doppler_shift,
        doppler_search_step_size,
    )
    previous_phase_shift = get_prn_phase_offset(
        get_correlation_with_doppler_shift(
            satellite_detector,
            first_ms,
            padded_satellite_prn_fft,
            previous_doppler_shift,
        )
    )
    # previous_doppler_shift = 0
    # previous_phase_shift = 0
    print(f'Phase offset {previous_phase_shift}')
    padded_satellite_prn_fft = adjust_prn_fft_phase(padded_satellite_prn_fft, previous_phase_shift)

    bits = []
    doppler_search_step_size = 3
    print(f'Doppler {previous_doppler_shift}')
    previous_doppler_shift = 00
    for previous_doppler_shift in range(0, 500, 50):
        bad_chunks = 0
        good_chunks = 0
        bits = [0]
        print(previous_doppler_shift)
        for antenna_data_chunk_that_should_contain_one_prn in chunks(antenna_data, samples_chunk_size):
            papr = get_non_coherent_correlation_strength_with_doppler_shift(
                satellite_detector,
                antenna_data_chunk_that_should_contain_one_prn,
                padded_satellite_prn_fft,
                previous_doppler_shift,
            )
            print(papr)
            if papr < 14:
                print(f'skipping bad chunk, replicating last pseudo-symbol ({bad_chunks} / {good_chunks})')
                bad_chunks += 1
                bits.append(bits[-1])
                continue
            good_chunks += 1

            correlation_of_chunk = get_correlation_with_doppler_shift(
                satellite_detector,
                antenna_data_chunk_that_should_contain_one_prn,
                padded_satellite_prn_fft,
                previous_doppler_shift,
            )

            # We should be phase-aligned, so we should be able to find the peak within the first few samples
            # This helps reject peaks from noise elsewhere in the chunk
            start_of_chunk = correlation_of_chunk[:20]
            # start_of_chunk = correlation_of_chunk
            significant_peak_indices = np.argmax(np.abs(start_of_chunk) ** 2)
            bit_value = np.sign(correlation_of_chunk[significant_peak_indices].real)
            bits.append(bit_value)

            if False:
                plt.plot(correlation_of_chunk)
                plt.scatter(significant_peak_indices, 5 if bit_value == 1.0 else -5, color="red")
                plt.show()

            # Maybe need to reject anything that's way different from the last sample?
            # PT: The Doppler shift is jumping all over the place because we're trying to maximize the correlation strength,
            # but the correlation strength will jump all over the place based on the stregth of the signal?
            doppler_shift = binary_search_best_doppler_shift(
                satellite_detector,
                antenna_data_chunk_that_should_contain_one_prn,
                padded_satellite_prn_fft,
                previous_doppler_shift,
                doppler_search_step_size,
            )

            doppler_shift_diff = doppler_shift - previous_doppler_shift
            print(f'Best shift: {doppler_shift}Hz (shifted {doppler_shift_diff}Hz from the previous chunk)')
            previous_doppler_shift = doppler_shift
            if False:
                plt.plot(
                    np.abs(
                        get_correlation_with_doppler_shift(
                            satellite_detector,
                            antenna_data_chunk_that_should_contain_one_prn,
                            padded_satellite_prn_fft,
                            doppler_shift,
                        )
                    ) ** 2
                )
                plt.show()

            # 50bps * 20 pseudo-symbols per bit
            if len(bits) > 20 * 4:
                break
        plt.plot(bits)
        plt.show()
    return

    for antenna_data_chunk_that_should_contain_one_prn in chunks(antenna_data, samples_chunk_size):
        correlation_of_chunk = get_correlation_with_doppler_shift(
            satellite_detector,
            antenna_data_chunk_that_should_contain_one_prn,
            padded_satellite_prn_fft,
            previous_doppler_shift,
        )

        # We should be phase-aligned, so we should be able to find the peak within the first few samples
        # This helps reject peaks from noise elsewhere in the chunk
        start_of_chunk = correlation_of_chunk[:20]
        significant_peak_indices = np.argmax(np.abs(start_of_chunk) ** 2)
        bit_value = np.sign(correlation_of_chunk[significant_peak_indices].real)
        bits.append(bit_value)

        # plt.plot(correlation_of_chunk)
        # plt.scatter(significant_peak_indices, 5 if bit_value == 1.0 else -5, color="red")
        # plt.show()

        # 50bps * 20 pseudo-symbols per bit
        if len(bits) > 20 * 4:
            break

        if True:
            # Maybe need to reject anything that's way different from the last sample?
            # PT: The Doppler shift is jumping all over the place because we're trying to maximize the correlation strength,
            # but the correlation strength will jump all over the place based on the stregth of the signal?
            doppler_shift = binary_search_best_doppler_shift(
                satellite_detector,
                antenna_data_chunk_that_should_contain_one_prn,
                padded_satellite_prn_fft,
                previous_doppler_shift,
                doppler_search_step_size,
            )

            doppler_shift_diff = doppler_shift - previous_doppler_shift
            print(f'Best shift: {doppler_shift}Hz (shifted {doppler_shift_diff}Hz from the previous chunk)')
            previous_doppler_shift = doppler_shift
            if False:
                plt.plot(
                    np.abs(
                        get_correlation_with_doppler_shift(
                            satellite_detector,
                            antenna_data_chunk_that_should_contain_one_prn,
                            padded_satellite_prn_fft,
                            doppler_shift,
                        )
                    )
                )
                plt.show()
    plt.plot(bits[:50 * 20])
    plt.show()


def binary_search_best_doppler_shift(
        detector: GpsSatelliteDetector,
        data: np.ndarray,
        prn_fft: np.ndarray,
        initial_doppler_shift_frequency: float,
        initial_step_size: float,
) -> float:
    doppler_shift = initial_doppler_shift_frequency
    doppler_search_step_size = initial_step_size
    while doppler_search_step_size > 0.1:
        correlation_at_current_shift = get_non_coherent_correlation_strength_with_doppler_shift(
            detector,
            data,
            prn_fft,
            doppler_shift,
        )

        lower_doppler_shift = doppler_shift - doppler_search_step_size
        correlation_at_lower_shift = get_non_coherent_correlation_strength_with_doppler_shift(
            detector,
            data,
            prn_fft,
            lower_doppler_shift
        )
        higher_doppler_shift = doppler_shift + doppler_search_step_size
        correlation_at_higher_shift = get_non_coherent_correlation_strength_with_doppler_shift(
            detector,
            data,
            prn_fft,
            higher_doppler_shift,
        )

        # print(f'Correlation at current frequency {doppler_shift}Hz: {correlation_at_current_shift}')
        # print(f'Correlation at lower shift {lower_doppler_shift}Hz: {correlation_at_lower_shift}')
        # print(f'Correlation at higher shift {higher_doppler_shift}Hz: {correlation_at_higher_shift}')

        if correlation_at_lower_shift < correlation_at_current_shift and correlation_at_higher_shift < correlation_at_current_shift:
            # print(f'Neither direction improved correlation. Reducing shift bin size from {doppler_search_step_size}Hz to {doppler_search_step_size/2}Hz...')
            doppler_search_step_size /= 2
        elif correlation_at_lower_shift > correlation_at_current_shift:
            # print(f'Lower correlation was an improvement, pivoting to {lower_doppler_shift}Hz...')
            doppler_shift = lower_doppler_shift
        elif correlation_at_higher_shift > correlation_at_current_shift:
            # print(f'Higher correlation was an improvement, pivoting to {higher_doppler_shift}Hz...')
            doppler_shift = higher_doppler_shift
        else:
            raise ValueError(f'Unexpected result?')
    # print(f'Best shift: {doppler_shift}Hz')
    return doppler_shift


def get_non_coherent_correlation_strength_with_doppler_shift(
        detector: GpsSatelliteDetector,
        data: np.ndarray,
        prn_fft: np.ndarray,
        doppler_shift_frequency: float,
) -> float:
    correlation_of_chunk = get_correlation_with_doppler_shift(
        detector,
        data,
        prn_fft,
        doppler_shift_frequency,
    )
    # Detect the (non-coherent) correlation magnitude
    # non_coherent_integration = np.abs(correlation_of_chunk)
    # return np.max(non_coherent_integration)

    # PAPR
    # highest_peak_squared = np.max(correlation_of_chunk)**2
    # correlation_average_squared = np.mean(correlation_of_chunk**2)
    # papr = highest_peak_squared / correlation_average_squared
    peak_power = np.max(np.abs(correlation_of_chunk)) ** 2
    average_power = np.mean(np.abs(correlation_of_chunk) ** 2)
    papr = peak_power / average_power
    # print(f'PAPR {peak_power:.2f}/{average_power:.2f} = {papr:.2f}')
    return papr


def get_correlation_with_doppler_shift(
        detector: GpsSatelliteDetector,
        data: np.ndarray,
        prn_fft: np.ndarray,
        doppler_shift_frequency: float,
) -> np.ndarray:
    # Detect the (non-coherent) correlation magnitude
    doppler_shifted_carrier = detector.get_doppler_shifted_carrier(len(data), doppler_shift_frequency)
    doppler_shifted_data_fft = np.fft.fft(data * doppler_shifted_carrier)

    # Amplify by the PRN
    product_fft = prn_fft * np.conjugate(doppler_shifted_data_fft)
    # Compute inverse FFT to obtain correlation result for this chunk
    chunk_correlation = np.fft.ifft(product_fft)
    return chunk_correlation


def get_prn_phase_offset(prn_correlation: np.ndarray) -> int:
    # The index of the non-coherent correlation peak represents the number of samples we'll need to shift to
    # phase-align the PRN.
    return np.argmax(np.abs(prn_correlation) ** 2) % SAMPLES_PER_PRN_TRANSMISSION


def adjust_prn_fft_phase(prn_fft: np.ndarray, phase_offset: float) -> np.ndarray:
    freqs = np.fft.fftfreq(len(prn_fft))

    # Compute the phase shift for each frequency bin
    phase_shifts = -2 * np.pi * freqs * phase_offset

    # Generate the complex exponential for the phase shifts
    phase_adjustment = np.exp(-1j * phase_shifts)

    # Adjust the PRN FFT with the phase
    adjusted_prn_fft = prn_fft * phase_adjustment

    return adjusted_prn_fft


def main_try_costas_loop():
    input_source = INPUT_SOURCES[6]
    sample_rate = input_source.sdr_sample_rate
    antenna_data = get_samples_from_radio_input_source(input_source, sample_rate)
    satellite_detector = GpsSatelliteDetector()
    acquired_satellite_info = DetectedSatelliteInfo(satellite_id=GpsSatelliteId(id=5), doppler_frequency_shift=0,
                                                    time_offset=-0.0325, chip_offset=-33247.5, sample_offset=68541)
    satellite_prn = satellite_detector.satellites_by_id[acquired_satellite_info.satellite_id].prn_as_complex

    # Process one PRN period at a time
    samples_chunk_size = SAMPLES_PER_PRN_TRANSMISSION
    padded_satellite_prn = np.pad(satellite_prn, (0, samples_chunk_size - len(satellite_prn)))
    # Compute FFT of zero-padded PRN
    padded_satellite_prn_fft = np.fft.fft(padded_satellite_prn)

    # Look at the very first chunk to identify the initial Doppler shift/phase offset
    # Eventually, this could come from the detector
    first_ms = antenna_data[:samples_chunk_size]
    doppler_shift = acquired_satellite_info.doppler_frequency_shift
    doppler_shift = 400
    phase_shift = get_prn_phase_offset(
        get_correlation_with_doppler_shift(
            satellite_detector,
            first_ms,
            padded_satellite_prn_fft,
            doppler_shift,
        )
    )
    # Roll the PRN to be phase-aligned with the incoming antenna signal
    # phase_shifted_prn_fft = adjust_prn_fft_phase(padded_satellite_prn_fft, phase_shift)
    # phase_shifted_prn = np.fft.ifft(phase_shifted_prn_fft)

    # loop_bandwidth = 0.01
    loop_bandwidth = 1
    costas = CostasLoop(loop_bandwidth, doppler_shift, phase_shift)

    for i, antenna_data_chunk_that_should_contain_one_prn in enumerate(chunks(antenna_data, samples_chunk_size)):
        print(f'\nProcess ms #{i}...')
        plt.plot(antenna_data_chunk_that_should_contain_one_prn.real)
        plt.plot(antenna_data_chunk_that_should_contain_one_prn.imag)
        plt.title("I and Q")
        plt.show()

    for i, antenna_data_chunk_that_should_contain_one_prn in enumerate(chunks(antenna_data, samples_chunk_size)):
        print(f'\nProcess ms #{i}...')
        doppler_shifted_carrier = satellite_detector.get_doppler_shifted_carrier(samples_chunk_size, costas.freq)
        doppler_shifted_chunk = antenna_data_chunk_that_should_contain_one_prn * doppler_shifted_carrier
        # padded_satellite_prn = np.pad(satellite_prn, (0, samples_chunk_size - len(satellite_prn)))
        # phase_shifted_prn_fft = adjust_prn_fft_phase(padded_satellite_prn_fft, costas.phase)
        # phase_shifted_prn = np.fft.ifft(phase_shifted_prn_fft)

        errors_and_tracked_samples = [costas.step(sample) for sample in antenna_data_chunk_that_should_contain_one_prn]
        errors = [t[0] for t in errors_and_tracked_samples]
        tracked_samples = np.array([t[1] for t in errors_and_tracked_samples])
        print(f'Phase: {costas.phase}')
        print(f'Frequency: {costas.freq}')

        phase_shifted_prn = np.roll(satellite_prn, int(costas.phase))
        # = doppler_shifted_chunk * phase_shifted_prn

        # non_coherent_integration = np.abs(correlation_of_chunk)
        if i > 0:
            correlation_of_chunk = get_correlation_with_doppler_shift(
                satellite_detector,
                tracked_samples,
                np.fft.fft(phase_shifted_prn),
                costas.freq,
            )
            plt.plot(np.abs(correlation_of_chunk));
            plt.title("Non-coherent correlation");
            plt.show()
            correlation_of_chunk = get_correlation_with_doppler_shift(
                satellite_detector,
                doppler_shifted_chunk,
                np.fft.fft(phase_shifted_prn),
                costas.freq,
            )
            plt.plot(np.abs(correlation_of_chunk));
            plt.title("Original sampls NCC");
            plt.show()
        # print(f'Correlation strength: {strength}')
        # plt.plot(errors);plt.title("Errors")
        # plt.show()

        if False:
            # plt.plot(tracked_samples)
            # plt.show()
            correlation_of_chunk = get_correlation_with_doppler_shift(
                satellite_detector,
                tracked_samples,
                phase_shifted_prn_fft,
                costas.freq,
            )
            plt.plot(correlation_of_chunk);
            plt.title("Correlation");
            plt.show()

        continue
        # We should be phase-aligned, so we should be able to find the peak within the first few samples
        # This helps reject peaks from noise elsewhere in the chunk
        start_of_chunk = correlation_of_chunk[:20]
        significant_peak_indices = np.argmax(np.abs(start_of_chunk) ** 2)
        bit_value = np.sign(correlation_of_chunk[significant_peak_indices].real)
        bits.append(bit_value)

        continue
        for sample in doppler_shifted_chunk_mixed_with_prn:
            pass

        # Amplify by the PRN
        product_fft = prn_fft * np.conjugate(doppler_shifted_data_fft)
        # Compute inverse FFT to obtain correlation result for this chunk
        chunk_correlation = np.fft.ifft(product_fft)
        papr = get_non_coherent_correlation_strength_with_doppler_shift(
            satellite_detector,
            antenna_data_chunk_that_should_contain_one_prn,
            padded_satellite_prn_fft,
            previous_doppler_shift,
        )


class CostasLoop:
    def __init__(self, loop_bandwidth: float, estimated_doppler_shift_from_acquisition: float,
                 estimated_phase_shift_from_acquisition: float):
        self.phase = estimated_phase_shift_from_acquisition
        self.freq = estimated_doppler_shift_from_acquisition
        self.alpha = loop_bandwidth / (np.sqrt(2) * np.pi)
        self.beta = (loop_bandwidth * loop_bandwidth) / (2 * np.pi)
        self.vco = np.exp(1j * self.phase)

    def step(self, sample):
        # 1. Multiply sample by VCO output to get downconverted sample
        downconverted_sample = sample * np.conj(self.vco)

        # 2. Compute the I and Q components
        I = downconverted_sample.real
        Q = downconverted_sample.imag

        # 3. Compute the phase error (I * Q is the basic error detector for BPSK)
        error = I * Q

        # 4. Update loop (filter) using the error
        self.freq += self.beta * error
        self.phase += self.freq + self.alpha * error

        # Wrap the phase to prevent it from growing indefinitely
        self.phase = self.phase % (2 * np.pi)

        # 5. Update the VCO based on the new phase estimate
        self.vco = np.exp(1j * self.phase)

        return error, downconverted_sample


def main_try_costas_loop2():
    input_source = INPUT_SOURCES[6]
    sample_rate = input_source.sdr_sample_rate
    antenna_data = get_samples_from_radio_input_source(input_source, sample_rate)
    satellite_detector = GpsSatelliteDetector()
    acquired_satellite_info = DetectedSatelliteInfo(satellite_id=GpsSatelliteId(id=5), doppler_frequency_shift=0,
                                                    time_offset=-0.0325, chip_offset=-33247.5, sample_offset=68541)
    satellite_prn = satellite_detector.satellites_by_id[acquired_satellite_info.satellite_id].prn_as_complex

    samples_chunk_size = SAMPLES_PER_PRN_TRANSMISSION
    padded_satellite_prn = np.pad(satellite_prn, (0, samples_chunk_size - len(satellite_prn)))
    # Compute FFT of zero-padded PRN
    padded_satellite_prn_fft = np.fft.fft(padded_satellite_prn)

    # Look at the very first chunk to identify the initial Doppler shift/phase offset
    # Eventually, this could come from the detector
    first_ms = antenna_data[:samples_chunk_size]
    doppler_shift = 362.1
    phase_shift = get_prn_phase_offset(
        get_correlation_with_doppler_shift(
            satellite_detector,
            first_ms,
            padded_satellite_prn_fft,
            doppler_shift,
        )
    )
    # Roll the PRN to be phase-aligned with the incoming antenna signal
    # phase_shifted_prn_fft = adjust_prn_fft_phase(padded_satellite_prn_fft, phase_shift)
    # phase_shifted_prn = np.fft.ifft(phase_shifted_prn_fft)
    print(f'phase shift {phase_shift}')
    phase_shifted_prn = np.roll(satellite_prn, phase_shift)
    plt.plot(satellite_prn)
    plt.plot(phase_shifted_prn)
    plt.show()

    doppler_shifted_carrier = satellite_detector.get_doppler_shifted_carrier(samples_chunk_size, doppler_shift)
    phase = 0
    freq = doppler_shift
    alpha = 0.132
    beta = 0.00932
    freq_log = []
    plt.ion()
    plt.show()
    for antenna_data_chunk_that_should_contain_one_prn in chunks(antenna_data, samples_chunk_size):
        out = np.zeros(samples_chunk_size, dtype=complex)

        # phase_shifted_prn = np.roll(satellite_prn, phase_shift)
        # phase_shifted_prn = np.roll(satellite_prn, phase_shift_in_samples)
        # antenna_data_with_despread_prn = antenna_data_chunk_that_should_contain_one_prn * phase_shifted_prn

        for sample_idx in range(samples_chunk_size):
            # Despread the PRN
            # Simplified terms because our sample rate is exactly 2x chipping rate
            # Should be offset(samp) = (offset(rad) / 2pi) * samples_per_chip
            phase_shift_in_samples = phase / np.pi
            prn_sample = satellite_prn[int(phase_shift_in_samples)]
            prn_modulated_sample = antenna_data_chunk_that_should_contain_one_prn[sample_idx] * prn_sample

            # Adjust the input sample by the inverse of the estimated phase offset
            out[sample_idx] = prn_modulated_sample * np.exp(-1j * phase)
            # This is the error formula for 2nd order Costas Loop (e.g. for BPSK)
            error = np.real(out[sample_idx]) * np.imag(out[sample_idx])

            # Advance the loop (recalc phase and freq offset)
            freq += (beta * error)
            # Convert from angular velocity to Hz for logging
            freq_log.append(freq * sample_rate / (2 * np.pi) / 1000. / 1000. / 1000.)
            phase += freq + (alpha * error)

            # Optional: Adjust phase so its always between 0 and 2pi, recall that phase wraps around every 2pi
            phase %= (2 * np.pi)

        plt.plot(freq_log)
        # plt.plot(out)
        # plt.title("Output")
        plt.pause(0.0000001)
        # plt.show()

        if False:
            correlation_of_chunk = get_correlation_with_doppler_shift(
                satellite_detector,
                antenna_data_chunk_that_should_contain_one_prn,
                padded_satellite_prn_fft,
                doppler_shift,
            )
            non_coherent = np.abs(correlation_of_chunk)
            plt.plot(correlation_of_chunk)
            plt.plot(non_coherent)
            plt.show()

    N = SAMPLES_PER_PRN_TRANSMISSION * 4000
    fs = SAMPLES_PER_SECOND
    phase = 0
    freq = 0
    # These next two params is what to adjust, to make the feedback loop faster or slower (which impacts stability)
    alpha = 0.132
    beta = 0.00932
    out = np.zeros(N, dtype=complex)
    freq_log = []
    for i in range(N):
        if i % SAMPLES_PER_PRN_TRANSMISSION == 0:
            print(i // SAMPLES_PER_PRN_TRANSMISSION)
        out[i] = doppler_shifted_data[i] * np.exp(
            -1j * phase)  # adjust the input sample by the inverse of the estimated phase offset
        error = np.real(out[i]) * np.imag(out[i])  # This is the error formula for 2nd order Costas Loop (e.g. for BPSK)

        # Advance the loop (recalc phase and freq offset)
        freq += (beta * error)
        freq_log.append(freq * fs / (2 * np.pi))  # convert from angular velocity to Hz for logging
        phase += freq + (alpha * error)

        # Optional: Adjust phase so its always between 0 and 2pi, recall that phase wraps around every 2pi
        phase %= (2 * np.pi)

        downconverted_sample = sample * np.conj(self.vco)

        # 2. Compute the I and Q components
        I = downconverted_sample.real
        Q = downconverted_sample.imag

        # 3. Compute the phase error (I * Q is the basic error detector for BPSK)
        error = I * Q

        # 4. Update loop (filter) using the error
        self.freq += self.beta * error
        self.phase += self.freq + self.alpha * error

        # Wrap the phase to prevent it from growing indefinitely
        self.phase = self.phase % (2 * np.pi)

        # 5. Update the VCO based on the new phase estimate
        self.vco = np.exp(1j * self.phase)

    # Plot freq over time to see how long it takes to hit the right offset
    plt.plot(freq_log, '.-')
    plt.show()
    return


def main_try_iq_power_tracking():
    input_source = INPUT_SOURCES[6]
    sample_rate = input_source.sdr_sample_rate
    antenna_data = get_samples_from_radio_input_source(input_source, sample_rate)
    print(f'Length of antenna data: {len(antenna_data)}')
    satellite_detector = GpsSatelliteDetector()
    acquired_satellite_info = DetectedSatelliteInfo(satellite_id=GpsSatelliteId(id=5), doppler_frequency_shift=0,
                                                    time_offset=-0.0325, chip_offset=-33247.5, sample_offset=68541)
    satellite_prn = satellite_detector.satellites_by_id[acquired_satellite_info.satellite_id].prn_as_complex

    samples_chunk_size = SAMPLES_PER_PRN_TRANSMISSION
    satellite_prn_fft = np.fft.fft(satellite_prn)

    # Look at the very first chunk to identify the initial Doppler shift/phase offset
    # Eventually, this could come from the detector
    # PT: Doppler shift from experimentation
    # doppler_shift = 362.1+100
    # doppler_shift = 300
    doppler_shift = 362.1
    # doppler_shift = 1000
    # PT: It seems *fine* to correct the phase/frequency on every PRN repetition, as the
    # navigation message is *way* less frequent than PRNs! (20 PRNs per navigation bit)
    start = SAMPLES_PER_PRN_TRANSMISSION * 0
    end = start + samples_chunk_size
    first_ms = antenna_data[start:end]
    phase_shift = get_prn_phase_offset(
        get_correlation_with_doppler_shift(
            satellite_detector,
            first_ms,
            satellite_prn_fft,
            doppler_shift,
        )
    )
    print(f'phase shift {phase_shift}')
    phase_shifted_prn = np.roll(satellite_prn, -phase_shift)

    if False:
        non_shifted_corr = np.abs(
            get_correlation_with_doppler_shift(
                satellite_detector,
                first_ms,
                satellite_prn_fft,
                doppler_shift,
            ) ** 2
        )
        plt.plot(non_shifted_corr, label="Non-shifted correlation")
        shifted_corr = np.abs(
            get_correlation_with_doppler_shift(
                satellite_detector,
                first_ms,
                np.fft.fft(phase_shifted_prn),
                doppler_shift,
            ) ** 2
        )
        plt.plot(shifted_corr, label="Phase-shifted correlation")
        plt.legend(loc="upper left")
        plt.show()

    # plt.ion()
    bits = []
    # figure, (correlation_ax, bits_ax) = plt.subplots(nrows=2, ncols=1)
    phase_shifted_prn_fft = np.fft.fft(phase_shifted_prn)

    correlations = []
    chunk_count = PRN_REPETITIONS_PER_SECOND * 6 * 8
    print(f'Desired chunk count: {chunk_count}')

    # one_ms = np.arange(0, SAMPLES_PER_PRN_TRANSMISSION) / SAMPLES_PER_SECOND
    one_ms = np.arange(0, SAMPLES_PER_PRN_TRANSMISSION)

    # Filter parameters
    # 0.06 and 0.0025 look good?!
    # [array([179]), array([529]), array([608])]
    # [array([75]), array([394]), array([676])]
    # alpha = 0.8  # Proportional gain (you'll need to adjust these)
    # beta = 0.01 # Integral gain (you'll need to adjust these)

    # alpha = 0.001  # Proportional gain (you'll need to adjust these)
    # beta = 0.1 # Integral gain (you'll need to adjust these)

    # alpha = 0.0005  # Proportional gain (you'll need to adjust these)
    # beta = 0.0002 # Integral gain (you'll need to adjust these)
    alpha = 0.0008  # Proportional gain (you'll need to adjust these)
    beta = 0.0001  # Integral gain (you'll need to adjust these)

    # Error values (you'd update these each iteration)
    error_integral = 0

    GPS_L1_FREQUENCY = 1575.42e6
    BANDWIDTH_PLL = 3
    BANDWIDTH_DLL = 6
    EARLY_LATE_SPACING = 1
    code_phase = -phase_shift
    phase_errors = []
    for i, antenna_data_chunk_that_should_contain_one_prn in enumerate(chunks(antenna_data, samples_chunk_size)):
        local_carrier_cos = np.cos(2.0 * np.pi * (doppler_shift) * one_ms)
        local_carrier_sin = np.sin(2.0 * np.pi * (doppler_shift) * one_ms)
        # local_carrier_sin = np.sin(2.0 * np.pi * doppler_shift * SAMPLES_PER_PRN_TRANSMISSION * one_ms)
        # But wait... one antenna data chunk could contain more or less than one PRN!!
        # Just because we 'want' 2046 samples to contain one PRN, doesn't mean that's actually the case (due to Doppler shift)
        # Do we need to adjust our chunk size?
        # No -- the chipping rate does shift, but proportionally to the carrier the shift is very small (as it's 1/1023 of the carrier shift)
        # print(f'Process ms #{i}...')
        if i % PRN_REPETITIONS_PER_SECOND == 0:
            print(i)
        if i > chunk_count:
            print(f'i {i} > chunk_count {chunk_count}, breaking')
            break

        if False:
            # Recalibrate Doppler and phase shift
            doppler_shift = binary_search_best_doppler_shift(
                satellite_detector,
                antenna_data_chunk_that_should_contain_one_prn,
                satellite_prn_fft,
                doppler_shift,
                50,
            )
            phase_shift = get_prn_phase_offset(
                get_correlation_with_doppler_shift(
                    satellite_detector,
                    antenna_data_chunk_that_should_contain_one_prn,
                    satellite_prn_fft,
                    doppler_shift,
                )
            )
            phase_shifted_prn = np.roll(satellite_prn, -phase_shift)
            phase_shifted_prn_fft = np.fft.fft(phase_shifted_prn)

        # I_chunk = antenna_data_chunk_that_should_contain_one_prn * phase_shifted_prn * local_carrier_cos
        # Q_chunk = antenna_data_chunk_that_should_contain_one_prn * phase_shifted_prn * local_carrier_sin
        # plt.plot(I_chunk, label="I")
        # plt.plot(Q_chunk, label="Q")
        # plt.legend()
        # plt.show()
        # phase_error = np.arctan2(Q, I)

        # I = np.sum(antenna_data_chunk_that_should_contain_one_prn * phase_shifted_prn * local_carrier_cos).real
        # Q = np.sum(antenna_data_chunk_that_should_contain_one_prn * phase_shifted_prn * local_carrier_sin).real
        # phase_error = I * Q

        integer_code_phase = int(code_phase)
        prn_early = np.roll(satellite_prn, -integer_code_phase - EARLY_LATE_SPACING)
        prn_prompt = np.roll(satellite_prn, -integer_code_phase)
        prn_late = np.roll(satellite_prn, -integer_code_phase + EARLY_LATE_SPACING)
        I_early = np.sum(antenna_data_chunk_that_should_contain_one_prn * prn_early * local_carrier_cos).real
        Q_early = np.sum(antenna_data_chunk_that_should_contain_one_prn * prn_early * local_carrier_sin).real

        I_prompt = np.sum(antenna_data_chunk_that_should_contain_one_prn * prn_prompt * local_carrier_cos).real
        Q_prompt = np.sum(antenna_data_chunk_that_should_contain_one_prn * prn_prompt * local_carrier_sin).real

        I_late = np.sum(antenna_data_chunk_that_should_contain_one_prn * prn_late * local_carrier_cos).real
        Q_late = np.sum(antenna_data_chunk_that_should_contain_one_prn * prn_late * local_carrier_sin).real

        # Update error values
        phase_error = I_prompt * Q_prompt
        phase_errors.append(phase_error)

        correlation = get_correlation_with_doppler_shift(
            satellite_detector,
            antenna_data_chunk_that_should_contain_one_prn,
            np.fft.fft(np.roll(satellite_prn, -(integer_code_phase % SAMPLES_PER_PRN_TRANSMISSION))),
            doppler_shift,
        )

        error_integral += phase_error
        filtered_error = alpha * phase_error + beta * error_integral
        doppler_shift += filtered_error
        # dll_error = I_early - I_late
        # code_phase += BANDWIDTH_DLL * dll_error
        code_phase_error = (I_late - I_early) - (Q_late - Q_early)
        code_phase += BANDWIDTH_DLL * code_phase_error

        # print(f'ms={i} I={I:.2f} Q={Q:.2f}, Phase error = {phase_error:.2f}, Error integral = {error_integral:.2f}, Filtered error = {filtered_error:.2f}, Doppler = {doppler_shift:.2f}')
        print(f'ms={i} Filtered error = {filtered_error:.2f}, Doppler = {doppler_shift:.2f}, Phase = {code_phase:.2f}')

        if True and i > 10000:
            plt.plot(correlation, label="Correlation")
            plt.plot(Q_early, label="Q_early")
            plt.plot(Q_prompt, label="Q_prompt")
            plt.plot(Q_late, label="Q_late")
            plt.plot(phase_errors, label="Phase errors")
            plt.legend()
            print(f'\tQ_prompt {Q_prompt}')
            print(f'\tI_prompt {I_prompt}')
            plt.show()
        if False:
            shifted_corr = np.abs(correlation ** 2)
            correlation_ax.clear()

            correlation_ax.set_ylim([-3, 30])
            correlation_ax.set_xlim([-50, samples_chunk_size])
            correlation_ax.autoscale(False)
            correlation_ax.set_title("Non-coherent correlation magnitude")

            # correlation_ax.plot((correlation.real**2), label="Coherent correlation")
            bits.append(np.sign(max(correlation, key=abs)))
        correlations.append(correlation ** 2)
        # bits_ax.plot(bits)
        # bits_ax.set_title("Correlation signs (bits)")
        # plt.pause(0.0000001)
    print(f'Got {len(correlations)} ms correlations...')

    plt.ioff()

    confidence_scores = []
    for roll in range(0, 20):
        print(f'*** Trying roll {roll}')
        # rolled_correlations = np.roll(np.array(correlations), roll)
        phase_shifted_correlations = correlations[roll:]

        confidences = []
        for twenty_corrs in chunks(phase_shifted_correlations, 20):
            # for i, corr in enumerate(twenty_corrs):
            #    plt.plot(corr, label=f"corr {i}")
            full_sequence = np.concatenate(twenty_corrs)
            integrated_value = np.sum(full_sequence.real)
            # plt.plot(full_sequence.real)
            # plt.show()
            # print(integrated_value)
            # bit_value = np.sign(integrated_value)
            # bits.append(bit_value)
            confidences.append(abs(integrated_value))
        # Compute an overall confidence score for this offset
        confidence_scores.append(np.mean(confidences))

        # print(bits)
        # print(confidences)
        # plt.plot(bits, label="bits")
        # plt.plot(confidences, label="confidences")
        # plt.show()
        # plt.pause(0.1)
    print(f'Confidence scores: {confidence_scores}')
    best_offset = np.argmax(confidence_scores)
    print(f"Best Offset: {best_offset} ({confidence_scores[best_offset]})")
    plt.ioff()
    plt.cla()
    plt.clf()
    plt.plot(confidence_scores, label="Confidence score for bit phase")
    plt.legend()
    plt.show()

    bit_phase = best_offset
    phase_shifted_correlations = correlations[bit_phase:]
    bits = []
    for twenty_corrs in chunks(phase_shifted_correlations, 20):
        full_sequence = np.concatenate(twenty_corrs)
        # plt.plot(full_sequence)
        # plt.show()
        integrated_value = np.sum(full_sequence.real)
        bit_value = np.sign(integrated_value)
        bits.append(bit_value)

    digital_bits = [1 if b == 1.0 else 0 for b in bits]
    inverted_bits = [0 if b == 1.0 else 1 for b in bits]
    print(f'Bit count: {len(digital_bits)}')
    print(f'Bits:          {digital_bits}')
    print(f'Inverted bits: {inverted_bits}')

    preamble = [1, 0, 0, 0, 1, 0, 1, 1]
    print(f'Preamble {preamble} found in bits? {contains(preamble, digital_bits)}')
    print(f'Preamble {preamble} found in inverted bits? {contains(preamble, inverted_bits)}')
    if False:
        plt.ioff()
        plt.cla()
        plt.clf()
        plt.plot(digital_bits, label="Navigation message bits")
        plt.legend()
        plt.show()

    def get_matches(l, sub):
        return [l[pos:pos + len(sub)] == sub for pos in range(0, len(l) - len(sub) + 1)]

    preamble_starts_in_digital_bits = (
    [x[0] for x in (np.argwhere(np.array(get_matches(digital_bits, preamble)) == True))])
    print(f'Preamble starts in bits:          {preamble_starts_in_digital_bits}')
    from itertools import pairwise
    for (i, j) in pairwise(preamble_starts_in_digital_bits):
        diff = j - i
        print(f'\tDiff from {j} to {i}: {diff}')
    plt.plot([1 if x in preamble_starts_in_digital_bits else 0 for x in range(len(digital_bits))],
             label="Preambles in upright bits")

    preamble_starts_in_inverted_bits = (
    [x[0] for x in (np.argwhere(np.array(get_matches(inverted_bits, preamble)) == True))])
    print(f'Preamble starts in inverted bits: {preamble_starts_in_inverted_bits}')
    for (i, j) in pairwise(preamble_starts_in_inverted_bits):
        diff = j - i
        print(f'\tDiff from {j} to {i}: {diff}')
    plt.plot([1 if x in preamble_starts_in_inverted_bits else 0 for x in range(len(digital_bits))],
             label="Preambles in inverted bits")
    plt.legend()
    plt.show()

    all_peaks = []
    for twenty_corrs in chunks(phase_shifted_correlations, 20):
        full_sequence = np.concatenate(twenty_corrs)
        peaks = np.argwhere(np.abs(full_sequence) > 16)
        # plt.plot(full_sequence[peaks])
        all_peaks.extend(np.real(full_sequence[peaks]))
        # plt.plot(full_sequence)
    plt.plot(all_peaks)
    plt.title("PRN correlation magnitudes")
    plt.show()

    if False:
        # Compute the correlation
        padded_preamble = np.pad(preamble, len(digital_bits) - len(preamble))
        preamble_correlation = np.correlate(digital_bits, preamble, mode='full')
        inverted_preamble_correlation = np.correlate(inverted_bits, preamble, mode='full')
        print(preamble_correlation)

        # Plot the correlation
        plt.autoscale(True)
        plt.plot(preamble_correlation, label="Preamble correlation")
        plt.plot(inverted_preamble_correlation, label="Inverted preamble correlation")
        plt.legend()
        plt.show()


def try_nav_bits_integration():
    nav_message = np.array([1, 1, -1, 1, -1, 1, -1, 1, 1, -1])
    bit_duration = 20  # In samples

    # Generate a signal starting in the middle of a bit
    signal = np.repeat(nav_message, bit_duration)[bit_duration // 2:]

    # Sliding window integration
    integrated_values = np.convolve(signal, np.ones(bit_duration), mode='valid')

    rect_pulse = np.ones(bit_duration)

    # Compute the cross-correlation
    correlation = np.correlate(signal, rect_pulse, mode='valid')

    # Plot
    plt.plot(correlation)
    plt.title("Cross-Correlation with Rectangular Pulse")
    plt.xlabel("Sample Number")
    plt.ylabel("Correlation Value")
    plt.axhline(0, color='red', linestyle='--')
    plt.show()

    # Plot
    plt.plot(signal)
    plt.plot(integrated_values)
    plt.title("Sliding Window Integration")
    plt.xlabel("Sample Number")
    plt.ylabel("Integrated Value")
    plt.axhline(0, color='red', linestyle='--')
    plt.show()
    import sys
    sys.exit(0)


def contains(subseq, inseq):
    return any(inseq[pos:pos + len(subseq)] == subseq for pos in range(0, len(inseq) - len(subseq) + 1))


def main_decode_bits():
    # **** 1 second ***** (1,000 PRN transmissions)
    bits = [1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0,
            1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # **** 6 seconds ***** (12,000 PRN transmissions)
    bits = [1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0,
            1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
            1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1,
            0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
            1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
            0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0,
            1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1]
    # **** 12 seconds ***** (24,000 PRN transmissions)
    bits2 = [1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0,
             1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
             1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1,
             0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
             1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
             0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0,
             1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0,
             1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0,
             0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
             0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
             1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1,
             1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1,
             0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    bits = [1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
            0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1,
            1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
            1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1,
            1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1,
            1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1,
            0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
            1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1,
            0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1,
            0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
            1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1,
            1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0,
            1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1,
            1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    bits = [1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
            0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1,
            1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
            1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1,
            1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1,
            1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1,
            0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
            1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1,
            0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1,
            0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
            1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1,
            1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0,
            1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1,
            1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    bits_with_tracking = [1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
                          1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1,
                          1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0,
                          1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
                          1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1,
                          0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
                          0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0,
                          1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0,
                          1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1,
                          0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1,
                          1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0,
                          1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1,
                          1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1,
                          0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1,
                          1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1,
                          1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
                          1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1,
                          1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1,
                          1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    bits_with_tracking = [1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
                          1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
                          0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1,
                          1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0,
                          1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
                          1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1,
                          0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0,
                          1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1,
                          0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1,
                          1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0,
                          0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
                          0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
                          0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0,
                          0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1,
                          0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0,
                          1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                          0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0,
                          0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0,
                          0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0,
                          0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0,
                          1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0,
                          0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                          1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                          1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1,
                          1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0,
                          1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0,
                          1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1,
                          0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0,
                          1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0,
                          1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0,
                          0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0]

    print(len(bits))

    # preamble = [1, 0, 0, 0, 1, 0, 1, 1]
    preamble = [1, 0, 0, 0, 1, 0, 1, 1]
    inverted_preamble = [0, 1, 1, 1, 0, 1, 0, 0]

    print(contains(preamble, bits))
    print(contains(inverted_preamble, bits))
    # preamble =  [0, 1, 1, 1, 0, 1, 0, 0]


def main_try_manual_adjustment():
    from matplotlib.widgets import Slider

    # Create the figure and the line that we will manipulate
    fig, axes = plt.subplots(nrows=2)
    # line, = ax.plot(t, f(t, init_amplitude, init_frequency), lw=2)
    axes[0].set_xlabel('Time [s]')
    axes[0].set_ylabel('PRN correlation strength')
    # adjust the main plot to make room for the sliders
    fig.subplots_adjust(bottom=0.4)

    # Make a horizontal slider to control the frequency.
    axalpha = fig.add_axes([0.25, 0.35, 0.65, 0.03])
    axbeta = fig.add_axes([0.25, 0.25, 0.65, 0.03])
    axfreq = fig.add_axes([0.25, 0.15, 0.65, 0.03])
    axphase = fig.add_axes([0.25, 0.05, 0.65, 0.03])
    freq_slider = Slider(
        ax=axfreq,
        label='Doppler shift nudge',
        valmin=-200,
        valmax=200,
        valinit=0,
    )
    phase_slider = Slider(
        ax=axphase,
        label='Carrier phase nudge',
        # valmin=-2046,
        # valmax=2046,
        valmin=-1,
        valmax=1,
        valinit=0,
    )
    alpha_slider = Slider(
        ax=axalpha,
        label='alpha',
        valmin=0,
        valmax=0.2,
        valinit=0,
    )
    beta_slider = Slider(
        ax=axbeta,
        label='beta',
        valmin=0,
        valmax=0.02,
        valinit=0,
    )

    input_source = INPUT_SOURCES[6]
    sample_rate = input_source.sdr_sample_rate
    antenna_data = get_samples_from_radio_input_source(input_source, sample_rate)
    detector = GpsSatelliteDetector()

    # The function to be called anytime a slider's value changes
    def update(val):
        i_values, q_values, new_correlations = _get_correlations(antenna_data, detector, freq_slider.val,
                                                                 phase_slider.val, alpha_slider.val, beta_slider.val)
        if True:
            axes[0].cla()

            axes[0].plot(i_values, label="I")
            axes[0].plot(q_values, label="Q")
            axes[0].legend()
            axes[0].autoscale(False)
            axes[0].set_ylim([-6, 6])
            axes[0].set_xlim([0, len(i_values)])

        axes[1].cla()
        axes[1].autoscale(True)
        all_peaks = []
        for twenty_corrs in chunks(new_correlations, 20):
            full_sequence = np.concatenate(twenty_corrs)
            peaks = np.argwhere(np.abs(full_sequence) > 8)
            all_peaks.extend(np.real(full_sequence[peaks]))
        axes[1].plot(all_peaks)

        fig.canvas.draw_idle()
        correlations = new_correlations
        print(f'Got {len(correlations)} ms correlations...')

        plt.ioff()

        confidence_scores = []
        for roll in range(0, 20):
            print(f'*** Trying roll {roll}')
            # rolled_correlations = np.roll(np.array(correlations), roll)
            phase_shifted_correlations = correlations[roll:]

            confidences = []
            for twenty_corrs in chunks(phase_shifted_correlations, 20):
                # for i, corr in enumerate(twenty_corrs):
                #    plt.plot(corr, label=f"corr {i}")
                full_sequence = np.concatenate(twenty_corrs)
                integrated_value = np.sum(full_sequence.real)
                # plt.plot(full_sequence.real)
                # plt.show()
                # print(integrated_value)
                # bit_value = np.sign(integrated_value)
                # bits.append(bit_value)
                confidences.append(abs(integrated_value))
            # Compute an overall confidence score for this offset
            confidence_scores.append(np.mean(confidences))

            # print(bits)
            # print(confidences)
            # plt.plot(bits, label="bits")
            # plt.plot(confidences, label="confidences")
            # plt.show()
            # plt.pause(0.1)
        print(f'Confidence scores: {confidence_scores}')
        best_offset = np.argmax(confidence_scores)
        print(f"Best Offset: {best_offset} ({confidence_scores[best_offset]})")
        #plt.ioff()
        #plt.cla()
        #plt.clf()
        #plt.plot(confidence_scores, label="Confidence score for bit phase")
        #plt.legend()
        #plt.show()

        bit_phase = best_offset
        phase_shifted_correlations = correlations[bit_phase:]
        bits = []
        for twenty_corrs in chunks(phase_shifted_correlations, 20):
            full_sequence = np.concatenate(twenty_corrs)
            # plt.plot(full_sequence)
            # plt.show()
            integrated_value = np.sum(full_sequence.real)
            bit_value = np.sign(integrated_value)
            bits.append(bit_value)

        digital_bits = [1 if b == 1.0 else 0 for b in bits]
        inverted_bits = [0 if b == 1.0 else 1 for b in bits]
        print(f'Bit count: {len(digital_bits)}')
        print(f'Bits:          {digital_bits}')
        print(f'Inverted bits: {inverted_bits}')

        preamble = [1, 0, 0, 0, 1, 0, 1, 1]
        print(f'Preamble {preamble} found in bits? {contains(preamble, digital_bits)}')
        print(f'Preamble {preamble} found in inverted bits? {contains(preamble, inverted_bits)}')

        def get_matches(l, sub):
            return [l[pos:pos + len(sub)] == sub for pos in range(0, len(l) - len(sub) + 1)]

        preamble_starts_in_digital_bits = (
            [x[0] for x in (np.argwhere(np.array(get_matches(digital_bits, preamble)) == True))])
        print(f'Preamble starts in bits:          {preamble_starts_in_digital_bits}')
        from itertools import pairwise
        for (i, j) in pairwise(preamble_starts_in_digital_bits):
            diff = j - i
            print(f'\tDiff from {j} to {i}: {diff}')
        #plt.plot([1 if x in preamble_starts_in_digital_bits else 0 for x in range(len(digital_bits))],
        #         label="Preambles in upright bits")

        preamble_starts_in_inverted_bits = (
            [x[0] for x in (np.argwhere(np.array(get_matches(inverted_bits, preamble)) == True))])
        print(f'Preamble starts in inverted bits: {preamble_starts_in_inverted_bits}')
        for (i, j) in pairwise(preamble_starts_in_inverted_bits):
            diff = j - i
            print(f'\tDiff from {j} to {i}: {diff}')
        #plt.plot([1 if x in preamble_starts_in_inverted_bits else 0 for x in range(len(digital_bits))],
        #         label="Preambles in inverted bits")
        #plt.legend()
        #plt.show()

        all_peaks = []
        for twenty_corrs in chunks(phase_shifted_correlations, 20):
            full_sequence = np.concatenate(twenty_corrs)
            peaks = np.argwhere(np.abs(full_sequence) > 20)
            # plt.plot(full_sequence[peaks])
            all_peaks.extend(np.real(full_sequence[peaks]))
            # plt.plot(full_sequence)
        #plt.plot(all_peaks)
        #plt.title("PRN correlation magnitudes")
        #plt.show()

    # register the update function with each slider
    freq_slider.on_changed(update)
    phase_slider.on_changed(update)
    alpha_slider.on_changed(update)
    beta_slider.on_changed(update)
    update(0)

    plt.show()


def _get_correlations2(antenna_data: np.ndarray, satellite_detector: GpsSatelliteDetector,
                      doppler_shift_adjustment: float, carrier_phase_adjustment: float, alpha, beta) -> list[np.ndarray]:
    # print(f'Length of antenna data: {len(antenna_data)}')
    acquired_satellite_info = DetectedSatelliteInfo(satellite_id=GpsSatelliteId(id=5), doppler_frequency_shift=0,
                                                    time_offset=-0.0325, chip_offset=-33247.5, sample_offset=68541)
    satellite_prn = satellite_detector.satellites_by_id[acquired_satellite_info.satellite_id].prn_as_complex

    samples_chunk_size = SAMPLES_PER_PRN_TRANSMISSION
    satellite_prn_fft = np.fft.fft(satellite_prn)

    doppler_shift = 362.1
    first_ms = antenna_data[:samples_chunk_size]
    prn_phase_shift = get_prn_phase_offset(
        get_correlation_with_doppler_shift(
            satellite_detector,
            first_ms,
            satellite_prn_fft,
            doppler_shift,
        )
    )

    correlations = []
    one_ms = np.arange(0, SAMPLES_PER_PRN_TRANSMISSION)
    # alpha = 0.0008  # Proportional gain
    # beta = 0.0001 # Integral gain
    #alpha = 0.00025  # Proportional gain
    #beta =  0.00005  # Integral gain
    #alpha = 0.132  # Proportional gain
    #beta =  0.00932  # Integral gain
    error_integral = 0

    BANDWIDTH_DLL = 6
    EARLY_LATE_SPACING = 1
    rolled_prn = np.roll(satellite_prn, -prn_phase_shift)
    rolled_prn_fft = np.fft.fft(rolled_prn)
    i_values = []
    q_values = []

    carrier_phase = 0
    angular_doppler_shift = 0
    for i, antenna_data_chunk_that_should_contain_one_prn in enumerate(chunks(antenna_data, samples_chunk_size)):
        if i > 200:
            break

        out = np.zeros(samples_chunk_size, dtype=complex)
        local_i_values = []
        local_q_values = []
        for j, sample_value in enumerate(antenna_data_chunk_that_should_contain_one_prn):
            adjusted_sample = sample_value * np.exp(-1j * carrier_phase)
            local_i_values.append(adjusted_sample.real)
            local_q_values.append(adjusted_sample.imag)
            error = np.real(adjusted_sample) * np.imag(adjusted_sample)
            angular_doppler_shift += (beta * error)
            doppler_shift_hz = angular_doppler_shift * SAMPLES_PER_SECOND / (2.*np.pi)
            carrier_phase += angular_doppler_shift + (alpha * error)
            #print(f'Doppler shift {angular_doppler_shift} Carrier phase {carrier_phase}')
            carrier_phase %= 2*np.pi

            out[j] = adjusted_sample

        #time_domain = np.arange(samples_chunk_size) / SAMPLES_PER_SECOND
        #doppler_shift_hz = angular_doppler_shift * samples_chunk_size / (2.*np.pi)
        #doppler_shift_hz = angular_doppler_shift / (2.*np.pi)
        #print(f'{i} Doppler shift {doppler_shift_hz}')
        #i_components = np.cos((2. * np.pi * time_domain * doppler_shift_hz) + (carrier_phase * 2. * np.pi))
        #q_components = np.sin((2. * np.pi * time_domain * doppler_shift_hz) + (carrier_phase * 2. * np.pi))
        #doppler_shifted_carrier = np.array([complex(i, q) for i, q in zip(i_components, q_components)])
        #doppler_shifted_data_fft = np.fft.fft(antenna_data_chunk_that_should_contain_one_prn * doppler_shifted_carrier)
        doppler_shifted_data_fft = np.fft.fft(out)
        # Amplify by the PRN
        product_fft = rolled_prn_fft * np.conjugate(doppler_shifted_data_fft)
        # Compute inverse FFT to obtain correlation result for this chunk
        correlation = np.fft.ifft(product_fft)
        #plt.plot(correlation)
        #plt.show()
        correlations.append(correlation**2)
        i_values.append(np.sum(local_i_values))
        q_values.append(np.sum(local_q_values))

    return i_values, q_values, correlations

    for i, antenna_data_chunk_that_should_contain_one_prn in enumerate(chunks(antenna_data, samples_chunk_size)):
        if i > 6000:
            break
        #used_doppler_shift = doppler_shift + doppler_shift_adjustment

        # time_domain = np.arange(samples_chunk_size) / SAMPLES_PER_SECOND
        #used_carrier_phase = carrier_phase + carrier_phase_adjustment
        used_doppler_shift = doppler_shift
        used_carrier_phase = carrier_phase
        # local_carrier_cos = np.cos((2.0 * np.pi * (used_doppler_shift) * time_domain) + (used_carrier_phase * 2.0 * np.pi))
        # local_carrier_sin = np.sin((2.0 * np.pi * (used_doppler_shift) * time_domain) + (used_carrier_phase * 2.0 * np.pi))

        # Generate the local oscillator
        time_domain = np.arange(samples_chunk_size) / SAMPLES_PER_SECOND
        i_components = np.cos((2. * np.pi * time_domain * used_doppler_shift) + (used_carrier_phase * 2. * np.pi))
        q_components = np.sin((2. * np.pi * time_domain * used_doppler_shift) + (used_carrier_phase * 2. * np.pi))
        doppler_shifted_carrier = np.array([complex(i, q) for i, q in zip(i_components, q_components)])

        I = np.sum(antenna_data_chunk_that_should_contain_one_prn.real * rolled_prn.real * doppler_shifted_carrier.real)
        Q = np.sum(antenna_data_chunk_that_should_contain_one_prn.real * rolled_prn.real * doppler_shifted_carrier.imag)

        if True:
            I = np.sum(
                (antenna_data_chunk_that_should_contain_one_prn.real * rolled_prn.real) * doppler_shifted_carrier.real -
                (antenna_data_chunk_that_should_contain_one_prn.imag * rolled_prn.real) * doppler_shifted_carrier.imag
            )

            # Mixing with the sine (Q component)
            Q = np.sum(
                (antenna_data_chunk_that_should_contain_one_prn.real * rolled_prn.real) * doppler_shifted_carrier.imag +
               (antenna_data_chunk_that_should_contain_one_prn.imag * rolled_prn.real) * doppler_shifted_carrier.real
            )
            I = np.sum(antenna_data_chunk_that_should_contain_one_prn.real * rolled_prn * doppler_shifted_carrier.real -
                       antenna_data_chunk_that_should_contain_one_prn.imag * rolled_prn * doppler_shifted_carrier.imag)

            Q = np.sum(antenna_data_chunk_that_should_contain_one_prn.real * rolled_prn * doppler_shifted_carrier.imag +
                       antenna_data_chunk_that_should_contain_one_prn.imag * rolled_prn * doppler_shifted_carrier.real)

        # I = np.sum(antenna_data_chunk_that_should_contain_one_prn * rolled_prn * local_carrier_cos).real
        # Q = np.sum(antenna_data_chunk_that_should_contain_one_prn * rolled_prn * local_carrier_sin).real
        i_values.append(I)
        q_values.append(Q)
        # Update error values
        #phase_error = I * Q
        #phase_error = np.arctan2(Q, I)

        # correlation = get_correlation_with_doppler_shift(
        #    satellite_detector,
        #    antenna_data_chunk_that_should_contain_one_prn,
        #    rolled_prn_fft,
        #    used_doppler_shift,
        # )
        doppler_shifted_data_fft = np.fft.fft(antenna_data_chunk_that_should_contain_one_prn * doppler_shifted_carrier)
        # Amplify by the PRN
        product_fft = rolled_prn_fft * np.conjugate(doppler_shifted_data_fft)
        # Compute inverse FFT to obtain correlation result for this chunk
        correlation = np.fft.ifft(product_fft)
        #plt.plot(correlation)
        #plt.show()

        error_integral += phase_error
        #filtered_error = alpha * phase_error + beta * error_integral
        # Generate the control signals
        phase_adjustment = alpha * phase_error
        frequency_adjustment = beta * error_integral

        carrier_phase += phase_adjustment
        doppler_shift += frequency_adjustment

        #doppler_shift -= alpha * phase_error
        #carrier_phase -= beta * phase_error
        #doppler_shift += filtered_error
        #carrier_phase += filtered_error
        carrier_phase %= (2. * np.pi)

        # doppler_shift += filtered_error
        # code_phase_error = (I_late - I_early) - (Q_late - Q_early)
        # code_phase += BANDWIDTH_DLL * code_phase_error

        #print(f'ms={i} Doppler = {used_doppler_shift:.2f} ({doppler_shift:.2f}), Carrier phase {carrier_phase:.2f}rad')

        correlations.append(correlation ** 2)

    return i_values, q_values, correlations


def _get_correlations(antenna_data: np.ndarray, satellite_detector: GpsSatelliteDetector,
                       doppler_shift_adjustment: float, carrier_phase_adjustment: float, alpha, beta) -> list[np.ndarray]:
    acquired_satellite_info = DetectedSatelliteInfo(satellite_id=GpsSatelliteId(id=5), doppler_frequency_shift=0,
                                                    time_offset=-0.0325, chip_offset=-33247.5, sample_offset=68541)
    satellite_prn = satellite_detector.satellites_by_id[acquired_satellite_info.satellite_id].prn_as_complex

    samples_chunk_size = SAMPLES_PER_PRN_TRANSMISSION
    satellite_prn_fft = np.fft.fft(satellite_prn)

    doppler_shift = 362.1
    prn_phase_shift = 1022
    rolled_prn = np.roll(satellite_prn, -prn_phase_shift)
    rolled_prn_fft = np.fft.fft(rolled_prn)

    correlations = []
    error_integrator = 0
    i_values = []
    q_values = []

    carrier_phase = 0
    angular_doppler_shift = 0

    time_domain = np.array(range(0,samples_chunk_size))/SAMPLES_PER_SECOND
    for i, antenna_data_that_should_contain_one_prn in enumerate(chunks(antenna_data, samples_chunk_size)):
        if i > 2000:
            break
        doppler_shifted_carrier_arg = ((doppler_shift * 2.0 * np.pi) * time_domain) + carrier_phase
        doppler_shifted_cos = np.cos(doppler_shifted_carrier_arg)
        doppler_shifted_sin = np.sin(doppler_shifted_carrier_arg)
        i_baseband = antenna_data_that_should_contain_one_prn * doppler_shifted_cos
        q_baseband = antenna_data_that_should_contain_one_prn * doppler_shifted_sin
        #antenna_data_modulated_with_doppler_shift = antenna_data_that_should_contain_one_prn * np.exp(-1j * carrier_phase)
        #local_i_values = antenna_data_modulated_with_doppler_shift.real
        #local_q_values = antenna_data_modulated_with_doppler_shift.imag
        i = np.sum(i_baseband * rolled_prn)
        q = np.sum(q_baseband * rolled_prn)
        phase_error = np.arctan(q / i) / (2.0 * np.pi)
        #carrNco = oldCarrNco + coeffCar1 * (carrError - oldCarrError) + carrError * coeffCar2
        #oldCarrNco   = carrNco
        #oldCarrError = carrError
        
        plt.plot(local_i_values)
        plt.plot(local_q_values)
        plt.show()
        i_values.append(np.sum(local_i_values))
        q_values.append(np.sum(local_q_values))
        phase_error = np.sum(local_i_values * local_q_values)
        error_integrator += phase_error
        # Generate the control signals
        phase_adjustment = alpha * phase_error
        frequency_adjustment = beta * error_integrator

        doppler_shift += frequency_adjustment
        carrier_phase += phase_adjustment
        carrier_phase %= (2. * np.pi)
        print(f'doppler {doppler_shift:.2f} carrier {carrier_phase:.2f}')

        #ji_components = np.cos((2. * np.pi * time_domain * used_doppler_shift) + (used_carrier_phase * 2. * np.pi))
        #q_components = np.sin((2. * np.pi * time_domain * used_doppler_shift) + (used_carrier_phase * 2. * np.pi))
        #doppler_shifted_carrier = np.array([complex(i, q) for i, q in zip(i_components, q_components)])

        doppler_shifted_data_fft = np.fft.fft(antenna_data_modulated_with_doppler_shift)
        # Amplify by the PRN
        product_fft = rolled_prn_fft * np.conjugate(doppler_shifted_data_fft)
        # Compute inverse FFT to obtain correlation result for this chunk
        correlation = np.fft.ifft(product_fft)
        correlations.append(correlation ** 2)
    return i_values, q_values, correlations

    for i, antenna_data_chunk_that_should_contain_one_prn in enumerate(chunks(antenna_data, samples_chunk_size)):
        out = np.zeros(samples_chunk_size, dtype=complex)
        local_i_values = []
        local_q_values = []
        for j, sample_value in enumerate(antenna_data_chunk_that_should_contain_one_prn):
            adjusted_sample = sample_value * np.exp(-1j * carrier_phase)
            local_i_values.append(adjusted_sample.real)
            local_q_values.append(adjusted_sample.imag)
            error = np.real(adjusted_sample) * np.imag(adjusted_sample)
            angular_doppler_shift += (beta * error)
            doppler_shift_hz = angular_doppler_shift * SAMPLES_PER_SECOND / (2.*np.pi)
            carrier_phase += angular_doppler_shift + (alpha * error)
            #print(f'Doppler shift {angular_doppler_shift} Carrier phase {carrier_phase}')
            carrier_phase %= 2*np.pi

            out[j] = adjusted_sample

        #time_domain = np.arange(samples_chunk_size) / SAMPLES_PER_SECOND
        #doppler_shift_hz = angular_doppler_shift * samples_chunk_size / (2.*np.pi)
        #doppler_shift_hz = angular_doppler_shift / (2.*np.pi)
        #print(f'{i} Doppler shift {doppler_shift_hz}')
        #i_components = np.cos((2. * np.pi * time_domain * doppler_shift_hz) + (carrier_phase * 2. * np.pi))
        #q_components = np.sin((2. * np.pi * time_domain * doppler_shift_hz) + (carrier_phase * 2. * np.pi))
        #doppler_shifted_carrier = np.array([complex(i, q) for i, q in zip(i_components, q_components)])
        #doppler_shifted_data_fft = np.fft.fft(antenna_data_chunk_that_should_contain_one_prn * doppler_shifted_carrier)
        doppler_shifted_data_fft = np.fft.fft(out)
        # Amplify by the PRN
        product_fft = rolled_prn_fft * np.conjugate(doppler_shifted_data_fft)
        # Compute inverse FFT to obtain correlation result for this chunk
        correlation = np.fft.ifft(product_fft)
        #plt.plot(correlation)
        #plt.show()
        correlations.append(correlation**2)
        i_values.append(np.sum(local_i_values))
        q_values.append(np.sum(local_q_values))

    return i_values, q_values, correlations

    for i, antenna_data_chunk_that_should_contain_one_prn in enumerate(chunks(antenna_data, samples_chunk_size)):
        if i > 6000:
            break
        #used_doppler_shift = doppler_shift + doppler_shift_adjustment

        # time_domain = np.arange(samples_chunk_size) / SAMPLES_PER_SECOND
        #used_carrier_phase = carrier_phase + carrier_phase_adjustment
        used_doppler_shift = doppler_shift
        used_carrier_phase = carrier_phase
        # local_carrier_cos = np.cos((2.0 * np.pi * (used_doppler_shift) * time_domain) + (used_carrier_phase * 2.0 * np.pi))
        # local_carrier_sin = np.sin((2.0 * np.pi * (used_doppler_shift) * time_domain) + (used_carrier_phase * 2.0 * np.pi))

        # Generate the local oscillator
        time_domain = np.arange(samples_chunk_size) / SAMPLES_PER_SECOND
        i_components = np.cos((2. * np.pi * time_domain * used_doppler_shift) + (used_carrier_phase * 2. * np.pi))
        q_components = np.sin((2. * np.pi * time_domain * used_doppler_shift) + (used_carrier_phase * 2. * np.pi))
        doppler_shifted_carrier = np.array([complex(i, q) for i, q in zip(i_components, q_components)])

        I = np.sum(antenna_data_chunk_that_should_contain_one_prn.real * rolled_prn.real * doppler_shifted_carrier.real)
        Q = np.sum(antenna_data_chunk_that_should_contain_one_prn.real * rolled_prn.real * doppler_shifted_carrier.imag)

        if True:
            I = np.sum(
                (antenna_data_chunk_that_should_contain_one_prn.real * rolled_prn.real) * doppler_shifted_carrier.real -
                (antenna_data_chunk_that_should_contain_one_prn.imag * rolled_prn.real) * doppler_shifted_carrier.imag
            )

            # Mixing with the sine (Q component)
            Q = np.sum(
                (antenna_data_chunk_that_should_contain_one_prn.real * rolled_prn.real) * doppler_shifted_carrier.imag +
                (antenna_data_chunk_that_should_contain_one_prn.imag * rolled_prn.real) * doppler_shifted_carrier.real
            )
            I = np.sum(antenna_data_chunk_that_should_contain_one_prn.real * rolled_prn * doppler_shifted_carrier.real -
                       antenna_data_chunk_that_should_contain_one_prn.imag * rolled_prn * doppler_shifted_carrier.imag)

            Q = np.sum(antenna_data_chunk_that_should_contain_one_prn.real * rolled_prn * doppler_shifted_carrier.imag +
                       antenna_data_chunk_that_should_contain_one_prn.imag * rolled_prn * doppler_shifted_carrier.real)

        # I = np.sum(antenna_data_chunk_that_should_contain_one_prn * rolled_prn * local_carrier_cos).real
        # Q = np.sum(antenna_data_chunk_that_should_contain_one_prn * rolled_prn * local_carrier_sin).real
        i_values.append(I)
        q_values.append(Q)
        # Update error values
        #phase_error = I * Q
        #phase_error = np.arctan2(Q, I)

        # correlation = get_correlation_with_doppler_shift(
        #    satellite_detector,
        #    antenna_data_chunk_that_should_contain_one_prn,
        #    rolled_prn_fft,
        #    used_doppler_shift,
        # )
        doppler_shifted_data_fft = np.fft.fft(antenna_data_chunk_that_should_contain_one_prn * doppler_shifted_carrier)
        # Amplify by the PRN
        product_fft = rolled_prn_fft * np.conjugate(doppler_shifted_data_fft)
        # Compute inverse FFT to obtain correlation result for this chunk
        correlation = np.fft.ifft(product_fft)
        #plt.plot(correlation)
        #plt.show()

        error_integral += phase_error
        #filtered_error = alpha * phase_error + beta * error_integral
        # Generate the control signals
        phase_adjustment = alpha * phase_error
        frequency_adjustment = beta * error_integral

        carrier_phase += phase_adjustment
        doppler_shift += frequency_adjustment

        #doppler_shift -= alpha * phase_error
        #carrier_phase -= beta * phase_error
        #doppler_shift += filtered_error
        #carrier_phase += filtered_error
        carrier_phase %= (2. * np.pi)

        # doppler_shift += filtered_error
        # code_phase_error = (I_late - I_early) - (Q_late - Q_early)
        # code_phase += BANDWIDTH_DLL * code_phase_error

        #print(f'ms={i} Doppler = {used_doppler_shift:.2f} ({doppler_shift:.2f}), Carrier phase {carrier_phase:.2f}rad')

        correlations.append(correlation ** 2)

    return i_values, q_values, correlations


if __name__ == '__main__':
    # main()
    # main_test()
    # main_try_doppler_track()
    # main_try_costas_loop()
    # main_try_costas_loop2()
    # main_try_iq_power_tracking()
    main_try_manual_adjustment()
    # main_decode_bits()
    # main_run_satellite_detection()
    # main_decode_nav_bits()
