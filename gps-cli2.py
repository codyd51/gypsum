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
def generate_cosine_with_frequency(num_samples: int, frequency: int) -> list[complex]:
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

        #self.doppler_wave_generator_memoizer = numpy_memoizer(generate_cosine_with_frequency)
        #self.doppler_wave_generator_memoizer = Memoized(generate_cosine_with_frequency)

    def detect_satellites_in_data(self, antenna_data: np.ndarray) -> dict[GpsSatelliteId, DetectedSatelliteInfo]:
        # PT: Currently, this expects the provided antenna data to represent exactly one second of sampling
        start_time = 0
        #end_time = 1
        time = len(antenna_data) / SAMPLES_PER_SECOND
        end_time = time
        print(f'detected time {time}')
        time_domain = np.arange(start_time, end_time, 1/(float(SAMPLES_PER_SECOND)))
        # We're going to read PRN_CORRELATION_CYCLE_COUNT repetitions of the PRN
        correlation_bucket_sample_count = int(PRN_CORRELATION_CYCLE_COUNT * SAMPLES_PER_PRN_TRANSMISSION)
        print(f'Correlation bucket sample count: {correlation_bucket_sample_count}')
        time_domain_for_correlation_bucket = time_domain[:correlation_bucket_sample_count]

        # Throw away extra signal data that doesn't fit in a modulo of our window size
        trimmed_antenna_data = antenna_data[:round_to_previous_multiple_of(len(antenna_data), correlation_bucket_sample_count)]

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
        antenna_data_chunks = list(chunks(trimmed_antenna_data, correlation_bucket_sample_count, step=SAMPLES_PER_SECOND//4))
        #antenna_data = list(trimmed_antenna_data[::SAMPLES_PER_SECOND])
        for i, antenna_data_chunk in tqdm(list(enumerate(antenna_data_chunks))):
            #print(f'Searching bucket {i}/{len(antenna_data_chunks)}')
            for satellite_id in ALL_SATELLITE_IDS:
                # Only necessary to look for a satellite if we haven't already detected it
                if satellite_id in detected_satellites_by_id:
                    #print(f'Will not search for satellite {satellite_id.id} again because we\'ve already identified it')
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
                #print(f"Finished searching this correlation bucket, but we've only detected {len(detected_satellites_by_id)} satellites. Moving on to the next bucket...")
                pass
        return detected_satellites_by_id

    def _compute_correlation2(
        self,
        antenna_data: np.ndarray,
        doppler_shifted_carrier_wave: np.ndarray,
        prn_fft: np.ndarray,
    ) -> np.ndarray:
        correlation_sample_count = len(antenna_data)
        # Multiply the input signal with our Doppler-shifted cosine wave,
        # to align the input signal with our reference PRN.
        # In other words, this multiplication aligns our
        # received data to baseband (if the Doppler shift is correct).
        antenna_data_multiplied_with_doppler_shifted_carrier = antenna_data * doppler_shifted_carrier_wave
        #fft_of_doppler_shifted_signal = np.fft.fft(antenna_data_multiplied_with_doppler_shifted_carrier)
        # Perform BPSK demodulation
        #threshold = 0.0  # Adjust the threshold based on the signal characteristics
        #demodulated_bits = (np.real(antenna_data_multiplied_with_doppler_shifted_carrier) > threshold).astype(int) * 2 - 1
        fft_of_demodulated_doppler_shifted_signal = np.fft.fft(antenna_data_multiplied_with_doppler_shifted_carrier)

        mult = prn_fft * np.conjugate(fft_of_demodulated_doppler_shifted_signal)
        mult_ifft = np.fft.ifft(mult)
        scaled_ifft = mult_ifft * ((1 / correlation_sample_count) * correlation_sample_count)
        correlation = np.absolute(scaled_ifft)
        return correlation

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
        #correlation_sign = np.sign(normalized_correlation.real) + 1j * np.sign(normalized_correlation.imag)

        # Compute the magnitude and phase of the correlation result
        #correlation_magnitude = np.abs(correlation_sign)
        #correlation_phase = np.angle(correlation_sign)
        #print(f'Correlation magnitude: {correlation_magnitude}')
        #print(f'Correlation phase: {correlation_phase}')

        #plt.plot(correlation_magnitude)
        #plt.show()
        #plt.plot(correlation_phase)
        #plt.show()

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
            #print(f'Sat {satellite_id.id} shift {doppler_shift} corr min {np.min(correlation)} corr max {np.max(correlation)}')

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
                #print(f'We are {chip_offset_from_satellite} chips ahead of satellite {satellite_id}. This represents a time delay of {time_offset}')
                print(f'*** Identified satellite {satellite_id} at doppler shift {doppler_shift}, correlation magnitude of {correlation[sample_offset_where_we_started_receiving_prn]} at {sample_offset_where_we_started_receiving_prn}, time offset of {time_offset}, chip offset of {chip_offset_from_satellite}')
                print(f'*** SNR: {snr_ratio}')
                plt.plot(correlation.real)
                plt.show()

                # PT: It seems as though we don't yet have enough information to determine
                # whether the satellite's clock is ahead of or behind our own (which makes sense).
                # All we can say for now is that the received PRN is out of phase with our PRN
                # by some number of chips/some time delay (which we can choose to be either positive or negative).
                # It seems like the next step now is to use the delay to decode the navigation message, and figure
                # out later our time differential.
                a= DetectedSatelliteInfo(
                    satellite_id=satellite_id,
                    doppler_frequency_shift=doppler_shift,
                    time_offset=time_offset,
                    chip_offset=chip_offset_from_satellite,
                    sample_offset=int(sample_offset_where_we_started_receiving_prn),
                )
                print(a)

        # Failed to find a PRN correlation peak strong enough to count as a detection
        return None


def main_run_satellite_detection():
    input_source = INPUT_SOURCES[6]
    sample_rate = input_source.sdr_sample_rate

    sdr_data = get_samples_from_radio_input_source(input_source, sample_rate)
    sdr_data = sdr_data / np.max(np.abs(sdr_data))

    if False:
        cutoff_frequency = 2e6  # Set the cutoff frequency according to your signal bandwidth
        print(sample_rate)
        nyquist = 0.5 * sample_rate  # Nyquist frequency for the decimated sample rate
        normal_cutoff = cutoff_frequency / nyquist
        print(normal_cutoff)
        b, a = butter(6, 0.9, btype='low', analog=False)

        # Apply the filter to the samples
        sdr_data = filtfilt(b, a, sdr_data)

    satellite_detector = GpsSatelliteDetector()
    subset_of_antenna_data_for_satellite_detection = sdr_data
    detected_satellites = satellite_detector.detect_satellites_in_data(subset_of_antenna_data_for_satellite_detection)
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
        detected_satellites = satellite_detector.detect_satellites_in_data(subset_of_antenna_data_for_satellite_detection)
    else:
        # With BPSK demodulation
        detected_satellites = {
            GpsSatelliteId(id=10): DetectedSatelliteInfo(satellite_id=GpsSatelliteId(id=10), doppler_frequency_shift=-2500, time_offset=0.0002785923753665689, chip_offset=285.0, sample_offset=1476),
            GpsSatelliteId(id=24): DetectedSatelliteInfo(satellite_id=GpsSatelliteId(id=24), doppler_frequency_shift=-2500, time_offset=0.00012170087976539589, chip_offset=124.5, sample_offset=1797),
            GpsSatelliteId(id=10): DetectedSatelliteInfo(satellite_id=GpsSatelliteId(id=15), doppler_frequency_shift=500, time_offset=0.0009608993157380253, chip_offset=983.0, sample_offset=80),
            GpsSatelliteId(id=10): DetectedSatelliteInfo(satellite_id=GpsSatelliteId(id=12), doppler_frequency_shift=-5500, time_offset=0.0007038123167155425, chip_offset=720.0, sample_offset=606),
        }
        # Prior to BPSK demodulation
        detected_satellites = {
            GpsSatelliteId(id=24): DetectedSatelliteInfo(satellite_id=GpsSatelliteId(id=24), doppler_frequency_shift=-2500, time_offset=0.00012121212121212121, chip_offset=124.0, sample_offset=1798),
            GpsSatelliteId(id=10): DetectedSatelliteInfo(satellite_id=GpsSatelliteId(id=10), doppler_frequency_shift=-2500, time_offset=0.0002785923753665689, chip_offset=285.0, sample_offset=1476),
            GpsSatelliteId(id=15): DetectedSatelliteInfo(satellite_id=GpsSatelliteId(id=15), doppler_frequency_shift=500, time_offset=0.0009608993157380253, chip_offset=983.0, sample_offset=80),
            GpsSatelliteId(id=12): DetectedSatelliteInfo(satellite_id=GpsSatelliteId(id=12), doppler_frequency_shift=-5500, time_offset=0.0007028347996089931, chip_offset=719.0, sample_offset=608),
        }

    for detected_satellite_id, detected_satellite_info in detected_satellites.items():
        print(detected_satellite_id, detected_satellite_info)

        start_time = 0
        end_time = 3
        time_domain = np.arange(start_time, end_time, 1/(float(SAMPLES_PER_SECOND)))
        # For now just use the same sample count as the other bit of code, just testing
        test_sample_count = len(time_domain)
        test_sample_count = int(PRN_CORRELATION_CYCLE_COUNT * SAMPLES_PER_PRN_TRANSMISSION)
        # 200ms
        # This should give us between 8-10 data bits (depending on whether we were in the middle of a bit transition
        # when we started listening)
        test_sample_count = int(SAMPLES_PER_PRN_TRANSMISSION) * 200
        trimmed_time_domain = time_domain[:test_sample_count]
        # Take a sample of antenna data and multiply it by the computed Doppler and time shift
        #antenna_data = subset_of_antenna_data_for_satellite_detection[:test_sample_count]
        antenna_data = sdr_data[:test_sample_count]

        # Doppler the carrier wave to match the correlation peak
        doppler_shifted_carrier_wave = generate_cosine_with_frequency(trimmed_time_domain, detected_satellite_info.doppler_frequency_shift)
        #antenna_data_multiplied_with_doppler_shifted_carrier = antenna_data * doppler_shifted_carrier_wave

        satellite_prn_fft = satellite_detector.satellites_by_id[detected_satellite_id].fft_of_prn_of_length(test_sample_count)

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

                indexes_of_peaks_above_prn_correlation_threshold = list(sorted(np.argwhere(correlation >= PRN_CORRELATION_MAGNITUDE_THRESHOLD)))
                if len(indexes_of_peaks_above_prn_correlation_threshold):
                    best_sample_offset = indexes_of_peaks_above_prn_correlation_threshold[0][0]
                    print(f'Found new highest peak with doppler shift of {precise_doppler_shift}: {highest_peak} (sample offset {best_sample_offset})')

        print(f'Found best Doppler shift: {best_doppler_shift}')
        precise_carrier_wave = generate_cosine_with_frequency(trimmed_time_domain, best_doppler_shift)
        #antenna_data_with_precise_wave = antenna_data * precise_carrier_wave

        #plt.plot(antenna_data_with_precise_wave)
        #plt.show()
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
        #fft_of_doppler_shifted_signal = np.fft.fft(antenna_data_multiplied_with_doppler_shifted_carrier)
        # Perform BPSK demodulation
        threshold = 0.0  # Adjust the threshold based on the signal characteristics
        demodulated_bits = (np.real(antenna_data_multiplied_with_doppler_shifted_carrier) > threshold).astype(int) * 2 - 1
        fft_of_demodulated_doppler_shifted_signal = np.fft.fft(demodulated_bits)

        mult = prn_fft * np.conjugate(fft_of_demodulated_doppler_shifted_signal)
        mult_ifft = np.fft.ifft(mult)
        scaled_ifft = mult_ifft * ((1 / correlation_sample_count) * correlation_sample_count)
        correlation = np.absolute(scaled_ifft)


        compensated_signal = np.exp(-1j * 2 * np.pi * best_doppler_shift / sample_rate * np.arange(test_sample_count)) * antenna_data
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

        #plt.scatter(trimmed_time_domain, [x.real for x in demodulated])
        #plt.scatter(trimmed_time_domain, [x.imag for x in demodulated])
        plt.scatter(trimmed_time_domain, demodulated)
        plt.show()

        return

        # Multiply them
        demodulated = antenna_data_multiplied_with_doppler_shifted_carrier * trimmed_time_shifted_prn
        plt.plot(trimmed_time_domain, demodulated)
        #plt.plot(antenna_data_multiplied_with_doppler_shifted_carrier)
        #plt.plot(trimmed_time_domain, demodulated)
        plt.show()
        return

        if False:
            satellite_prn_fft = self.satellites_by_id[satellite_id].fft_of_prn_of_length(correlation_bucket_sample_count)
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
    #detected_satellite = DetectedSatelliteInfo(satellite_id=GpsSatelliteId(id=24), doppler_frequency_shift=-2500, time_offset=0.00012121212121212121, chip_offset=124.0, sample_offset=1798)
    detected_satellite = DetectedSatelliteInfo(satellite_id=GpsSatelliteId(id=18), doppler_frequency_shift=-1000, time_offset=-0.0021959921798631477, chip_offset=-2246.5, sample_offset=6539)
    detected_satellite_id = detected_satellite.satellite_id

    start_time = 0
    end_time = 1
    time_domain = np.arange(start_time, end_time, 1/(float(SAMPLES_PER_SECOND)))

    # 200ms
    # This should give us between 8-10 data bits (depending on whether we were in the middle of a bit transition
    # when we started listening)
    test_sample_count = int(SAMPLES_PER_PRN_TRANSMISSION) * 800
    trimmed_time_domain = time_domain[:test_sample_count]
    antenna_data = sdr_data[:test_sample_count]

    satellite_prn_fft = satellite_detector.satellites_by_id[detected_satellite_id].fft_of_prn_of_length(test_sample_count)

    best_doppler_shift = detected_satellite.doppler_frequency_shift
    precise_doppler_shifted_carrier = generate_cosine_with_frequency(trimmed_time_domain, best_doppler_shift)
    correlation = satellite_detector._compute_correlation(
        antenna_data,
        np.array(precise_doppler_shifted_carrier),
        satellite_prn_fft,
    )
    plt.plot(correlation)
    #plt.show()
    phase_offset = np.argmax(np.abs(correlation)) % SAMPLES_PER_PRN_TRANSMISSION

    print(f'Rebasing antenna data phase {phase_offset}, {SAMPLES_PER_PRN_TRANSMISSION - phase_offset}...')
    #aligned_antenna_data = sdr_data[(SAMPLES_PER_PRN_TRANSMISSION - phase_offset):]
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
    #plt.show()

    antenna_data_multiplied_with_doppler_shifted_carrier = antenna_data * np.array(precise_doppler_shifted_carrier)
    #plt.plot(antenna_data_multiplied_with_doppler_shifted_carrier)
    #plt.show()

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

    #plt.plot(bit)
    #plt.plot(smoothed_data)
    #plt.plot(normalized_data)
    #plt.plot(cleaned_data)
    #plt.show()

    bit_length = 2*1023*20  # Length of each BPSK bit in samples
    bits = [binary_signal[i:i+bit_length] for i in range(0, len(binary_signal), bit_length)]
    for i, bit in enumerate(bits):
        smoothed_data = savgol_filter(bit, window_length=100, polyorder=3)
        normalized_data = (smoothed_data - np.mean(smoothed_data)) / np.std(smoothed_data)
        median = np.median(normalized_data)
        std_dev = np.std(normalized_data)
        threshold = median + 2 * std_dev  # Define a threshold (adjust multiplier as needed)
        cleaned_data = np.clip(normalized_data, median - threshold, median + threshold)
        #plt.plot(bit)
        #plt.plot(smoothed_data)
        #plt.plot(normalized_data)
        plt.plot(cleaned_data)
        plt.show()

        print(f'Bit {i}: (len({len(bit)})')
        #print(list(bit))


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
    #return correlation_result[:1-len(prn)]
    return correlation_result


def time_domain_correlation(prn, antenna_data):
    return np.correlate(antenna_data, prn, mode='valid')


def main_decode_nav_bits():
    input_source = INPUT_SOURCES[6]
    sample_rate = input_source.sdr_sample_rate

    sdr_data = get_samples_from_radio_input_source(input_source, sample_rate)
    detected_satellite = DetectedSatelliteInfo(satellite_id=GpsSatelliteId(id=18), doppler_frequency_shift=-1000, time_offset=-0.0021959921798631477, chip_offset=-2246.5, sample_offset=6539)
    detected_satellite_id = detected_satellite.satellite_id

    start_time = 0
    end_time = 1
    time_domain = np.arange(start_time, end_time, 1/(float(SAMPLES_PER_SECOND)))

    satellite_detector = GpsSatelliteDetector()
    best_doppler_shift = detected_satellite.doppler_frequency_shift

    ms_count = 20
    twenty_ms_chunk_size = int(SAMPLES_PER_PRN_TRANSMISSION) * ms_count
    satellite_prn_fft = satellite_detector.satellites_by_id[detected_satellite_id].fft_of_prn_of_length(twenty_ms_chunk_size)
    twenty_ms_time_chunk = time_domain[:twenty_ms_chunk_size]
    precise_doppler_shifted_carrier = generate_cosine_with_frequency(twenty_ms_chunk_size, best_doppler_shift)
    #plt.plot(precise_doppler_shifted_carrier)
    #plt.show()
    #plt.plot(satellite_prn_fft)
    #plt.show()
    #plt.plot(np.fft.ifft(satellite_prn_fft) * twenty_ms_chunk_size)
    #plt.plot(np.tile(np.fft.ifft(satellite_prn_fft), ms_count))
    #plt.show()

    for twenty_ms_antenna_data_chunk in chunks(sdr_data, twenty_ms_chunk_size):
        print(f'Processing 20ms chunk...')
        prn = satellite_detector.satellites_by_id[detected_satellite_id].prn_as_complex
        corr = frequency_domain_correlation(
            prn, twenty_ms_antenna_data_chunk * precise_doppler_shifted_carrier
        )
        #plt.plot(corr)
        #plt.show()

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
            #peak_sign = np.sign(np.max(segment, key=abs))
            peak_value = max(segment, key=abs)
            peak_sign = np.sign(peak_value)
            print(f'i={i} peak={peak_sign}')
            #bit_signs.append(peak_sign)
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

    #plt.plot(correlation)
    #plt.show()
    phase_offset = np.argmax(np.abs(correlation)) % SAMPLES_PER_PRN_TRANSMISSION

    print(f'Rebasing antenna data phase {phase_offset}, {SAMPLES_PER_PRN_TRANSMISSION - phase_offset}...')
    #aligned_antenna_data = sdr_data[(SAMPLES_PER_PRN_TRANSMISSION - phase_offset):]
    aligned_antenna_data = sdr_data[(SAMPLES_PER_PRN_TRANSMISSION - phase_offset):]
    aligned_antenna_data = aligned_antenna_data[:test_sample_count]
    correlation = satellite_detector._compute_correlation(
        aligned_antenna_data,
        np.array(generate_cosine_with_frequency(trimmed_time_domain, best_doppler_shift)),
        satellite_prn_fft,
    )
    plt.plot(correlation)
    plt.show()
    #phase_offset = np.argmax(np.abs(correlation))
    #print(phase_offset)


def main_test():
    input_source = INPUT_SOURCES[6]
    sample_rate = input_source.sdr_sample_rate

    antenna_data = get_samples_from_radio_input_source(input_source, sample_rate)
    #detected_satellite = DetectedSatelliteInfo(satellite_id=GpsSatelliteId(id=18), doppler_frequency_shift=-1000, time_offset=-0.0021959921798631477, chip_offset=-2246.5, sample_offset=6539)
    detected_satellite = DetectedSatelliteInfo(satellite_id=GpsSatelliteId(id=5), doppler_frequency_shift=0, time_offset=-0.0325, chip_offset=-33247.5, sample_offset=68541)
    detected_satellite_id = detected_satellite.satellite_id

    satellite_detector = GpsSatelliteDetector()
    best_doppler_shift = detected_satellite.doppler_frequency_shift

    prn = satellite_detector.satellites_by_id[detected_satellite_id].prn_as_complex
    # Zero-pad the PRN to match a chunk length (e.g., 2046 for direct correlation without overlap)
    chunk_length = 2046*40
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
        chunk = antenna_data[i:i+chunk_length]
        precise_doppler_shifted_carrier = generate_cosine_with_frequency(chunk_length, doppler_shift)
        chunk = chunk * precise_doppler_shifted_carrier
        chunk_fft = np.fft.fft(chunk)

        # Frequency domain multiplication
        product_fft = prn_fft * np.conjugate(chunk_fft)

        # Compute inverse FFT to obtain correlation result for this chunk
        correlation_chunk = np.fft.ifft(product_fft)

        #correlation_results.append(np.abs(correlation_chunk))
        non_coherent_integration = np.abs(correlation_chunk)
        print(np.max(non_coherent_integration))

        plt.plot(non_coherent_integration)
        plt.show()
        #continue

        # Get the index of the peak
        switch = 2
        if switch == 0:
            #peak_index = np.argmax(non_coherent_integration)
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
        #plt.plot(non_coherent_integration)
        plt.plot(peak_sign)
        print(peak_sign)
        plt.show()

    # Convert the list of arrays into a single numpy array
    correlation_results = np.concatenate(correlation_results)


if __name__ == '__main__':
    #main()
    main_test()
    #main_run_satellite_detection()
    #main_decode_nav_bits()
