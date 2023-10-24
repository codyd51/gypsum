from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
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
def generate_cosine_with_frequency(time_domain: np.ndarray, frequency: int) -> list[complex]:
    #print(f'Generating cosine at {frequency}Hz...')
    i_components = np.cos(2. * np.pi * time_domain * frequency)
    q_components = np.sin(2. * np.pi * time_domain * frequency)
    cosine = [i + (1j*q) for (i, q) in zip(i_components, q_components)]
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
        end_time = 1
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
            doppler_shift: generate_cosine_with_frequency(time_domain_for_correlation_bucket, doppler_shift)
            for doppler_shift in range(
                DOPPLER_SHIFT_FREQUENCY_LOWER_BOUND,
                DOPPLER_SHIFT_FREQUENCY_UPPER_BOUND,
                DOPPLER_SHIFT_SEARCH_INTERVAL,
            )
        }
        print(f'Finished precomputing Doppler-shifted carrier waves.')

        detected_satellites_by_id = {}
        antenna_data_chunks = list(chunks(trimmed_antenna_data, correlation_bucket_sample_count))
        for i, antenna_data_chunk in tqdm(list(enumerate(antenna_data_chunks))):
            print(f'Searching bucket {i}/{len(antenna_data_chunks)}')
            for satellite_id in ALL_SATELLITE_IDS:
                # Only necessary to look for a satellite if we haven't already detected it
                if satellite_id in detected_satellites_by_id:
                    print(f'Will not search for satellite {satellite_id.id} again because we\'ve already identified it')
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
                print(f"Finished searching this correlation bucket, but we've only detected {len(detected_satellites_by_id)} satellites. Moving on to the next bucket...")
        return detected_satellites_by_id

    def _compute_correlation(
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
        fft_of_doppler_shifted_signal = np.fft.fft(antenna_data_multiplied_with_doppler_shifted_carrier)

        mult = prn_fft * np.conjugate(fft_of_doppler_shifted_signal)
        mult_ifft = np.fft.ifft(mult)
        scaled_ifft = mult_ifft * ((1 / correlation_sample_count) * correlation_sample_count)
        correlation = np.absolute(scaled_ifft)
        return correlation

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

            indexes_of_peaks_above_prn_correlation_threshold = list(sorted(np.argwhere(correlation >= PRN_CORRELATION_MAGNITUDE_THRESHOLD)))
            # TODO(PT): Instead of immediately returning, we should hold out to find the best correlation across the search space
            if len(indexes_of_peaks_above_prn_correlation_threshold):
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
                print(f'We are {chip_offset_from_satellite} chips ahead of satellite {satellite_id}. This represents a time delay of {time_offset}')
                print(f'*** Identified satellite {satellite_id} at doppler shift {doppler_shift}, correlation magnitude of {correlation[sample_offset_where_we_started_receiving_prn]} at {sample_offset_where_we_started_receiving_prn}, time offset of {time_offset}, chip offset of {chip_offset_from_satellite}')

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


def main():
    input_source = INPUT_SOURCES[5]
    sample_rate = input_source.sdr_sample_rate

    # TODO(PT): When I switch back to live SDR processing, we'll need to ensure that we normalize the actual
    # sample rate coming off the SDR to the sample rate we expect (SAMPLES_PER_SECOND).
    sdr_data = get_samples_from_radio_input_source(input_source, sample_rate)
    # Read 1 second worth of antenna data to search through
    satellite_detector = GpsSatelliteDetector()
    subset_of_antenna_data_for_satellite_detection = sdr_data[:SAMPLES_PER_SECOND]
    #detected_satellites = satellite_detector.detect_satellites_in_data(subset_of_antenna_data_for_satellite_detection)
    if True:
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
        antenna_data_with_precise_wave = antenna_data * precise_carrier_wave

        # Time-shift the PRN to match the offset we detected
        prn = satellite_detector.satellites_by_id[detected_satellite_id].prn_as_complex
        # Just tile a bunch for now
        tiled_prn = np.tile(prn, 6000)
        # Apply time shift
        time_shifted_prn = tiled_prn[best_sample_offset:]
        trimmed_time_shifted_prn = time_shifted_prn[:test_sample_count]

        # Multiply them
        demodulated = antenna_data_with_precise_wave * trimmed_time_shifted_prn
        # Perform BPSK demodulation
        threshold = 0.0  # Adjust the threshold based on the signal characteristics
        demodulated_bits = (np.real(demodulated) > threshold).astype(int) * 2 - 1

        # Output demodulated bits
        print("Demodulated Bits:", [0 if x == -1 else 1 for x in demodulated_bits])

        #plt.scatter(trimmed_time_domain, [x.real for x in demodulated])
        #plt.scatter(trimmed_time_domain, [x.imag for x in demodulated])
        plt.scatter(trimmed_time_domain, demodulated_bits)
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


if __name__ == '__main__':
    main()
