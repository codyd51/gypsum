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
    print(f'Generating cosine at {frequency}Hz...')
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
            # Multiply the input signal with our Doppler-shifted cosine wave,
            # to align the input signal with our reference PRN.
            # In other words, this multiplication aligns our
            # received data to baseband (if the Doppler shift is correct).
            doppler_shifted_carrier_wave = doppler_shifted_carrier_waves[doppler_shift]
            antenna_data_multiplied_with_doppler_shifted_carrier = antenna_data_chunk * doppler_shifted_carrier_wave
            fft_of_doppler_shifted_signal = np.fft.fft(antenna_data_multiplied_with_doppler_shifted_carrier)

            mult = satellite_prn_fft * np.conjugate(fft_of_doppler_shifted_signal)
            mult_ifft = np.fft.ifft(mult)
            scaled_ifft = mult_ifft * ((1 / correlation_bucket_sample_count) * correlation_bucket_sample_count)
            correlation = np.absolute(scaled_ifft)

            #plt.cla()
            #plt.ylim((0, PRN_CORRELATION_CYCLE_COUNT * 10))
            # Correlation across time
            #plt.plot(correlation)
            #print(correlation)
            indexes_of_peaks_above_prn_correlation_threshold = list(sorted(np.argwhere(correlation >= PRN_CORRELATION_MAGNITUDE_THRESHOLD)))
            # TODO(PT): Instead of immediately returning, we should hold out to find the best correlation across the search space
            if len(indexes_of_peaks_above_prn_correlation_threshold):
                # We're matching against many PRN cycles to increase our correlation strength.
                # We now want to figure out the time slide (and therefore distance) for the transmitter.
                # Therefore, we only care about the distance to the first correlation peak (i.e. the phase offset)
                sample_offset_where_we_started_receiving_prn = indexes_of_peaks_above_prn_correlation_threshold[0]
                #  Divide by Nyquist frequency
                chip_index_where_we_started_receiving_prn = sample_offset_where_we_started_receiving_prn / 2
                # Convert to time
                time_per_prn_chip = ((1 / PRN_REPETITIONS_PER_SECOND) / PRN_CHIP_COUNT)
                chip_offset_from_satellite = PRN_CHIP_COUNT - chip_index_where_we_started_receiving_prn
                # TODO(PT): Maybe the sign varies depending on whether the Doppler shift is positive or negative?
                time_offset = chip_offset_from_satellite * time_per_prn_chip
                print(f'We are {chip_offset_from_satellite} chips ahead of satellite {satellite_id}. This represents a time delay of {time_offset}')
                print(f'*** Identified satellite {satellite_id} at doppler shift {doppler_shift}, correlation magnitude of {correlation[sample_offset_where_we_started_receiving_prn]} at {sample_offset_where_we_started_receiving_prn}, time offset of {time_offset}, chip offset of {chip_offset_from_satellite}')
                #plt.plot(correlation)
                #plt.show()

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
    detected_satellites = satellite_detector.detect_satellites_in_data(subset_of_antenna_data_for_satellite_detection)
    for detected_satellite_id, detected_satellite_info in detected_satellites.items():
        print(detected_satellite_id, detected_satellite_info)


def main2():
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['figure.figsize'] = (12, 4)
    plt.rcParams['agg.path.chunksize'] = 10000

    input_source = INPUT_SOURCES[5]
    sample_rate = input_source.sdr_sample_rate
    # Generate PRN signals for each satellite
    satellites_to_replica_prn_signals = generate_replica_prn_signals()
    satellites_by_id = {
        satellite_id: GpsSatellite(
            satellite_id=satellite_id,
            prn_code=code
        )
        for satellite_id, code in satellites_to_replica_prn_signals.items()
    }

    sdr_data = get_samples_from_radio_input_source(input_source, sample_rate)

    start_time = 0
    end_time = 1
    time = np.arange(start_time, end_time, 1/(float(sample_rate)))
    # TODO(PT): This should vary over a search space
    doppler_cosine = generate_cosine_with_frequency(time, -2500)

    if len(doppler_cosine) != len(sdr_data):
        raise ValueError(f'Expected the generated carrier frequency and read SDR data to be the same length')

    # We're going to read PRN_CORRELATION_CYCLE_COUNT repetitions of the PRN
    vector_size = int(PRN_CORRELATION_CYCLE_COUNT * sample_rate / 1000)
    print(f'Vector size: {vector_size}')

    # Multiply the input signal with our Doppler-shifted cosine wave, to align the input signal with our reference PRN
    # In other words, this multiplication aligns our received data to baseband (if the Doppler shift is correct).
    signal_multiplied_with_doppler_shifted_carrier = doppler_cosine * sdr_data
    # Throw away extra signal data that doesn't fit in a modulo of our window size
    signal_multiplied_with_doppler_shifted_carrier = signal_multiplied_with_doppler_shifted_carrier[:round_to_previous_multiple_of(len(signal_multiplied_with_doppler_shifted_carrier), vector_size)]

    while True:
        time_offsets = []
        for i, signal_chunk in enumerate(
            chunks(signal_multiplied_with_doppler_shifted_carrier, vector_size)
        ):
            prn_fft = satellites_by_id[GpsSatelliteId(id=24)].fft_of_prn_of_length(vector_size)
            print(f'{vector_size} chunk #{i}...')

            fft_of_doppler_shifted_signal = np.fft.fft(signal_chunk)

            mult = prn_fft * np.conjugate(fft_of_doppler_shifted_signal)
            mult_ifft = np.fft.ifft(mult)
            scaled_ifft = mult_ifft * ((1/vector_size)*vector_size)
            correlation = np.absolute(scaled_ifft)

            plt.cla()
            plt.ylim((0, PRN_CORRELATION_CYCLE_COUNT * 10))
            # Correlation across time
            plt.plot(correlation)
            #print(correlation)

            max_correlation_peak_time_offset = np.argmax(correlation)
            max_correlation_magnitude = correlation[max_correlation_peak_time_offset]
            print(f'Max correlation is at time offset {max_correlation_peak_time_offset}.')
            print(f'The correlation magnitude is {max_correlation_magnitude}.')

            time_offsets.append(max_correlation_peak_time_offset)

            plt.pause(1e-10)

        print(f'Time offsets:')
        from pprint import pprint
        pprint(sorted(time_offsets))


if __name__ == '__main__':
    main()
