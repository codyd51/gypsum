import numpy as np
import matplotlib.pyplot as plt

from gps.gps_ca_prn_codes import generate_replica_prn_signals, GpsSatelliteId
from gps.radio_input import INPUT_SOURCES, get_samples_from_radio_input_source
from gps_project_name.gps.config import PRN_CORRELATION_CYCLE_COUNT
from gps_project_name.gps.satellite import GpsSatellite
from gps_project_name.gps.utils import chunks

# PT: The SDR must be set to this center frequency
_GPS_L1_FREQUENCY = 1575.42e6


def show_and_quit(x, array):
    ax1 = plt.subplot(212)
    ax1.margins(0.02, 0.2)
    ax1.use_sticky_edges = False

    ax1.plot(x, array)
    plt.show()


def main():
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

    start_time = 0
    end_time = 1
    time = np.arange(start_time, end_time, 1/(float(sample_rate)))

    doppler_frequency = -2500
    i_components = np.cos(2. * np.pi * time * doppler_frequency)
    q_components = np.sin(2. * np.pi * time * doppler_frequency)
    doppler_cosine = [i + (1j*q) for (i, q) in zip(i_components, q_components)]
    print(f'Doppler cosine length: {len(doppler_cosine)}')

    sdr_data = get_samples_from_radio_input_source(input_source, sample_rate)
    print(sdr_data)

    if len(doppler_cosine) != len(sdr_data):
        raise ValueError(f'Expected the generated carrier frequency and read SDR data to be the same length')

    signal_multiplied_with_doppler_shifted_carrier = doppler_cosine * sdr_data

    vector_size = int(PRN_CORRELATION_CYCLE_COUNT * sample_rate / 1000)
    print(f'Vector size: {vector_size}')

    while True:
        for i, signal_chunk in enumerate(
            chunks(signal_multiplied_with_doppler_shifted_carrier, vector_size)
        ):
            prn_fft = satellites_by_id[GpsSatelliteId(id=24)].fft_of_prn_of_length(vector_size)
            print(f'****** i {i}, len(signal chunk) {len(signal_chunk)} len(prn_fft) {len(prn_fft)}')

            print(f'{vector_size} chunk #{i}...')

            fft_of_doppler_shifted_signal = np.fft.fft(signal_chunk)

            mult = prn_fft * np.conjugate(fft_of_doppler_shifted_signal)
            mult_ifft = np.fft.ifft(mult)
            scaled_ifft = mult_ifft * ((1/vector_size)*vector_size)
            correlation = np.absolute(scaled_ifft)

            plt.cla()
            plt.ylim((0, PRN_CORRELATION_CYCLE_COUNT * 10))
            plt.plot(correlation)
            print(correlation)
            plt.pause(1e-10)


if __name__ == '__main__':
    main()
