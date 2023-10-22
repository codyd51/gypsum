import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

import scipy.signal.windows

from gps.gps_ca_prn_codes import generate_replica_prn_signals, GpsSatelliteId, GpsReplicaPrnSignal
from gps.radio_input import INPUT_SOURCES, get_samples_from_radio_input_source
from gps_project_name.gps.utils import chunks

# PT: The SDR must be set to this center frequency
_GPS_L1_FREQUENCY = 1575.42e6


@dataclass
class GpsSatellite:
    satellite_id: GpsSatelliteId
    prn_code: GpsReplicaPrnSignal


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
    prn24 = satellites_to_replica_prn_signals[GpsSatelliteId(id=24)].inner

    # Repeat each chip data point twice
    # This is because we'll be sampling at Nyquist frequency (2 * signal frequency, which is 1023 data points)
    prn_with_repeated_data_points = np.repeat(prn24, 2)
    # Adjust domain from [0 - 1] to [-1, 1] to match the IQ samples we'll receive
    prn_with_adjusted_domain = np.array([-1 if chip == 0 else 1 for chip in prn_with_repeated_data_points])
    # Convert to complex with a zero imaginary part
    prn_as_complex = [x+0j for x in prn_with_adjusted_domain]
    t = np.linspace(0, 1, sample_rate)

    start_time = 0
    end_time = 1
    time = np.arange(start_time, end_time, 1/(float(sample_rate)))

    doppler_frequency = -2500
    i_components = np.cos(2. * np.pi * time * doppler_frequency)
    q_components = np.sin(2. * np.pi * time * doppler_frequency)
    #doppler_cosine = [complex(i, q) for (i, q) in zip(i_components, q_components)]
    doppler_cosine = [i + (1j*q) for (i, q) in zip(i_components, q_components)]
    print(f'Doppler cosine length: {len(doppler_cosine)}')

    sdr_data = get_samples_from_radio_input_source(input_source, sample_rate)
    print(sdr_data)

    if len(doppler_cosine) != len(sdr_data):
        raise ValueError(f'Expected the generated carrier frequency and read SDR data to be the same length')

    signal_multiplied_with_doppler_shifted_carrier = doppler_cosine * sdr_data

    Ncycles = 32
    vector_size = int(Ncycles * sample_rate /1000)
    print(f'Vector size: {vector_size}')

    while True:
        print(f'repeat')
        for i, (signal_chunk, prn_chunk) in enumerate(zip(
            chunks(signal_multiplied_with_doppler_shifted_carrier, vector_size),
            chunks(np.tile(prn_as_complex, 1000), vector_size)
        )):
            print(f'****** i {i}, len(signal chunk) {len(signal_chunk)} len(prn_chunk) {len(prn_chunk)}')

            print(f'{vector_size} chunk #{i}...')

            fft_of_complex_prn = np.fft.fft(prn_chunk)
            fft_of_doppler_shifted_signal = np.fft.fft(signal_chunk)

            mult = fft_of_complex_prn * np.conjugate(fft_of_doppler_shifted_signal)
            mult_ifft = np.fft.ifft(mult)
            scaled_ifft = mult_ifft * ((1/vector_size)*vector_size)
            correlation = np.absolute(scaled_ifft)

            plt.cla()
            plt.ylim((0, Ncycles*10))
            plt.plot(correlation)
            print(correlation)
            plt.pause(1e-10)


if __name__ == '__main__':
    main()
