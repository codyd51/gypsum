from typing import Any, Iterator

import numpy as np
import matplotlib.pyplot as plt
#from rtlsdr import RtlSdr
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

import scipy.signal.windows
from scipy.signal import max_len_seq, resample

from gps.gps_ca_prn_codes import generate_replica_prn_signals, GpsSatelliteId, GpsReplicaPrnSignal
from gps.radio_input import INPUT_SOURCES, get_samples_from_radio_input_source

# PT: The SDR must be set to this center frequency
_GPS_L1_FREQUENCY = 1575.42e6


def chunks(li: np.ndarray, chunk_size: int) -> Iterator[Any]:
    for i in range(0, len(li), chunk_size):
        yield li[i:i + chunk_size]


@dataclass
class GpsSatellite:
    satellite_id: GpsSatelliteId
    prn_code: GpsReplicaPrnSignal


def show_and_quit(x, array):
    #plt.ion()
    #plt.plot(x, array, '-bo')
    ax1 = plt.subplot(212)
    ax1.margins(0.02, 0.2)
    ax1.use_sticky_edges = False

    #ax1.plot(x, array, 'ro', linestyle="None")
    ax1.plot(x, array)
    plt.show()
    #import sys
    #sys.exit(0)
    return

    plt.draw()
    ax1.figure.canvas.draw()
    ax1.figure.canvas.flush_events()
    print('sleeping!')
    import time
    time.sleep(3)
    ax1.figure.canvas.draw()
    ax1.figure.canvas.flush_events()
    time.sleep(5)

    ax1.plot(x, array, 'bo', linestyle="None")
    time.sleep(10)


    #plt.show()

    import sys
    sys.exit(0)


def main():
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['figure.figsize'] = (12, 4)
    plt.rcParams['agg.path.chunksize'] = 10000

    input_source = INPUT_SOURCES[5]
    sample_rate = input_source.sdr_sample_rate
    # Generate PRN signals for each satellite
    satellites_to_replica_prn_signals = generate_replica_prn_signals()
    prn24 = satellites_to_replica_prn_signals[GpsSatelliteId(id=1)].inner

    if False:
        for satellite_id, prn in satellites_to_replica_prn_signals.items():
            print(f'\"{satellite_id.id}\": {list(prn.inner)}')
        import sys
        sys.exit(0)
    print(prn24)
    print(len(prn24))
    print(list(prn24))
    #prn24 = np.array([0 for _ in range(1023)])

    #plt.axis([0, 10, 0, 1])

    # Repeat each chip data point twice
    prn_with_repeated_data_points = np.repeat(prn24, 2)
    # Convert to complex with a zero imaginary part
    #print(prn_with_repeated_data_points)
    prn_with_adjusted_domain = np.array([-1 if chip == 0 else 1 for chip in prn_with_repeated_data_points])
    #prn_as_complex = prn_with_adjusted_domain.astype(complex)
    prn_as_complex = [x+0j for x in prn_with_adjusted_domain]
    #l = np.linspace(0, len(prn_as_complex), len(prn_as_complex))
    #show_and_quit(l, prn_as_complex)

    from scipy import signal
    # Same as SDR sample rate
    print(sample_rate)
    doppler_cosine = signal.windows.cosine(sample_rate, sym=False)
    doppler_cosine = signal.windows.cosine(1023*2)

    t = np.linspace(0, 1, sample_rate)
    #w = 1023*1000. * np.pi * sample_rate
    #w = 2. * np.pi * sample_rate * -2500
    w = 2. * np.pi * sample_rate * 0
    #doppler_cosine = np.cos(w * t)
    #doppler_cosine = np.cos(-2500*t)

    t = np.linspace(0, 1, sample_rate)

    freq = -2500 # in Hz
    phi = 0
    amp = 1
    k = 2*np.pi*freq*t + phi
    cwv = amp * np.exp(-1j* k) # complex sine wave
    doppler_cosine = cwv
    #print(doppler_cosine)
    #plt.plot(np.real(doppler_cosine), np.imag(doppler_cosine), '.')

    cycles = 2500 # how many sine cycles
    resolution = sample_rate # how many datapoints to generate
    length = np.pi * 2 * cycles
    i_components = -np.cos(np.arange(0, length, length / resolution))
    q_components = -np.sin(np.arange(0, length, length / resolution))
    doppler_cosine = [i+q for (i, q) in zip(i_components, q_components)]

    start_time = 0
    end_time = 1
    time = np.arange(start_time, end_time, 1/(float(sample_rate)))
    theta = 0
    frequency = -2500
    amplitude = 1
    sinewave = amplitude * np.sin(2 * np.pi * frequency * time + theta)
    coswave = amplitude * np.cos(2 * np.pi * frequency * time + theta)
    doppler_cosine = [i+(-1j*q) for (i, q) in zip(coswave, sinewave)]
    #plt.plot(time, coswave)
    #plt.plot(time, sinewave)
    #plt.plot(time, doppler_cosine)
    #plt.show()

    doppler_frequency = -2500
    i_components = np.cos(2. * np.pi * time * doppler_frequency)
    q_components = np.sin(2. * np.pi * time * doppler_frequency)
    #doppler_cosine = [complex(i, q) for (i, q) in zip(i_components, q_components)]
    doppler_cosine = [i + (1j*q) for (i, q) in zip(i_components, q_components)]
    print(f'Doppler cosine length: {len(doppler_cosine)}')

    #show_and_quit(t, doppler_cosine)
    #print(doppler_cosine)
    sdr_data = get_samples_from_radio_input_source(input_source, sample_rate)
    print(sdr_data)

    if len(doppler_cosine) != len(sdr_data):
        raise ValueError(f'Expected the generated carrier frequency and read SDR data to be the same length')

    #plt.plot(doppler_cosine)
    #plt.plot(sdr_data)
    signal_multiplied_with_doppler_shifted_carrier = doppler_cosine * sdr_data
    #plt.plot(np.real(signal_multiplied_with_doppler_shifted_carrier))
    #plt.plot(np.imag(signal_multiplied_with_doppler_shifted_carrier))
    #plt.show()

    Ncycles = 32
    vector_size = int(Ncycles * sample_rate /1000)
    print(f'Vector size: {vector_size}')

    fft_size = len(signal_multiplied_with_doppler_shifted_carrier)

    #plt.autoscale(True)
    #plt.ylim((0, 4_000_000))
    #ax1 = plt.subplot()
    #ax1.ylim((0, 200_000))
    #ax1 = plt.subplot(0, 0)
    #ax1 = plt.subplot()
    #ax1.margins(0.02, 0.2)
    #ax1.use_sticky_edges = False
    #plt.ion()
    #plt.show()

    #plt.plot(np.linspace(0, 0.001, 2046), prn_as_complex)
    #jplt.show()
    #return
    while True:
        print(f'repeat')
        for i, (signal_chunk, prn_chunk) in enumerate(zip(
            chunks(signal_multiplied_with_doppler_shifted_carrier, vector_size),
            chunks(np.tile(prn_as_complex, 1000), vector_size)
        )):
            print(f'****** i {i}, len(signal chunk) {len(signal_chunk)} len(prn_chunk) {len(prn_chunk)}')
            #plt.plot(prn_chunk)
            #plt.show()
            #return

            print(f'{vector_size} chunk #{i}...')
            #fft_of_complex_prn = np.fft.fft(np.repeat(prn_as_complex, 1000))
            #fft_of_doppler_shifted_signal = np.fft.fft(signal_multiplied_with_doppler_shifted_carrier)

            #mult = fft_of_complex_prn * np.conjugate(fft_of_doppler_shifted_signal)
            #print(len(mult))
            #print(mult)

            #fft_of_mult = np.fft.fft(mult)
            #scaled_fft = [((x / fft_size) * fft_size) for x in fft_of_mult]
            #absolute = np.absolute(scaled_fft)
            #show_and_quit(np.linspace(0, 1, fft_size), absolute)
            window = scipy.signal.windows.boxcar(len(prn_chunk))
            fft_of_complex_prn = np.fft.fft(prn_chunk)
            fft_of_doppler_shifted_signal = np.fft.fft(signal_chunk)

            mult = fft_of_complex_prn * np.conjugate(fft_of_doppler_shifted_signal)
            mult_ifft = np.fft.ifft(mult*window)
            #plt.plot(mult)
            #plt.plot(fft_of_mult)
            #plt.show()
            #print(f'fft_of_mult: {fft_of_mult[:100]}')
            #scaled_fft = [x * ((1 / float(vector_size)) * vector_size) for x in fft_of_mult]
            scaled_ifft = mult_ifft * ((1/vector_size)*vector_size)
            correlation = np.absolute(scaled_ifft)

            #ax1.plot(absolute, 'ro', linestyle="None")
            plt.cla()
            plt.ylim((0, Ncycles*10))
            #ax1.figure.clear()
            plt.plot(correlation)
            print(correlation)
            #ax1.plot(fft_of_complex_prn)
            #ax1.plot(prn_chunk)
            plt.pause(1e-10)
            #plt.pause(1)
            #plt.pause(1)
            #ax1.plot(x, array, 'ro', linestyle="None")
            #show_and_quit(np.linspace(0, vector_size, vector_size), absolute)

    show_and_quit(t, sdr_data)
    plt.figure(0)
    plt.plot(signal_multiplied_with_doppler_shifted_carrier)
    plt.figure(1)
    plt.plot(doppler_cosine)
    plt.figure(2)
    plt.plot(sdr_data)
    plt.show()
    return

    # Repeat each chip data point twice
    #plt.plot(prn1_fft)
    #plt.show()

    data = get_samples_from_radio_input_source(input_source)
    #plt.plot(data)
    plt.plot(np.fft.fft(data))
    plt.show()
    return


def main_test():
    path = Path("/Users/philliptennen/Documents/GPS/gps_project_name/vendored_signals/learnSDR_pure_prn")
    data = np.fromfile(path.as_posix(), dtype=np.float32, count=4*1023*1000)
    print(data)
    data = (data[0::2] + (1j * data[1::2]))
    print(data)
    import sys
    sys.exit(0)


if __name__ == '__main__':
    #main_test()

    main()
