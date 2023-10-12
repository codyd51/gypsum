from typing import Any, Iterator

import numpy as np
import matplotlib.pyplot as plt
#from rtlsdr import RtlSdr
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

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


def main():
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['figure.figsize'] = (9, 7)

    input_source = INPUT_SOURCES[4]
    sample_rate = input_source.sdr_sample_rate
    # Generate PRN signals for each satellite
    satellites_to_replica_prn_signals = generate_replica_prn_signals()
    satellites = [
        GpsSatellite(
            satellite_id=satellite_id,
            prn_code=code
        )
        for satellite_id, code in satellites_to_replica_prn_signals.items()
    ]
    #rint(satellites)

    prn1 = satellites_to_replica_prn_signals[GpsSatelliteId(id=1)].inner
    prn2 = satellites_to_replica_prn_signals[GpsSatelliteId(id=2)].inner
    prn22 = satellites_to_replica_prn_signals[GpsSatelliteId(id=22)].inner

    prn_scale_factor = int(np.lcm(1023, 120000) / 1023)
    prn1_fft = np.repeat(np.fft.fft(prn1), prn_scale_factor)
    #plt.plot(prn1_fft)
    #plt.show()

    if False:
        prn1_fft = np.fft.fft(prn1)
        fig = plt.figure()
        fig.suptitle("Correlation of PRN 1 with PRN 1")
        plt.plot(np.fft.ifft(prn1_fft*prn1_fft))
        plt.show()


    data = get_samples_from_radio_input_source(input_source)
    #plt.plot(data)
    plt.plot(np.fft.fft(data))
    plt.show()
    return
    data_first_50_ms = data[:50 * (int(sample_rate) // 1000)]
    data_scale_factor = int(np.lcm(1023, 120000) / 120000)
    print(data_scale_factor)
    #data_first_50_ms = np.repeat(data_first_50_ms, data_scale_factor)

    plt.plot(data_first_50_ms)
    data_first_50_ms_fft = np.fft.fft(data_first_50_ms)
    #plt.plot(data_first_50_ms_fft)
    #plt.plot(data_first_50_ms)

    #prn1_fft = np.fft.fft(prn1)
    #plt.plot(prn1_fft)

    #c = data_first_50_ms_fft * prn1_fft
    #plt.plot(c)

    plt.show()
    return

    data_fft = np.fft.fft(data[:4000])

    for doppler in range(-10_000, 10_000, 500):
        rolled_data_fft = np.roll(data_fft, doppler)

        plt.plot(data_fft)
        plt.plot(rolled_data_fft)
        plt.show()
        return

    data_fft = np.fft.fft(np.repeat(interval, scale_factor // samples_per_ms))
    for doppler in range(-doppler_offset, doppler_offset+1, 1000):
        #offset = int(doppler / sample_rate * (samples_per_prn)
        offset = int(doppler / sample_rate * samples_per_ms)
        doppler_adjusted_fft = np.roll(ca_fft, offset)
        #plt.plot(doppler_adjusted_fft)
        #X = ca_prn_ffts[satellite_id] * S_
        cross = doppler_adjusted_fft * data_fft
        if cross.max() > 15000:
            plt.plot(cross)
            print(f'Max (offset {doppler}): {cross.max()}')
            identified_satellites[satellite_id] = True
            break
    return

    plt.plot(satellites_to_replica_prn_signals[GpsSatelliteId(id=1)].inner)
    plt.figure()
    plt.plot(satellites_to_replica_prn_signals[GpsSatelliteId(id=22)].inner)
    plt.show()
    return

    print(f'Analyzing data containing {len(data)} datapoints...')

    samples_per_second = 4_000_000
    samples_per_ms = samples_per_second // 1000
    subsample_factor = 1023 / samples_per_ms
    print(subsample_factor)

    scale_factor = int(np.lcm(1023, int(samples_per_ms)))
    ca_prn_ffts = {
        satellite.satellite_id: np.fft.fft(np.repeat(satellite.prn_code.inner, int(scale_factor / 1023)))
        for satellite in satellites
    }

    if False:
        for sat_id, fft in ca_prn_ffts.items():
            plt.figure(sat_id.id, figsize=(30, 6), dpi=80)
            plt.plot(satellites_to_replica_prn_signals[sat_id].inner)
            if sat_id.id > 3:
                break
            #plt.plot(fft)
        plt.show()

    doppler_offset = 30000
    samples_per_prn = input_source.samples_in_prn_period
    #data_fft = np.fft.fft(data[:1023])
    # Look in the first bit of the data
    identified_satellites = {}

    #for interval in chunks(data, 1023):
    #    if len(interval) != 1023:
    #        continue
    for interval in chunks(data, samples_per_ms):
        if len(interval) != samples_per_ms:
            continue
        for satellite_id, ca_fft in ca_prn_ffts.items():
            if satellite_id in identified_satellites:
                # Already identified
                continue
            print(f'check {satellite_id}')
            #print(satellite_id, ca_fft)
            #data_fft = np.fft.fft(data[interval:(interval + 1023)])
            data_fft = np.fft.fft(np.repeat(interval, scale_factor // samples_per_ms))
            for doppler in range(-doppler_offset, doppler_offset+1, 1000):
                #offset = int(doppler / sample_rate * (samples_per_prn)
                offset = int(doppler / sample_rate * samples_per_ms)
                doppler_adjusted_fft = np.roll(ca_fft, offset)
                #plt.plot(doppler_adjusted_fft)
                #X = ca_prn_ffts[satellite_id] * S_
                cross = doppler_adjusted_fft * data_fft
                if cross.max() > 15000:
                    plt.plot(cross)
                    print(f'Max (offset {doppler}): {cross.max()}')
                    identified_satellites[satellite_id] = True
                    break

    plt.show()
    return

    # Search for each PRN code within the radio data
    for satellite in satellites:
        print(f'Searching for signal from satellite {satellite.satellite_id}...')
        #satellite.prn_code.inner = resample_signal_to_match_sample_rate(satellite.prn_code.inner, 0.001, sample_rate)

    #chip_window = min(10, num_chips//2)
    #chip_window = 10
    chip_window = 1
    num_chips = 2
    #plt.plot(satellites[0].prn_code.inner)
    #plt.show()
    #plt.plot(np.fft.fft(satellites[0].prn_code.inner))
    #plt.show()
    ca_prn_ffts = {
        satellite.satellite_id: np.conjugate(np.fft.fft(
            # Pad up to 2 durations???
            np.concatenate((
                np.tile(satellite.prn_code.inner, chip_window),
                np.zeros(satellite.prn_code.inner.shape[0] * (num_chips - chip_window))
            ))
        ))
        for satellite in satellites
    }
    doppler_offset = 30000
    yy = np.arange(-doppler_offset, doppler_offset+1, 250)
    for satellite_id, ca_chip in ca_prn_ffts.items():
        print(satellite_id)
        print(ca_chip)
        # Cp* is fft_ca_chips[1]
        S = np.fft.fft(data)
        print(f'data len: {len(data)}')
        print(f'S len: {len(S)}')
        samples_per_prn = input_source.samples_in_prn_period
        results_phase = np.zeros(samples_per_prn)
        results_doppler = []

        xx = np.arange(samples_per_prn)/samples_per_prn

        for doppler in range(-doppler_offset, doppler_offset+1, 250):
            print(f'\tchecking doppler shift {doppler}')
            offset = int(doppler / sample_rate * samples_per_prn * chip_window)
            S_ = np.roll(S, offset)
            if True:
                plt.figure(0)
                plt.plot(S)
                #plt.plot(S)
                plt.figure(1)
                plt.plot(S_)
                plt.show()
            X = ca_prn_ffts[satellite_id] * S_
            x = np.fft.ifft(X)
            plt.figure(0)
            #plt.plot(ca_prn_ffts[satellite_id])
            plt.plot(satellites_to_replica_prn_signals[satellite_id].inner)
            plt.figure(1)
            plt.plot(x)
            plt.show()
            x = np.sum(x.reshape(num_chips, samples_per_prn), axis=0)
            x_abs = abs(x)**2
            results_doppler.append(x_abs.max())
            results_phase = np.maximum(x_abs, results_phase)
        results_doppler = np.array(results_doppler)
        snr_doppler = results_doppler.max() / results_doppler.mean()
        snr_phase = results_phase.max() / results_phase.mean()
        threshold = 10
        if (snr_doppler * snr_phase > threshold):
            print(f'Found match over doppler / phase, doppler {snr_doppler} phase {snr_phase} threshold {threshold} this {snr_phase*snr_doppler}')
            # Plot matches over doppler and phase
            fig, axs = plt.subplots(1,2,figsize=(10,5))
            axs[0].plot(xx, results_phase)
            axs[0].set_title(f"PRN {satellite_id.id} phase {1000*xx[np.argmax(results_phase)]}us")
            axs[1].plot(yy, results_doppler)
            axs[1].set_title(f"PRN {satellite_id.id} offset {yy[np.argmax(results_doppler)]} Hz")
            plt.show()


    return


def test2() -> dict[GpsSatelliteId, np.ndarray]:
    import numpy as np
    from blipgps.gps_sat_info import sats_info
    from blipgps.gps_chip import ca_table
    fn = (Path(__file__).parents[0] / "vendored_signals" / "2013_04_04_GNSS_SIGNAL_at_CTTC_SPAIN.dat").as_posix()
    # Set up inputs to algorithm
    fc = 1575.42e6  # Assumed centre frequency of receiver
    fs = 4e6 # Sample rate of receiver
    chip_samples = int(fs//1000) # Samples per chip (a GPS chip is exactly 1ms long)
    offset = 0
    num_chips = 2
    count = num_chips * chip_samples

    data = np.fromfile(fn, dtype=np.int16, count=int(offset+count)*2)
    # Quantise to 2 bits I and 2 bits Q per sample
    data = np.clip(np.floor_divide(data, 150), -2, 1) + 0.5
    # Convert to complex numpy array
    data = np.reshape(data, (data.shape[0]//2, 2))
    data = data[offset:, 0] + 1j * data[offset:, 1]

    # Precalculate reference tables
    # These are reference samples of all known GPS chips.
    ca_chip_table = ca_table(int(fs))
    chip_window = min(10, num_chips//2)
    fft_ca_chips = {
        GpsSatelliteId(id=i): np.conjugate(np.fft.fft(
            # Pad to 2 durations
            np.concatenate((
                np.tile(ca, chip_window),
                np.zeros(ca.shape[0] * (num_chips - chip_window))
            ))
        )) for (i, ca) in ca_chip_table.items()
    }
    for i, ca_chip in fft_ca_chips.items():
        print(i)
        print(ca_chip)
    return fft_ca_chips


def my_method() -> dict[GpsSatelliteId, np.ndarray]:
    input_source = INPUT_SOURCES[0]
    sample_rate = input_source.sdr_sample_rate
    # Generate PRN signals for each satellite
    satellites_to_replica_prn_signals = generate_replica_prn_signals(int(sample_rate))
    satellites = [
        GpsSatellite(
            satellite_id=satellite_id,
            prn_code=code
        )
        for satellite_id, code in satellites_to_replica_prn_signals.items()
    ]
    print(satellites)

    data = get_samples_from_radio_input_source(input_source)
    print(f'Analyzing data containing {len(data)} datapoints...')

    # Search for each PRN code within the radio data
    for satellite in satellites:
        print(f'Searching for signal from satellite {satellite.satellite_id}...')

    chip_window = 1
    num_chips = 2
    ca_prn_ffts = {
        satellite.satellite_id: np.conjugate(np.fft.fft(
            # Pad up to 2 durations???
            np.concatenate((
                np.tile(satellite.prn_code.inner, chip_window),
                np.zeros(satellite.prn_code.inner.shape[0] * (num_chips - chip_window))
            ))
        ))
        for satellite in satellites
    }
    return ca_prn_ffts


def main2():
    orig = test2()
    mine = my_method()

    for satellite_id, fft in mine.items():
        try:
            orig_fft = orig[satellite_id]
        except KeyError:
            print(f'skipping sat {satellite_id}')
            continue
        print(f'Checking equality for {satellite_id}...')
        print(f'\t{fft == orig_fft}')
        if not np.array_equal(fft, orig_fft):
            raise ValueError(f"Not equal for satellite {satellite_id}: {fft}, {orig_fft}")
    print('Everything matched!')


def test2_2() -> dict[int, np.ndarray]:
    from blipgps.gps_chip import ca_table
    # Set up inputs to algorithm
    fs = 4e6 # Sample rate of receiver

    # Precalculate reference tables
    # These are reference samples of all known GPS chips.
    ca_chip_table = ca_table(int(fs))
    print(ca_chip_table)
    return ca_chip_table


def my_method_2() -> dict[int, np.ndarray]:
    input_source = INPUT_SOURCES[0]
    sample_rate = input_source.sdr_sample_rate
    # Generate PRN signals for each satellite
    #satellites_to_replica_prn_signals = generate_replica_prn_signals(int(sample_rate))
    satellites_to_replica_prn_signals = generate_replica_prn_signals(int(4e6))
    print(satellites_to_replica_prn_signals)
    return {s.id: v.inner for s, v in satellites_to_replica_prn_signals.items()}


def method3() -> dict[int, np.ndarray]:
    def shift(register, feedback, output):
        """GPS Shift Register
        :param list feedback: which positions to use as feedback (1 indexed)
        :param list output: which positions are output (1 indexed)
        :returns output of shift register:
        """
        # calculate output
        out = [register[i-1] for i in output]
        if len(out) > 1:
            out = sum(out) % 2
        else:
            out = out[0]

        # modulo 2 add feedback
        fb = sum([register[i-1] for i in feedback]) % 2

        # shift to the right
        for i in reversed(range(len(register[1:]))):
            register[i+1] = register[i]

        # put feedback in position 1
        register[0] = fb
        return out

    SV = {
        1: [2,6],
        2: [3,7],
        3: [4,8],
        4: [5,9],
        5: [1,9],
        6: [2,10],
        7: [1,8],
        8: [2,9],
        9: [3,10],
        10: [2,3],
        11: [3,4],
        12: [5,6],
        13: [6,7],
        14: [7,8],
        15: [8,9],
        16: [9,10],
        17: [1,4],
        18: [2,5],
        19: [3,6],
        20: [4,7],
        21: [5,8],
        22: [6,9],
        23: [1,3],
        24: [4,6],
        25: [5,7],
        26: [6,8],
        27: [7,9],
        28: [8,10],
        29: [1,6],
        30: [2,7],
        31: [3,8],
        32: [4,9],
    }
    def PRN(sv):
        """Build the CA code (PRN) for a given satellite ID
        :param int sv: satellite code (1-32)
        :returns list: ca code for chosen satellite
        """
        # init registers
        G1 = [1 for _ in range(10)]
        G2 = [1 for _ in range(10)]

        ca = []
        # create sequence
        for _ in range(1023):
            g1 = shift(G1, [3,10], [10])
            g2 = shift(G2, [2,3,6,8,9,10], SV[sv])

            # modulo 2 add and append to the code
            ca.append((g1 + g2) % 2)

        # return C/A code!
        return ca

    f_prn = 10.23e6 / 10  # chipping frequency
    # find ca code for sat 24, and make 0 into -1 to use in BPSK
    out = {}
    for sat_id in range(1, 33):
        vals = [-1 if x == 0 else 1 for x in PRN(sat_id)]
        #prn = lambda x: sat_24[int(x*f_prn)%1023]
        out[sat_id] = np.array(vals)
    return out


def main2_2():
    orig = test2_2()
    mine = my_method_2()
    new = method3()
    print(new)

    for satellite_id, fft in mine.items():
        print(f'Checking equality for {satellite_id}...')
        if False:
            orig_fft = orig[satellite_id]
            print(f'\t{fft == orig_fft}')
            if not np.array_equal(fft, orig_fft):
                raise ValueError(f"Not equal for satellite {satellite_id}: {fft}, {orig_fft}")

        n = new[satellite_id]
        print(fft == n)
        if not np.array_equal(fft, n):
            raise ValueError(f"Not equal with new method for satellite {satellite_id}: {fft}, {n}")

    print('Everything matched!')


def resample_signal_to_match_sample_rate(orig: np.ndarray, intended_duration: float, samples_per_second: float) -> np.ndarray:
    samples_in_duration = int(samples_per_second * intended_duration)
    num_data_points = len(orig)
    return np.interp(np.linspace(0, num_data_points - 1, num=samples_in_duration), np.arange(num_data_points), orig)


def main_new():
    input_source = INPUT_SOURCES[0]
    sample_rate = input_source.sdr_sample_rate
    # Generate PRN signals for each satellite
    satellites_to_replica_prn_signals = generate_replica_prn_signals()
    satellites = [
        GpsSatellite(
            satellite_id=satellite_id,
            prn_code=code
        )
        for satellite_id, code in satellites_to_replica_prn_signals.items()
    ]
    print(satellites)

    data = get_samples_from_radio_input_source(input_source)
    print(f'Analyzing data containing {len(data)} datapoints...')

    # Search for each PRN code within the radio data
    for satellite in satellites:
        print(f'Searching for signal from satellite {satellite.satellite_id}...')
        prn_resampled_to_radio_sample_rate = resample_signal_to_match_sample_rate(satellite.prn_code.inner, 0.001, sample_rate)


if __name__ == '__main__':
    main()
