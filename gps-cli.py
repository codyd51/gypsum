import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
#from rtlsdr import RtlSdr
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

from scipy.signal import max_len_seq, resample


def main2():
    t = np.arange(100)
    s = np.sin(0.15 * 2 * np.pi * t)

    S = np.fft.fftshift(np.fft.fft(s))

    #plt.plot(s)
    #plt.plot(S)

    print(s)
    print(S)

    S_mag = np.abs(S)
    S_phase = np.angle(S)
    plt.plot(t, S_mag, ".-")
    plt.plot(t, S_phase, ".-")

    plt.show()


def main3():
    freq = 1
    n = 100

    t = np.arange(100)
    s = np.sin(0.15 * 2 * np.pi * t)
    # Filter window
    s = s * np.hamming(100)
    S = np.fft.fftshift(np.fft.fft(s))
    S_mag = np.abs(S)
    S_phase = np.angle(S)

    f = np.arange(freq / -2, freq / 2, freq / n)
    plt.figure(0)
    plt.plot(f, S_mag, '.-')
    plt.figure(1)
    plt.plot(f, S_phase, '.-')
    plt.show()


def main4():
    sample_rate = 1e6
    # Tone plus noise
    t = np.arange(1024*1000) / sample_rate
    # Tone frequency
    f = 50e3
    x = np.sin(2*np.pi*f*t) + 0.2 * np.random.randn(len(t))

    # Filter window
    #s = x * np.hamming(100)
    s = x
    S = np.fft.fftshift(np.fft.fft(s))
    S_mag = np.abs(S)
    S_phase = np.angle(S)

    if False:
        plt.plot(S_mag)
        plt.plot(S_phase)
        plt.show()

    fft_size = 1024
    num_rows = len(x) // fft_size
    spectrogram = np.zeros((num_rows, fft_size))
    for i in range(num_rows):
        spectrogram[i,:] = 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(x[i * fft_size:(i + 1) * fft_size]))) ** 2)

    plt.imshow(spectrogram, aspect='auto', extent = [sample_rate/-2/1e6, sample_rate/2/1e6, 0, len(x)/sample_rate])
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Time [s]")
    plt.show()


def main5():
    sdr = RtlSdr()
    print(sdr)

    # configure device
    #sdr.sample_rate = 2.048e6  # Hz
    #sdr.center_freq = 1575.42e6     # Hz
    #sdr.freq_correction = 60   # PPM
    #sdr.gain = 'auto'

    gps_l1_center = 1575.42e6

    sdr.sample_rate = 2.048e6  # Hz
    sdr.center_freq = gps_l1_center     # Hz
    sdr.freq_correction = 60   # PPM
    sdr.gain = 'auto'

    samples = (sdr.read_samples(512*1024))

    # use matplotlib to estimate and plot the PSD
    plt.psd(samples, NFFT=1024, Fs=sdr.sample_rate / 1e6, Fc=sdr.center_freq / 1e6)
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Relative power (dB)')
    plt.show()

    if False:
        from time import sleep
        while True:
            print(sdr.read_samples(512))
            sleep(1)


def generate_sin_wave2(frequency: int) -> npt.ArrayLike:
    start_time = 0
    end_time = 1
    sample_rate = 10000
    time = np.arange(start_time, end_time, 1/sample_rate)
    theta = 0
    amplitude = 1
    wave = amplitude * np.sin(2 * np.pi * frequency * time + theta)
    return wave

def generate_sin_wave(frequency: int) -> npt.ArrayLike:
    # Set the carrier frequency and message frequency
    # Generate the time array
    t = np.arange(0, 1, 1/(2_400_000))

    # Generate the carrier signal
    c = np.sin(2 * np.pi * frequency * t)
    return c


def generate_prn(chip: list[int], x) -> list[int]:
    prn = lambda x: prn_seq[int(x*f_prn)%16]
    pass


def main6() -> None:
    mhz = 1_000_000
    carrier_wave_frequency = int(157542 * 10e3)
    carrier_wave_frequency = int(157542 * 10e3)
    print(carrier_wave_frequency)
    wave = generate_sin_wave(carrier_wave_frequency)
    #plt.figure(figsize=(200,6))
    #yyplt.xlim((carrier_wave_frequency - mhz, carrier_wave_frequency + mhz))
    plt.xlim((0, mhz/1000))
    plt.plot(wave)
    plt.show()

    # Set the carrier frequency and message frequency
    fc = int(157542 * 10e3)

    print(fc)
    fm = 5

    # Generate the time array
    t = np.arange(0, 1, 1/1000000)

    # Generate the carrier signal
    c = np.sin(2 * np.pi * fc * t)

    # Generate the message signal
    m = np.sin(2 * np.pi * fm * t)

    # Generate the AM signal
    am = c * (2 + m)

    # Plot the AM signal
    plt.plot(am, 'r')
    plt.show()
    #plt.figure(figsize=(2000, 6), dpi=80)
    plt.figure()

    carrier_wave_frequency = int(157542 * 10e3)
    print(carrier_wave_frequency)
    wave = generate_sin_wave(carrier_wave_frequency)
    plt.plot(wave)
    plt.show()
    # Chipping frequency
    prn_chip_frequency = 10.23e6 / 10

    if False:
        # Plot PRN chips
        #chip = np.array([0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1])
        chip = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        #adjusted_to_freq = [chip[int(x * prn_chip_frequency) % 16] for x in range(20)]
        for x in range(20):
            print(f'x * chip freq: {x * prn_chip_frequency}')
            print(f'idx: {int(x * prn_chip_frequency) % len(chip)}')
            print(f'val: {chip[int(x * prn_chip_frequency) % len(chip)]}')
        adjusted_to_freq = [chip[int(x * prn_chip_frequency) % len(chip)] for x in range(20)]
        print(adjusted_to_freq)
        plt.step(np.arange(0, len(adjusted_to_freq)), adjusted_to_freq)

    word =  [1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1]

    # for proper scaling
    #const = 48000

    graph_data_x = []
    graph_data_y = []
    for i in range(len(word)):
        graph_data_x.append(i * prn_chip_frequency)
        graph_data_x.append((i + 1) * prn_chip_frequency)
        graph_data_y.append(word[i])
        graph_data_y.append(word[i])


    #plt.plot(graph_data_x, graph_data_y)

    plt.show()


class InputFileType(Enum):
    Raw = auto()
    Wav = auto()


@dataclass
class InputFileInfo:
    path: Path
    format: InputFileType
    sdr_sample_rate: float


_INPUT_FILES_WITH_PARAMETERS = [
    InputFileInfo(
        path=Path(__file__).parent / "vendored_signals" / "2013_04_04_GNSS_SIGNAL_at_CTTC_SPAIN.dat",
        format=InputFileType.Raw,
        sdr_sample_rate=4e6,
    ),
]


# PT: The SDR must be set to this center frequency
_GPS_L1_FREQUENCY = 1575.42e6


@dataclass
class ChipDelayMs:
    """New-type to represent the delay assigned to a given GPS satellite PRN.
    Each chip is transmitted over 1 millisecond, so the chip delay is directly expressed in milliseconds.
    """
    delay_ms: int

    def __init__(self, delay_ms: int) -> None:
        self.delay_ms = delay_ms


@dataclass
class GpsSatelliteId:
    """New-type to semantically store GPS satellite IDs by their PRN signal ID"""
    id: int

    def __init__(self, id: int) -> None:
        self.id = id

    def __hash__(self) -> int:
        return hash(self.id)


@dataclass
class GpsReplicaPrnSignal:
    """New-type to semantically store the 'replica' PRN signal for a given satellite"""
    inner: np.ndarray


def _generate_ca_code_rolled_by(delay_ms: int) -> np.ndarray:
    """Generate the C/A code by generating the 'G1' and 'G2' sequences, then mixing them.
    Note that we generate the 'pure' G2 code, then roll it by `delay_ms` chips before mixing. This code is only ever
    used in the context of the PRN for a particular GPS satellite, so we don't need to access it pre-roll (though it's
    fine to pass `delay_ms=0` if this is desired.

    Note that the signal generated by this function needs further post-processing (to adjust its domain and range)
    before it represents a replica PRN signal.

    Ref: IS-GPS-200L §3.3.2.3: C/A-Code Generation
    """
    seq_bit_count = 10
    # PT: I'm out of my depth here, but it appears as though the scipy implementation needs to be passed the 'companion
    # set' of taps to produce the correct output sequence.
    #
    # If I'm understanding correctly, the GPS documentation specifies the taps in 'Galois generator' form, while the
    # scipy implementation needs to be provided the taps in 'Fibonacci generator' form.
    # Further reading: https://web.archive.org/web/20181001062252/http://www.newwaveinstruments.com/resources/articles/m_sequence_linear_feedback_shift_register_lfsr.htm
    #
    # Another way of understanding this which may also be correct: the GPS documentation vs. software libraries simply
    # describe the shift register in different orders from each other.
    #
    # (Terms 1 and X^n can be omitted as they're implicit)
    # G1 = X^10 + X^3 + 1
    g1 = max_len_seq(seq_bit_count, taps=[seq_bit_count - 3])[0]
    # G2 = X^10 + X^9 + X^8 + X^6 + X^3 + X^2 + 1
    g2 = max_len_seq(seq_bit_count, taps=[
        seq_bit_count - 9,
        seq_bit_count - 8,
        seq_bit_count - 6,
        seq_bit_count - 3,
        seq_bit_count - 2,
    ])[0]
    return np.bitwise_xor(
        g1,
        np.roll(g2, delay_ms)
    )


def generate_replica_prn_signals(sample_rate: int) -> dict[GpsSatelliteId, GpsReplicaPrnSignal]:
    # Ref: https://www.gps.gov/technical/icwg/IS-GPS-200L.pdf
    # Table 3-Ia. Code Phase Assignments
    # "The G2i sequence is a G2 sequence selectively delayed by pre-assigned number of chips, thereby
    # generating a set of different C/A-codes."
    # "The PRN C/A-code for SV ID number i is a Gold code, Gi(t), of 1 millisecond in length at a chipping
    # rate of 1023 kbps."
    # In other words, the C/A code for each satellite is a time-shifted version of the same signal, and the
    # delay is expressed in terms of a number of chips (each of which occupies 1ms).
    # PT: The above comment is wrong, the *total 1023-chip PRN* is transmitted every 1ms!
    satellite_id_to_delay_by_chip_count = {
        GpsSatelliteId(1): ChipDelayMs(5),
        GpsSatelliteId(2): ChipDelayMs(6),
        GpsSatelliteId(3): ChipDelayMs(7),
        GpsSatelliteId(4): ChipDelayMs(8),
        GpsSatelliteId(5): ChipDelayMs(17),
        GpsSatelliteId(6): ChipDelayMs(18),
        GpsSatelliteId(7): ChipDelayMs(139),
        GpsSatelliteId(8): ChipDelayMs(140),
        GpsSatelliteId(9): ChipDelayMs(141),
        GpsSatelliteId(10): ChipDelayMs(251),
        GpsSatelliteId(11): ChipDelayMs(252),
        GpsSatelliteId(12): ChipDelayMs(254),
        GpsSatelliteId(13): ChipDelayMs(255),
        GpsSatelliteId(14): ChipDelayMs(256),
        GpsSatelliteId(15): ChipDelayMs(257),
        GpsSatelliteId(16): ChipDelayMs(258),
        GpsSatelliteId(17): ChipDelayMs(469),
        GpsSatelliteId(18): ChipDelayMs(470),
        GpsSatelliteId(19): ChipDelayMs(471),
        GpsSatelliteId(20): ChipDelayMs(472),
        GpsSatelliteId(21): ChipDelayMs(473),
        GpsSatelliteId(22): ChipDelayMs(474),
        GpsSatelliteId(23): ChipDelayMs(509),
        GpsSatelliteId(24): ChipDelayMs(512),
        GpsSatelliteId(25): ChipDelayMs(513),
        GpsSatelliteId(26): ChipDelayMs(514),
        GpsSatelliteId(27): ChipDelayMs(515),
        GpsSatelliteId(28): ChipDelayMs(516),
        GpsSatelliteId(29): ChipDelayMs(859),
        GpsSatelliteId(30): ChipDelayMs(860),
        GpsSatelliteId(31): ChipDelayMs(861),
        GpsSatelliteId(32): ChipDelayMs(862),
    }
    satellite_id_to_replica_prn = {}
    for sat_name, prn_chip_delay_ms in satellite_id_to_delay_by_chip_count.items():

        # Generate the pure PRN signal
        prn_signal = _generate_ca_code_rolled_by(prn_chip_delay_ms.delay_ms)

        # We're going to be searching for this marker in a real/analog signal that we digitally sample at
        # a discrete rate.
        # Therefore, we need to 'sample' the PRN signal we've just created, so we know what it *should* look like in the
        # real measured signal.
        #
        # Firstly, the domain of our generated signal is currently [0 to 1], centered at 0.5.
        # The data that comes in via our antenna will instead vary from [-1 to 1], centered at 0.
        # Translate our generated signal so that it's centered at 0 instead of ...
        translated_prn_signal = prn_signal - 0.5
        # And scale it so that the domain goes from [-1 to 1] instead of [-0.5 to 0.5].
        # Note our signal has exactly 1023 data points (which is the correct/exact length of the G2 code)
        scaled_prn_signal = translated_prn_signal * 2
        # Finally, sample our generated signal so that we can see the sort of thing to look for in the data on the wire
        sampled_signal = resample(scaled_prn_signal, sample_rate // 1000)

        # All done, save the replica signal
        satellite_id_to_replica_prn[sat_name] = GpsReplicaPrnSignal(sampled_signal)
    return satellite_id_to_replica_prn


def main():
    input_file_info = _INPUT_FILES_WITH_PARAMETERS[0]

    sample_rate = input_file_info.sdr_sample_rate
    # Samples per chip (a GPS chip is exactly 1ms long)
    chip_samples = int(sample_rate // 1000)
    offset = 0
    num_chips = 2
    count = num_chips * chip_samples

    # PT: Only relevant for a .wav
    if input_file_info.format == InputFileType.Wav:
        byte_length = np.fromfile(input_file_info.path.as_posix(), dtype=np.int32, count=1, offset=44)[0]
        print(byte_length)
        data = np.fromfile(input_file_info.path.as_posix(), dtype=np.int16, count=byte_length//2, offset=48)
        if len(data) % 2 != 0:
            data = data[:-1]
    elif input_file_info.format == InputFileType.Raw:
        data = np.fromfile(input_file_info.path.as_posix(), dtype=np.int16, count=int(offset+count)*2, offset=44)
        # Quantise to 2 bits I and 2 bits Q per sample
        data = np.clip(np.floor_divide(data, 150), -2, 1) + 0.5
        # Convert to complex numpy array
        data = np.reshape(data, (data.shape[0]//2, 2))
        data = data[offset:, 0] + 1j * data[offset:, 1]
    else:
        raise ValueError(f'Unrecognized format')
    print(len(data))

    # Generate PRN signals for each satellite
    generate_replica_prn_signals(int(sample_rate))

    return


if __name__ == '__main__':
    main()
