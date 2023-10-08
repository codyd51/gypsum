import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from rtlsdr import RtlSdr
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path


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
        path=Path(__file__).parent / "2013_04_04_GNSS_SIGNAL_at_CTTC_SPAIN.dat",
        format=InputFileType.Raw,
        sdr_sample_rate=4e6,
    ),
]


# PT: The SDR must be set to this center frequency
_GPS_L1_FREQUENCY = 1575.42e6


def generate_replica_prn_signals():
    pass


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
