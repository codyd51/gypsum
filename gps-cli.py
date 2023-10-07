import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from rtlsdr import RtlSdr


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


def generate_sin_wave(frequency: int) -> npt.ArrayLike:
    start_time = 0
    end_time = 1
    sample_rate = 1000
    time = np.arange(start_time, end_time, 1/sample_rate)
    theta = 0
    frequency = 100
    amplitude = 1
    wave = amplitude * np.sin(2 * np.pi * frequency * time + theta)
    return wave


def main() -> None:
    plt.figure(figsize=(20, 6), dpi=80)

    wave = generate_sin_wave(1024)
    plt.plot(wave)

    plt.show()


if __name__ == '__main__':
    main()
