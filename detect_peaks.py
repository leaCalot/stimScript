"""
Sample module to detect high peaks in audio samples and insert a corresponding section at the location.

 Author: Lea
"""
from scipy.io import wavfile
import scipy.signal as signal
import numpy as np
import argparse
import matplotlib.pyplot as plt


def butter_bandpass(lowcut, highcut, fs, order=5):
    """ Compose Butterworth bandpass filter.
    :param lowcut Lower cut-off frequency.
    :param highcut Higher cut-off frequency.
    :param order Order of filter
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Apply butterworth filter.
    :param data: Input time-series
    :param lowcut: higher cut-off frequency
    :param highcut: lower cut-off frequency
    :param fs: Sampling rate
    :param order: order of filter
    :return:
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find peaks and replace in audio file.')
    parser.add_argument('--input', required=True, help='Input file')
    parser.add_argument('--output', required=True, help='Output file')
    parser.add_argument('--insert', type=str, default='beep.wav', help='Insert waveform for peaks.')
    parser.add_argument('--thresh', type=float, default=0.5, help='Peak gradient threshold.')
    parser.add_argument('--low', type=float, default=1500, help='Lower cut-off frequency for Butterworth filter.')
    parser.add_argument('--smooth', type=float, default=100, help='Smooth frequency.')
    parser.add_argument('--show', action='store_true', help='Display results before saving.')

    args = parser.parse_args()

    rate, input_buffer = wavfile.read(args.input)
    insert_rate, insert_buffer = wavfile.read(args.insert)

    # FIXME: Implement handling of other than 44.1kHz wav mono files
    assert insert_rate == rate

    # High-pass (whips are usually high pitched)
    input_buffer = butter_bandpass_filter(input_buffer, args.low, rate/4.0, rate)

    # Get Amplitude envelope
    amplitude_envelope = np.abs(input_buffer)

    # Smooth a bit
    steps = int(rate / args.smooth)
    amplitude_envelope = np.convolve(amplitude_envelope, np.ones(steps)/steps, mode='same')

    gradient = np.gradient(amplitude_envelope)

    # Find peaks in the gradient
    peaks = np.zeros(gradient.shape, dtype=np.int16)
    peak_abs_threshold = args.thresh * np.max(gradient)
    peaks_idx, _ = signal.find_peaks(gradient, peak_abs_threshold, width=3, distance=rate)
    peaks[peaks_idx] = np.max(peak_abs_threshold)

    # Insert replacement file
    for peak in peaks_idx:
        if len(peaks[peak:-1]) > len(insert_buffer):
            peaks[peak:peak+len(insert_buffer)] = insert_buffer

    # Show results if needed
    if args.show:
        plt.plot(amplitude_envelope)
        plt.plot(peaks)
        plt.show()

    # Go to disk
    wavfile.write(args.output, rate, peaks)

