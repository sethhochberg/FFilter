SAMPLE_RATE = 44100 # Hz

import wave, struct, math, argparse
from numpy import fft

#Make this thing useful as a command line program
parser = argparse.ArgumentParser(description='A DFT Filtering Utility. For usage instructions, type \'ffilter --help\'.')
parser.add_argument('-f', type=str, dest='filename', required=True, help='the name of the WAV file to process')
parser.add_argument('-n', type=int, dest='FFT_LENGTH', required=True, help='the number of points to take in the DFT')
parser.add_argument('-hp', type=float, dest='highpass', required=True, help='cuttoff frequency for brick wall high pass')
parser.add_argument('-lp', type=float, dest='lowpass', required=True, help='cuttoff frequency for brick wall low pass')
parser.add_argument('-o', type=str, dest='outfilename', default='out.wav', help='optional file name for output, defaults to out.wav')

args = parser.parse_args()

FFT_LENGTH = args.FFT_LENGTH
lowpass = args.lowpass
highpass = args.highpass
filename = args.filename
outfilename = args.outfilename

OVERLAP = 512
FFT_SAMPLE = FFT_LENGTH - OVERLAP
NYQUIST_RATE = SAMPLE_RATE / 2.0

# Convert frequencies from Hz to our digital sampling units
lowpass /= (NYQUIST_RATE / (FFT_LENGTH / 2.0))
highpass /= (NYQUIST_RATE / (FFT_LENGTH / 2.0))

zeros = [ 0 for x in range(0, OVERLAP) ]

# Builds filter mask. Note that this filter is BAD, a
# good filter must have a smooth curve, respecting a
# dB/octave attenuation ramp!
mask = []
for f in range(0, FFT_LENGTH / 2 + 1):
    rampdown = 1.0
    if f > lowpass:
        rampdown = 0.0
    elif f < highpass:
        rampdown = 0.0
    mask.append(rampdown)

def bound(sample):
    # hard-limit clipped samples
    if sample > 1.0:
        print "!",
        sample = 1.0
    elif sample < -1.0:
        print "!",
        sample = -1.0
    return sample

infile = wave.open(filename, "r")
filtered = wave.open(outfilename, "w")
filtered.setnchannels(1)
filtered.setsampwidth(2)
filtered.setframerate(SAMPLE_RATE)

n = infile.getnframes()
original = struct.unpack('%dh' % n, infile.readframes(n))
# scale from 16-bit signed WAV to float
original = [s / 32768.0 for s in original]

saved_td = zeros

for pos in range(0, len(original), FFT_SAMPLE):
    time_sample = original[pos : pos + FFT_LENGTH]

    # convert frame to frequency domain representation
    frequency_domain = fft.fft(time_sample, FFT_LENGTH)
    l = len(frequency_domain)

    # mask positive frequencies (f[0] is DC component)
    for f in range(0, l/2+1):
        frequency_domain[f] *= mask[f]

    # mask negative frequencies
    for f in range(l-1, l/2, -1):
        cf = l - f
        frequency_domain[f] *= mask[cf]

    # convert frame back to time domain
    time_domain = fft.ifft(frequency_domain)

    # modified overlap-add logic: previously saved samples
    # prevail in the beginning, and they are ramped down
    # in favor of this frame's samples, to avoid 'clicks'
    # in either end of frame.
    for i in range(0, OVERLAP):
        time_domain[i] *= (i + 0.0) / OVERLAP
        time_domain[i] += saved_td[i] * (1.0 - (i + 0.00) / OVERLAP)

    # reserve last samples for the next frame
    saved_td = time_domain[FFT_SAMPLE:]
    # do not write reserved samples right now
    time_domain = time_domain[:FFT_SAMPLE]

    # scale back into WAV 16-bit and write
    time_domain = [ bound(sample) * 32767.0 for sample in time_domain ]
    filtered.writeframes(struct.pack('%dh' % len(time_domain),
                                     *time_domain))

