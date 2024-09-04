
# Seismic Signal Denoising

In this kernel, I will walk you through some methods used to denoise seismic signals (and other signals in general) and explain exactly how they work. We will look at [Wavelet Denoising with High-Pass Filters](#1) and [Average Smoothing](#1). The first method is used to remove the artificial impulse and the latter one is used to remove general noise.

![Seismic Signals](https://i.imgur.com/hBPv3fh.png)

## Problem Overview

The main problem in the specific case of seismic signals is the fact that the signal we measure with a seismograph is not an accurate representation of the actual underground seismic signal we are trying to uncover. In seismology, we (the people trying to measure the seismic signals) artificially generate signals called impulse signals. These impulse signals interact with the Earth's actual seismic signal (which is what we need) to produce the final signal which our seismograph picks up (this same process takes place in the laboratory simulation of an earthquake). So, the real challenge is to uncover the actual signal from the mixed seismogram (which is a combination of the Earth's impulse signal and the artificial impulse signal).

This actual underlying signal would be a better predictor of earthquake timing than the original raw signal because it represents the actual seismic activity.

![Signal Comparison](https://qph.fs.quoracdn.net/main-qimg-4033b6af35e7154e6b497adf0d93c2d9-c)

## Acknowledgements

I would like to thank Jack for [his kernel](https://www.kaggle.com/jackvial/dwt-signal-denoising) and Theo for [his kernel](https://www.kaggle.com/theoviel/fast-fourier-transform-denoising) (both on denoising).

## Importing Necessary Libraries

```python
import os
import gc
import numpy as np
from numpy.fft import *
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import pywt 
from statsmodels.robust import mad
import scipy
from scipy import signal
from scipy.signal import butter, deconvolve
import warnings
warnings.filterwarnings('ignore')
```

## Signal Segment and Sampling Rate Specifications

```python
SIGNAL_LEN = 150000
SAMPLE_RATE = 4000
```

## Reading and Processing Data

```python
seismic_signals = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})
acoustic_data = seismic_signals.acoustic_data
time_to_failure = seismic_signals.time_to_failure
data_len = len(seismic_signals)
del seismic_signals
gc.collect()

signals = []
targets = []

for i in range(data_len//SIGNAL_LEN):
    min_lim = SIGNAL_LEN * i
    max_lim = min([SIGNAL_LEN * (i + 1), data_len])
    
    signals.append(list(acoustic_data[min_lim : max_lim]))
    targets.append(time_to_failure[max_lim])
    
del acoustic_data
del time_to_failure
gc.collect()
    
signals = np.array(signals)
targets = np.array(targets)
```

## Mean Absolute Deviation

```python
def maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)
```

## High-Pass Filter with SOS Filter

```python
def high_pass_filter(x, low_cutoff=1000, SAMPLE_RATE=SAMPLE_RATE):
    nyquist = 0.5 * SAMPLE_RATE
    norm_low_cutoff = low_cutoff / nyquist
    sos = butter(10, Wn=[norm_low_cutoff], btype='highpass', output='sos')
    filtered_sig = signal.sosfilt(sos, x)
    return filtered_sig
```

## Wavelet Denoising

```python
def denoise_signal(x, wavelet='db4', level=1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1/0.6745) * maddest(coeff[-level])
    uthresh = sigma * np.sqrt(2*np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
    return pywt.waverec(coeff, wavelet, mode='per')
```

## Visualizing the Effects of High-Pass Filter and Wavelet Denoising

The high-pass filter and wavelet denoising techniques are able to effectively denoise the signal by removing the unnecessary artificial impulse and additional noise from the seismic signals.

## Average Smoothing

```python
def average_smoothing(signal, kernel_size, stride):
    sample = []
    start = 0
    end = kernel_size
    while end <= len(signal):
        start = start + stride
        end = end + stride
        sample.append(np.mean(signal[start:end]))
    return np.array(sample)
```

## Visualizing the Effects of Average Smoothing

The average smoothing method is only able to remove some additional noise, but not the artificial impulse. This method smooths the curve but doesn't effectively remove the artificial signal.

## Conclusion

We can conclude that the wavelet denoising method is overall more effective than the average smoothing method. Denoising the signals can significantly boost the scores of all the models (NN or LightGBM).

---

Thanks for reading my kernel! Hope you found it useful :)
