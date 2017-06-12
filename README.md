SNDTRCK
=======

A simple data-type and io routines for audio partial tracking

# Dependencies

## Mandatory

* [numpy]  
* [bpf4]   -- interpolation curves
* [peach]  -- for pitch and note conversion
* [sndfileio] -- simple API for reading and writing sound-files
* [loristrck] -- partial tracking analysis based on Loris

## Optional but recommended
* [h5py]    -- allows saving partial-tracking data as HDF5. You need to install hdf5.

# Installation

First, do:

    $ pip install numpy scipy cython
    
Then, install fftw

### OSX

    $ brew install fftw
    
### Windows 

Follow instructions [here](http://www.fftw.org/install/windows.html))

## Install sndtrck

    $ git clone https://github.com/gesellkammer/sndtrck
    $ cd sndtrck
    $ pip install -r requirements.txt
    $ python setup.py install
    
---

# Basic Usage

```python

import sndtrck
spectrum = sndtrck.analyze("/path/to/sndfile", resolution=50)

# Get the chord at 500ms, but only the partials louder than -30 dB
print(spectrum.chord_at(0.5, minamp=-30))
# [A3+, C5+10, E5-13]
spectrum.plot()  # this will generate a matplotlib plot of the partials
spectrum.show()  # this will show you the spectrum in the default applicatio for your system
spectrum.write("spectrum.sdif")
```

# Features

* analysis of sound-files in many formats
* automatic configuration of analysis parameters
* filering of partials based on multiple criteria
* resynthesis
* plotting
* export the spectrum to many different formats (sdif, hdf5, midi) 

# Transcription

Go to [trnscrb] for automatic transcription of spectra into musical notation

[bpf4]: https://github.com/gesellkammer/bpf4
[peach]: https://github.com/gesellkammer/peach
[loristrck]: https://github.com/gesellkammer/loristrck
[sndfileio]: https://github.com/gesellkammer/sndfileio
[pandas]: http://pandas.pydata.org/
[trnscrb]: https://github.com/gesellkammer/trnscrb
