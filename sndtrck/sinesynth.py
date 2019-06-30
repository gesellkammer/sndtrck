import ctcsound
from ctcsound import MYFLT as _F


_csd_sinesynth = """
<CsoundSynthesizer>

<CsOptions>
-odac -+rtaudio={backend}
-m 0
-d
</CsOptions>

<CsInstruments>
sr     = {sr}
ksmps  = 128
nchnls = 2
0dbfs  = 1

gkMasterVol init 1

instr 1
    chnset {freq}, "freq"
    chnset {amp}, "amp"
    chnset {gain}, "gain"
    turnoff
endin 

instr 2
    iPortTime = {freqport}  ; ksmps * 4 / sr
    
    kfreq = chnget:k("freq")
    kamp = chnget:k("amp")
    kgain = chnget:k("gain")

    aamp = interp(port(kamp, iPortTime))
    ; kfreq2 = port(kfreq, iPortTime)
    kfreq2 = sc_lag(kfreq, iPortTime)
    a0 = oscili(aamp, kfreq2)
    kgain *= gkMasterVol
    aenv linsegr 0, 0.2, 1, 0.2, 0
    aenv *= interp(kgain)
    a0 *= aenv
    outch 1, a0, 2, a0
endin

instr 200
    igain = i(gkMasterVol)
    gkMasterVol linseg igain, p3-(ksmps/sr), 0
endin
</CsInstruments>

<CsScore>
i 1 0 -1
i 2 0 -1
f 0 14400    ; a 4 hours session should be enough
</CsScore>
</CsoundSynthesizer>
"""


class SineSynth:

    def __init__(self, freq=440, amp=0.5, sr=44100, backend='jack',
                 freqport=0.05, gain=1):
        self.freq = freq
        self.amp = amp
        self.sr = sr
        self.backend = backend
        self._cs = None
        self._pt = None
        self._exited = False
        self.freqport = freqport
        self.gain = gain
        self._start_csound()
        
    def _start_csound(self):        
        cs = ctcsound.Csound()
        csd = _csd_sinesynth.format(
            freq=self.freq, 
            amp=self.amp, 
            sr=self.sr,
            backend=self.backend,
            freqport=self.freqport,
            gain=self.gain)
        error = cs.compileCsdText(csd)
        if error:
            raise RuntimeError("Could not compile csound source")
        cs.start()
        pt = ctcsound.CsoundPerformanceThread(cs.csound())
        pt.play()
        self._cs = cs
        self._pt = pt
        self._setControlChannel = ctcsound.libcsound.csoundSetControlChannel

    def setOsc(self, idx, freq, amp, bw=0):
        self.setFreq(freq)
        self.setAmp(amp)

    def setFreq(self, freq):
        ctcsound.libcsound.csoundSetControlChannel(self._cs.cs, b"freq", _F(freq))
        # self._setControlChannel(self._cs, b"freq", _F(freq))

    def setAmp(self, amp):
        ctcsound.libcsound.csoundSetControlChannel(self._cs.cs, b"amp", _F(amp))
        
    def setGain(self, gain):
        ctcsound.libcsound.csoundSetControlChannel(self._cs.cs, b"gain", _F(gain))

    def fadeout(self, fadetime):
        self._pt.scoreEvent(0, 'i', [200, 0, fadetime])
        
    def stop(self):
        if self._exited:
            return
        self._pt.stop()
        self._cs.stop()
        self._cs.cleanup()
        self._exited = True


_csd_multisine = '''
sr     = {sr}
ksmps  = 128
nchnls = 2
0dbfs  = 1

#define MAXPARTIALS #1000#

gkFreqs[] init $MAXPARTIALS
gkAmps[]  init $MAXPARTIALS
gkBws[]   init $MAXPARTIALS

gkgain init 1

alwayson 2, {numosc}
schedule 1, 0, 1

instr 1
    kidx = 0
    while kidx < lenarray(gkFreqs) do
        gkFreqs[kidx] = 1000
        gkAmps[kidx] = 0
        gkBws[kidx] = 0
        kidx += 1
    od
    turnoff 
endin

instr 2
    inumosc = p4
    ; gaussian noise, freq. interpolation
    aout beadsynt gkFreqs, gkAmps, gkBws, inumosc, 5 
    aout *= interp(gkgain)
    outch 1, aout, 2, aout
endin

instr 100
    idx = int(p4)
    ifreq = p5
    iamp = p6
    ibw = p7
    iramptime = p3
    kfreq = sc_lag(k(ifreq), iramptime, i(gkFreqs, idx))
    kamp = sc_lag(k(iamp), iramptime, i(gkAmps, idx))
    ; kbw = sc_lag(k(ibw), iramptime, i(gkBws, idx))
    kbw = ibw
    gkFreqs[idx] = kfreq
    gkAmps[idx] = kamp
    gkBws[idx] = kbw
endin

instr 200
    ; puts "instr 200!", 1
    igain = i(gkgain)
    gkgain linseg igain, p3-(ksmps/sr), 0
endin
'''


class MultiSineSynth:

    def __init__(self, freqs, amps, bws, sr=44100, backend='jack', 
                 porttime=0.05):
        assert len(freqs) == len(amps) == len(bws)
        self.freqs = freqs
        self.amps = amps
        self.bws = bws
        self.sr = sr
        self.backend = backend
        self._cs = None
        self._pt = None
        self._exited = False
        self.porttime = porttime
        self.fadetime = 0.2
        self._csdstr = _csd_multisine
        self._numosc = len(freqs)
        self._start_csound()
        for i in range(self._numosc):
            self.setOsc(i, freqs[i], amps[i], bws[i])

    def _start_csound(self):        
        cs = ctcsound.Csound()
        orc = self._csdstr.format(
            sr=self.sr,
            backend=self.backend,
            numosc=self._numosc)
        options = [
            "-d",
            "-odac",
            "-+rtaudio=%s" % self.backend,
            "-m 0"
        ]
        for opt in options:
            cs.setOption(opt)
        cs.compileOrc(orc)
        cs.start()
        pt = ctcsound.CsoundPerformanceThread(cs.csound())
        pt.play()
        # pt.scoreEvent(False, 'i', (1, 0, -1))
        self._cs = cs
        self._pt = pt
    
    def setOsc(self, idx, freq, amp, bw):
        if idx > self._numosc - 1:
            raise IndexError("osc out of range")
        dur = self.porttime
        self._pt.scoreEvent(0, 'i', [100, 0, dur, idx, freq, amp, bw])
        # self._pt.inputMessage(f"i 100 0 {dur} {idx} {freq} {amp} {bw}")

    def fadeout(self):
        self._pt.scoreEvent(0, 'i', [200, 0, self.fadetime])

    def stop(self):
        self._pt.stop()
        self._cs.stop()
        self._cs.cleanup()
        self._exited = True
