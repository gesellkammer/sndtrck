<CsoundSynthesizer>

<CsOptions>
-odac 
-+rtaudio=jack
</CsOptions>

<CsInstruments>
sr     = 44100
ksmps  = 128
nchnls = 2
0dbfs  = 1

#define MAXPARTIALS #1000#

gkFreqs[] init $MAXPARTIALS
gkAmps[]  init $MAXPARTIALS
gkBws[]   init $MAXPARTIALS

instr init
	kidx = 0
	while kidx < lenarray(gkFreqs) do
		gkFreqs[kidx] = 1000
		gkAmps[kidx] = 0
		gkBws[kidx] = 0
		kidx += 1
	od
	turnoff 
endin

instr setosc
	idx = int(p4)
	ifreq = p5
	iamp = p6
	ibw = p7
	iramptime = p3
	kfreq = sc_lag(k(ifreq), iramptime, i(gkFreqs, idx))
	kamp = sc_lag(k(iamp), iramptime, i(gkAmps, idx))
	kbw = sc_lag(k(ibw), iramptime, i(gkBws, idx))
	gkFreqs[idx] = kfreq
	gkAmps[idx] = kamp
	gkBws[idx] = kbw
endin

instr oscil
	inumosc = p4
	aout beadsynt 1, 1, gkFreqs, gkAmps, gkBws, inumosc, -1, -1, 0 
	outch 1, aout, 2, aout
endin


</CsInstruments>

<CsScore>
i "init"  0 -1
; i "oscil" 0 -1 20
f 0 14400    ; a 4 hours session should be enough
</CsScore>
</CsoundSynthesizer>
<bsbPanel>
 <label>Widgets</label>
 <objectName/>
 <x>0</x>
 <y>0</y>
 <width>0</width>
 <height>0</height>
 <visible>true</visible>
 <uuid/>
 <bgcolor mode="nobackground">
  <r>255</r>
  <g>255</g>
  <b>255</b>
 </bgcolor>
</bsbPanel>
<bsbPresets>
</bsbPresets>
<EventPanel name="" tempo="60.00000000" loop="8.00000000" x="368" y="239" width="655" height="346" visible="false" loopStart="0" loopEnd="0">i "setosc" 0 0.5 2 808 0.1 0 </EventPanel>
