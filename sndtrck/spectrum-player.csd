<CsoundSynthesizer>
<CsOptions>
; --omacro:MATRIXPATH=/home/em/tmp/py/out.wav
-b 512
-B 2048
</CsOptions>
<CsInstruments>

sr = 44100
ksmps = 256
#ifdef STEREO
	nchnls = 2
#else
	nchnls = 1
#endif
0dbfs = 1.0

#define MAXPARTIALS        #4096#
#define DEFAULT_OSCPORT    #10002#
#define DEFAULT_NOTIFYPORT #10003#

gkdur, gkdatastart, gkdataend init 0
gkplay init 0
gkLoopmode init -1
gkendpos init 0
gkFilterActive init 0
gkFilterFreq0 init 0
gkFilterFreq1 init 24000
gkFilterBw0 init 0
gkFilterBw1 init 1
gkFreqScale init 1
gkBwScale init 1

#ifdef MASTERGAIN
	gkgain init $MASTERGAIN
#else
	gkgain init 1
#endif


#ifdef OSCPORT
	gioscport = $OSCPORT
#else
	gioscport = $DEFAULT_OSCPORT
#endif

#ifdef SPEED 
	gkSpeed init $SPEED 
#else
	gkSpeed init 1
#endif 

#ifdef POSITION
	gkplayhead init $POSITION
	gkedithead init $POSITION
	prints "gkplayhead %f\n", gkplayhead
#else
	gkedithead init 0
	gkplayhead init 0
#endif

#ifdef NOTIFYPORT
	ginotifyport = $NOTIFYPORT
#else
	ginotifyport = $DEFAULT_NOTIFYPORT
#endif

#ifdef BEADSYNTFLAGS
	giBeadsyntFlags = $BEADSYNTFLAGS
#else
	giBeadsyntFlags = 0
#endif

#ifdef OSCOUTFREQ
	giOscOutFreq = $OSCOUTFREQ
#else
	giOscOutFreq = 20
#endif

giosc OSCinit gioscport

; *** tables ***
gifreqs ftgen 0, 0, $MAXPARTIALS, 7, 0  ; empty buffer
giamps  ftgen 0, 0, $MAXPARTIALS, 7, 0
gibws   ftgen 0, 0, $MAXPARTIALS, 7, 0
gispectrum ftgen 0, 0, 0, -1, "$MATRIXPATH", 0, 0, 0

chn_k "play", 3
chn_k "playhead", 2
chn_k "abstime", 2
chn_k "speed", 3
chn_k "loop", 3

prints "Path: $MATRIXPATH \n"
prints "SR: %d\n", sr

; **************************

opcode setPlay,0,k
	kstate xin
	gkplay = kstate
	chnset kstate, "play"
	if kstate == 0 then
		gkplayhead = gkedithead
	endif
endop	

opcode kclip,k,kkk
	kx, kx0, kx1 xin
	idt = ksmps/sr
	kout = kx > kx1 ? (kx1 - idt*10): (kx <= kx0 ? kx0 : kx)
	xout kout
endop

opcode setEdithead,0,k
	kedithead xin
	gkedithead = kclip(kedithead, 0, gkdataend)
	gkplayhead = gkedithead
endop

opcode setLoopmode,0,k
	kstate xin
	if kstate != gkLoopmode then
		gkLoopmode = kstate
		chnset kstate, "loop"
	endif
endop

opcode setSpeed,0,k
	kspeed xin
	if kspeed != gkSpeed then
		gkSpeed = kspeed
		chnset kspeed, "speed"
	endif
endop

; --------------------------------------------

instr ctrl
	kbutton_play chnget "play"
	if changed(kbutton_play) == 1 then
		setPlay(kbutton_play)
	endif
	
	kloopmode chnget "loop"
	setLoopmode(kloopmode)
	
	
	kedithead0 chnget "edithead"	
	kedithead lineto kedithead0, 0.05
	if changed(kedithead) == 1 then
		setEdithead(kedithead * gkdur)
	endif
	
	kspeed chnget "speed"
	if changed(kspeed)== 1 then
		gkSpeed = kspeed
	endif
	
	kguitrig metro 12
	idur0 = i(gkdur)
	idur = idur0 > 0 ? idur0 : 0.001
	if (kguitrig) == 1 then
		chnset gkplayhead / idur, "playhead"
		chnset gkplayhead, "abstime"
	endif
endin


instr oscils
	ifn = gispectrum
	iskip    tab_i 0, ifn
	idt      tab_i 1, ifn
	inumcols tab_i 2, ifn
	inumrows tab_i 3, ifn
	it0      tab_i 4, ifn
	inumpartials = inumcols / 3 
#ifdef ABSTIME
	iabstime = $ABSTIME
	puts ">>>>>>>>>>>>>>>>>>>>>>>> Absolute time!", 1
#else
	iabstime = 0
#endif
	imaxrow = inumrows - 2
	
	kfirstcycle init 0
	if kfirstcycle > 0 kgoto perf

	event_i "i", "post", 0, -1
	gkdatastart = iabstime > 0 ? it0 : 0
	gkdur = imaxrow * idt
	gkdataend = gkdur + gkdatastart
	kfirstcycle = 1
	
	; >>>>>>>>> perf <<<<<<<<<<<<
perf:
	
	if (changed(gkplay) == 1) || (changed(gkedithead) == 1) then
		gkplayhead = gkedithead
	endif

	ktime = gkplayhead - gkdatastart

#ifdef VERBOSE
	ktrig init 0
	ktrig += changed2(ktime, gkplayhead) 
	printf "ktime: %f  gkplayhead: %f\n", ktrig, ktime, gkplayhead
#endif

	krow = ktime / idt
	if (krow < 0) || (krow > imaxrow) || (gkplay == 0) kgoto exit

	;          krow, ifn, ifndst,  inumcols  iskip  istart iend istep
	tabrowlin  krow, ifn, gifreqs, inumcols, iskip, 0,     0,   3
	tabrowlin  krow, ifn, giamps,  inumcols, iskip, 1,     0,   3
	tabrowlin  krow, ifn, gibws,   inumcols, iskip, 2,     0,   3
	
	kGain[] init inumpartials 

	kB[] vecview gibws,   0, inumpartials
	kA[] vecview giamps,  0, inumpartials
	kF[] vecview gifreqs, 0, inumpartials

	if (gkFilterBw0 > 0 || gkFilterBw1 < 1) then
		kGain bpf kB, gkFilterBw0 - 0.01, 0, gkFilterBw0, 1, gkFilterBw1, 1, gkFilterBw1+0.001, 0
		; kGain cmp gkFilterBw0, "<=", kB, "<", gkFilterBw1
		kA *= kGain
	endif

	if (gkFilterActive == 1) then
		kGain bpf kF, gkFilterFreq0-30, 0.001, gkFilterFreq0, 1, gkFilterFreq1, 1, gkFilterFreq1+30, 0.001
		kA *= kGain
	endif
	
	iwavefn = -1  ; built-in sine
	iphases = -1  ; random phases
	aout beadsynt gkFreqScale, gkBwScale, gifreqs, giamps, gibws, \ 
	              inumpartials, iwavefn, iphases, giBeadsyntFlags

	aout *= gkgain

#ifdef STEREO
	outs aout, aout
#else
	outch 1, aout
#endif

exit:
endin

opcode setEndpos,0,k
	kendpos xin
	kendpos2 = min(kendpos, gkdataend)
	gkendpos = kendpos2 >= 0 ? kendpos2 : kendpos
endop

opcode setFilter,0,kkk
	kfilteractive, kfreq0, kfreq1 xin
	gkFilterActive = kfilteractive
	gkFilterFreq0 = kfreq0
	gkFilterFreq1 = kfreq1
endop

opcode changedRatelimit,kk,kk
	kvalue, ktrig xin
	klast init -1
	if ktrig == 1 && kvalue != klast then
		klast = kvalue
		ktrig2 = 1
	else
		ktrig2 = 0
	endif
	xout ktrig2, klast
endop


instr init
	setEdithead(gkedithead)
	; setPlay(1)
	setLoopmode(0)
	#ifdef ENDPOS
		setEndpos($ENDPOS)
	#else
		setEndpos(gkdur)
	#endif
	setSpeed(gkSpeed)
	#ifdef LOOPMODE
		puts "Loopmode $LOOPMODE", 1
		setLoopmode($LOOPMODE) 
	#endif
	#ifdef AUTOPLAY
		puts "playing!", 1
		setPlay(1)
	#endif
	turnoff
endin

instr setEditheadLine
	idur = p4
	idest = p5
	iorig = i(gkedithead)
	kpos linseg iorig, idur, idest
	setEdithead(kpos)
	ktime = timeinsts()
	if ktime >= idur then
		turnoff 
	endif
endin

instr setPlayheadLine
	idur = p4
	idest = p5
	iorig = i(gkplayhead)
	
	kpos linseg iorig, idur, idest
	gkplayhead = kpos
	ktime = timeinsts()
	if ktime >= idur then
		turnoff 
	endif
endin


instr osc
	prints "\n <<<< OSC listening port: %d >>>> \n\n", gioscport
	prints "\n <<<< OSC notify port   : %d >>>> \n\n", ginotifyport
	
	k0, k1, k2 init 0
	
	kk OSClisten giosc, "/play", "i", k0
	if (kk == 1) then
		setPlay(k0)
	endif
	
	; args: isplaying, looping
	kk OSClisten giosc, "/play", "ii", k0, k1
	if (kk == 1) then
		setPlay(k0)
		setLoopmode(k1)
	endif
	
	kk OSClisten giosc, "/loop", "i", k0
	if (kk == 1) then
		setLoopmode(k0)
	endif
	
	; args: pos
	kk OSClisten giosc, "/setpos", "f", k0
	if (kk == 1) then
		setEdithead(k0)
	endif
	
	kk OSClisten giosc, "/setposline", "ff", k0, k1
	if (kk == 1) then
		event "i", "setEditheadLine", 0, -1, k1, k0
	endif

	kk OSClisten giosc, "/setplayhead", "ff", k0, k1
	if (kk == 1) then
		event "i", "setPlayheadLine", 0, -1, k1, k0
	endif

	; args: pos
	kk OSClisten giosc, "/setendpos", "f", k0
	if (kk == 1) then
		setEndpos(k0)
	endif

	; args: speed
	kk OSClisten giosc, "/speed", "f", k0
	if (kk == 1) then
		setSpeed(k0)
	endif
	
	; args: isactive, freq0, freq1
	kk OSClisten giosc, "/filter", "iff", k0, k1, k2
	if (kk == 1) then
		setFilter(k0, k1, k2)
	endif
	
	; args: isactive
	kk OSClisten giosc, "/filter", "i", k0
	if (kk == 1) then
		setFilter(k0, gkFilterFreq0, gkFilterFreq1)
	endif

	; args: bw0, bw1 
	kk OSClisten giosc, "/bwfilter", "ff", k0, k1
	if (kk == 1) then
		gkFilterBw0 = k0 >= 0 ? k0 : gkFilterBw0
		gkFilterBw1 = k1 >= 0 ? k1 : gkFilterBw1
	endif

	; args: gain, noisegain 
	kk OSClisten giosc, "/gain", "ff", k0, k1
	if (kk == 1) then
		gkgain = k0 >= 0 ? k0 : gkgain
		gkBwScale = k1 >= 0 ? k1 : gkBwScale
	endif
	
	; args: discarded
	kk OSClisten giosc, "/exit", "i", k0
	if (kk == 1) then
		event "i", "exit", 0, -1
	endif
	
	ksendpos, kplayhead changedRatelimit gkplayhead, metro(giOscOutFreq)
	OSCsend ksendpos, "127.0.0.1", ginotifyport, "/pos", "f", gkplayhead

	kplaying_changed = changed(gkplay)
	OSCsend kplaying_changed, "127.0.0.1", ginotifyport, "/play", "i", gkplay
	
endin
	
instr exit
	scoreline_i "e 0"
endin

instr +post
#ifdef EXITWHENDONE
	puts ">>>>>>>>>>>>>>>>>>>> will exit when done", 1
#endif
	it = ksmps / sr
	if gkplay == 0 goto exit
	gkplayhead += it * gkSpeed
	if (gkplayhead > gkendpos) || (gkplayhead <= 0) then
		gkplayhead = gkedithead
		if (gkLoopmode==0) then
			setPlay(0)
#ifdef EXITWHENDONE
            event "i", "exit", 0.050, -1
#endif
		endif
		setPlay(gkLoopmode)
	endif
exit:
endin

instr play
	setPlay(1)
	setLoopmode(1)
	turnoff
endin

</CsInstruments>
<CsScore>
i "init" 0.004 -1
i "oscils" 0 -1
; i "ctrl" 0.004 -1
i "osc" 0 -1
; i "play" 0.010 -1 1
f0 36000
</CsScore>
</CsoundSynthesizer>

















<bsbPanel>
 <label>Widgets</label>
 <objectName/>
 <x>0</x>
 <y>0</y>
 <width>656</width>
 <height>114</height>
 <visible>true</visible>
 <uuid/>
 <bgcolor mode="nobackground">
  <r>255</r>
  <g>255</g>
  <b>255</b>
 </bgcolor>
 <bsbObject version="2" type="BSBCheckBox">
  <objectName>play</objectName>
  <x>4</x>
  <y>69</y>
  <width>20</width>
  <height>20</height>
  <uuid>{8a7d19c4-093e-40a5-a0dd-4fba60092f46}</uuid>
  <visible>true</visible>
  <midichan>0</midichan>
  <midicc>-3</midicc>
  <selected>true</selected>
  <label/>
  <pressedValue>1</pressedValue>
  <randomizable group="0">false</randomizable>
 </bsbObject>
 <bsbObject version="2" type="BSBLabel">
  <objectName/>
  <x>25</x>
  <y>69</y>
  <width>71</width>
  <height>21</height>
  <uuid>{9fe15e6b-e41b-4fc1-bab4-5c3bfa7baaa6}</uuid>
  <visible>true</visible>
  <midichan>0</midichan>
  <midicc>-3</midicc>
  <label>PLAY</label>
  <alignment>left</alignment>
  <font>Arial</font>
  <fontsize>10</fontsize>
  <precision>3</precision>
  <color>
   <r>0</r>
   <g>0</g>
   <b>0</b>
  </color>
  <bgcolor mode="nobackground">
   <r>255</r>
   <g>255</g>
   <b>255</b>
  </bgcolor>
  <bordermode>noborder</bordermode>
  <borderradius>1</borderradius>
  <borderwidth>1</borderwidth>
 </bsbObject>
 <bsbObject version="2" type="BSBController">
  <objectName>playhead</objectName>
  <x>3</x>
  <y>4</y>
  <width>640</width>
  <height>8</height>
  <uuid>{dc2faa27-eeeb-482f-9fb8-c47d6a284271}</uuid>
  <visible>true</visible>
  <midichan>0</midichan>
  <midicc>0</midicc>
  <objectName2/>
  <xMin>0.00000000</xMin>
  <xMax>1.00000000</xMax>
  <yMin>0.00000000</yMin>
  <yMax>1.00000000</yMax>
  <xValue>0.17021484</xValue>
  <yValue>0.00000000</yValue>
  <type>line</type>
  <pointsize>1</pointsize>
  <fadeSpeed>0.00000000</fadeSpeed>
  <mouseControl act="press">jump</mouseControl>
  <color>
   <r>0</r>
   <g>234</g>
   <b>0</b>
  </color>
  <randomizable mode="both" group="0">false</randomizable>
  <bgcolor>
   <r>0</r>
   <g>0</g>
   <b>0</b>
  </bgcolor>
 </bsbObject>
 <bsbObject version="2" type="BSBController">
  <objectName>edithead</objectName>
  <x>3</x>
  <y>13</y>
  <width>640</width>
  <height>20</height>
  <uuid>{0a2b4638-ae3b-40de-b343-70fd41aa7592}</uuid>
  <visible>true</visible>
  <midichan>0</midichan>
  <midicc>0</midicc>
  <objectName2/>
  <xMin>0.00000000</xMin>
  <xMax>1.00000000</xMax>
  <yMin>0.00000000</yMin>
  <yMax>1.00000000</yMax>
  <xValue>0.17031250</xValue>
  <yValue>0.00000000</yValue>
  <type>line</type>
  <pointsize>1</pointsize>
  <fadeSpeed>0.00000000</fadeSpeed>
  <mouseControl act="press">jump</mouseControl>
  <color>
   <r>0</r>
   <g>170</g>
   <b>255</b>
  </color>
  <randomizable mode="both" group="0">false</randomizable>
  <bgcolor>
   <r>0</r>
   <g>0</g>
   <b>0</b>
  </bgcolor>
 </bsbObject>
 <bsbObject version="2" type="BSBDisplay">
  <objectName>abstime</objectName>
  <x>598</x>
  <y>36</y>
  <width>44</width>
  <height>25</height>
  <uuid>{c411f856-3311-4755-b2e6-e9fa9f3cd448}</uuid>
  <visible>true</visible>
  <midichan>0</midichan>
  <midicc>-3</midicc>
  <label>0.857</label>
  <alignment>left</alignment>
  <font>Arial</font>
  <fontsize>12</fontsize>
  <precision>3</precision>
  <color>
   <r>0</r>
   <g>0</g>
   <b>0</b>
  </color>
  <bgcolor mode="nobackground">
   <r>255</r>
   <g>255</g>
   <b>255</b>
  </bgcolor>
  <bordermode>border</bordermode>
  <borderradius>1</borderradius>
  <borderwidth>1</borderwidth>
 </bsbObject>
 <bsbObject version="2" type="BSBSpinBox">
  <objectName>speed</objectName>
  <x>125</x>
  <y>63</y>
  <width>74</width>
  <height>33</height>
  <uuid>{6ea04370-5c54-40fe-bfa8-331918dfa374}</uuid>
  <visible>true</visible>
  <midichan>0</midichan>
  <midicc>0</midicc>
  <alignment>left</alignment>
  <font>Arial</font>
  <fontsize>14</fontsize>
  <color>
   <r>0</r>
   <g>0</g>
   <b>0</b>
  </color>
  <bgcolor mode="nobackground">
   <r>255</r>
   <g>255</g>
   <b>255</b>
  </bgcolor>
  <resolution>0.00100000</resolution>
  <minimum>-999</minimum>
  <maximum>999</maximum>
  <randomizable group="0">false</randomizable>
  <value>0</value>
 </bsbObject>
 <bsbObject version="2" type="BSBCheckBox">
  <objectName>loop</objectName>
  <x>3</x>
  <y>42</y>
  <width>20</width>
  <height>20</height>
  <uuid>{e64432e0-e165-43a8-81d7-596e6f7677e2}</uuid>
  <visible>true</visible>
  <midichan>0</midichan>
  <midicc>-3</midicc>
  <selected>false</selected>
  <label/>
  <pressedValue>1</pressedValue>
  <randomizable group="0">false</randomizable>
 </bsbObject>
 <bsbObject version="2" type="BSBLabel">
  <objectName/>
  <x>24</x>
  <y>42</y>
  <width>48</width>
  <height>21</height>
  <uuid>{959e7eaa-3175-40e8-8580-860e62e2961a}</uuid>
  <visible>true</visible>
  <midichan>0</midichan>
  <midicc>-3</midicc>
  <label>LOOP</label>
  <alignment>left</alignment>
  <font>Arial</font>
  <fontsize>10</fontsize>
  <precision>3</precision>
  <color>
   <r>0</r>
   <g>0</g>
   <b>0</b>
  </color>
  <bgcolor mode="nobackground">
   <r>255</r>
   <g>255</g>
   <b>255</b>
  </bgcolor>
  <bordermode>noborder</bordermode>
  <borderradius>1</borderradius>
  <borderwidth>1</borderwidth>
 </bsbObject>
 <bsbObject version="2" type="BSBLabel">
  <objectName/>
  <x>125</x>
  <y>41</y>
  <width>44</width>
  <height>23</height>
  <uuid>{8de6fafa-3e6c-4d00-b534-70549bed15e6}</uuid>
  <visible>true</visible>
  <midichan>0</midichan>
  <midicc>-3</midicc>
  <label>speed</label>
  <alignment>left</alignment>
  <font>Arial</font>
  <fontsize>10</fontsize>
  <precision>3</precision>
  <color>
   <r>0</r>
   <g>0</g>
   <b>0</b>
  </color>
  <bgcolor mode="nobackground">
   <r>255</r>
   <g>255</g>
   <b>255</b>
  </bgcolor>
  <bordermode>noborder</bordermode>
  <borderradius>1</borderradius>
  <borderwidth>1</borderwidth>
 </bsbObject>
</bsbPanel>
<bsbPresets>
</bsbPresets>
<EventPanel name="" tempo="60.00000000" loop="8.00000000" x="810" y="70" width="655" height="346" visible="true" loopStart="0" loopEnd="0">i 2 0 -1 </EventPanel>
