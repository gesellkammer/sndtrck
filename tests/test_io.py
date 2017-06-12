import sndtrck


## IO
def test_hdf5_io():
	spectrum = sndtrck.readspectrum("test.sdif")
	spectrum.write("test2.h5")
	spectrum2 = sndtrck.readspectrum("test2.h5")
	assert spectrum == spectrum2

def test_sdif_io():
	spectrum1 = sndtrck.readspectrum("test.sdif")
	spectrum1.write("test2.sdif")
	spectrum2 = sndtrck.readspectrum("test2.sdif")
	assert spectrum1 == spectrum2

def test_txt_io():
	spectrum1 = sndtrck.readspectrum("test.sdif")
	spectrum1.write("test2.txt")
	spectrum2 = sndtrck.readspectrum("test2.sdif")
	assert spectrum1 == spectrum2

