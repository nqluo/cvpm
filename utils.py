import numpy as np
from scipy.signal import periodogram
from scipy import signal

def array_update(array, val):
	array_tmp = np.delete(array, 0)
	array_tmp = np.append(array_tmp, val)
	return array_tmp

def ppg_filter(ppg, frq=30, fL = 0.5, fH = 5): 
    order = 4  # filter order of 4*2 = 8
    stpa = 20  # 40 dB of stopband attenuation, amplitude decrease to 1%
#     fL = 0.5  # lower stopband frequency
#     fH = 5  # higher stopband frequency
    b, a = signal.cheby2(order, stpa, [fL, fH], btype = 'bandpass', fs = frq)

    ppg_filter = signal.filtfilt(b, a, ppg)

    return ppg_filter



def cal_hr(rPPG, time):
	rPPG = np.array(rPPG)
	fps = len(rPPG) / (time[-1] - time[0])# the real frame_fps is not stable, should be calculated
	## filter the rPPG before the FFT
	even_time = np.linspace(time[0], time[-1], len(rPPG))
	even_rPPG = np.interp(even_time, time, rPPG)
	even_rPPG = np.hamming(len(rPPG)) * even_rPPG
	even_rPPG = even_rPPG - np.mean(even_rPPG)

	f, Pxx_den = signal.periodogram(even_rPPG, fps)
	idx = np.argmax(Pxx_den)
	fmax = f[idx]
	if 0.5<=fmax<=2.5:
		hr = int(60.* fmax)
	else:
		hr = np.nan

	return hr


def pos_rppg(r, g, b, frame_fps):
	projection = np.array([[0, 1, -1],[-2, 1, 1]]) # 2*3 array
	X = np.array([r, g, b]) # 3 chanel signal: 3*N array
	clen = X.shape[1] # channel length of the RGB 
	wlen = int(1.6*frame_fps) # moving window size, wlen = 48, 1.6*30fps
	H = np.zeros(clen)

	## when the input signal length <= wlen
	if 1<=clen <= wlen:
		CN = X[:,:]
		CN = CN / np.mean(CN, axis = 1, keepdims = True)
		S = np.dot(projection, CN)
		H = S[0,:] + np.std(S[0,:]) / np.std(S[1,:]) * S[1,:]
		return H

	## when the input signal length > wlen, standard POS
	for n in range(wlen, clen): # wlen: moving window size, clen: channel data size
		m = n - wlen # m >= 0
		CN = X[:,m:n+1]
		CN = CN / np.mean(CN, axis = 1, keepdims = True) # temporal normalization
		S = np.dot(projection, CN) # projection, S: 2*N array
		HN = S[0,:] + np.std(S[0,:]) / np.std(S[1,:]) * S[1,:] # tuning, HN: 1*N array
		H[m:n+1] = H[m:n+1] + (HN - np.mean(HN)) # overlap-adding
	return H


def roi_shrink(x, y, w, h, keep = 0.8):
	x_move = int((1 - keep) * w / 2)
	y_move = int((1 - keep) * h /2)
	x = x+x_move
	y = y+y_move
	w = int(w * keep)
	h = int(h * keep)
	return (x, y, w, h)









