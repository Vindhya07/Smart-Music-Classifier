import librosa
import numpy
import pandas
import os
import sklearn
import config
import CreateDataset

song_name = "output.wav"

def main():
	samp_rate = config.CreateDataset.SAMPLING_RATE
	frame_size = config.CreateDataset.FRAME_SIZE
	hop_size = config.CreateDataset.HOP_SIZE

	print("Extracting features from audio...")
	sample_array = get_sample_arrays(song_name, samp_rate)	
	test_list = extract_features(sample_array, samp_rate, frame_size, hop_size)
	print(test_list)
	test_numpy = numpy.array(test_list)
	
	print(test_numpy)
	test_numpy = test_numpy.reshape(1,-1)

	data = pandas.read_csv("not_norm.csv")
	print(data)
	data_values=numpy.array(data)
	print(data_values)
	
	print("Normalizing the data...")
	
	scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
	norm_data = scaler.fit_transform(data_values)
	test_numpy = scaler.transform(test_numpy)
	
	print(test_numpy)
	
	Feature_Names = ['meanZCR', 'stdZCR', 'meanSpecCentroid', 'stdSpecCentroid', 'meanSpecContrast', 'stdSpecContrast',
					 'meanSpecBandwidth', 'stdSpecBandwidth', 'meanSpecRollof', 'stdSpecRollof',
					 'meanMFCC_1', 'stdMFCC_1', 'meanMFCC_2', 'stdMFCC_2', 'meanMFCC_3', 'stdMFCC_3',
					 'meanMFCC_4', 'stdMFCC_4', 'meanMFCC_5', 'stdMFCC_5', 'meanMFCC_6', 'stdMFCC_6',
					 'meanMFCC_7', 'stdMFCC_7', 'meanMFCC_8', 'stdMFCC_8', 'meanMFCC_9', 'stdMFCC_9',
					 'meanMFCC_10', 'stdMFCC_10', 'meanMFCC_11', 'stdMFCC_11', 'meanMFCC_12', 'stdMFCC_12',
					 'meanMFCC_13', 'stdMFCC_13'
					 ]
	test_pandas = pandas.DataFrame(test_numpy, columns=Feature_Names)
	test_pandas.to_csv("test_set.csv", index=False)
	return test_numpy



def get_sample_arrays(song_name, samp_rate):
	
	x, sr = librosa.load(song_name, sr=samp_rate, offset=0.0, duration=5.0)
	print(x.shape)
	y, sr = librosa.load(song_name, sr=samp_rate, offset=12.0, duration=5.0)
	print("Y",y.shape)
	z, sr = librosa.load(song_name, sr=samp_rate, offset=24.0, duration=5.0)
	print("z",z.shape)     

	final = []
	j=0
	for j in range (y.size):
		avg = (x.item(j)+y.item(j)+z.item(j))/3
		final.append(avg)
	audios_numpy = numpy.array(final)
	return audios_numpy


def extract_features(signal, sample_rate, frame_size, hop_size):
	zero_crossing_rate = librosa.feature.zero_crossing_rate(y=signal, frame_length=frame_size, hop_length=hop_size)
	spectral_centroid = librosa.feature.spectral_centroid(y=signal, sr=sample_rate, n_fft=frame_size,
														  hop_length=hop_size)
	spectral_contrast = librosa.feature.spectral_contrast(y=signal, sr=sample_rate, n_fft=frame_size,
														  hop_length=hop_size)
	spectral_bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sample_rate, n_fft=frame_size,
															hop_length=hop_size)
	spectral_rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sample_rate, n_fft=frame_size, hop_length=hop_size)
	mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_fft=frame_size, hop_length=hop_size)

	return [

		numpy.mean(zero_crossing_rate),
		numpy.std(zero_crossing_rate),
		numpy.mean(spectral_centroid),
		numpy.std(spectral_centroid),
		numpy.mean(spectral_contrast),
		numpy.std(spectral_contrast),
		numpy.mean(spectral_bandwidth),
		numpy.std(spectral_bandwidth),
		numpy.mean(spectral_rolloff),
		numpy.std(spectral_rolloff),

		numpy.mean(mfccs[1, :]),
		numpy.std(mfccs[1, :]),
		numpy.mean(mfccs[2, :]),
		numpy.std(mfccs[2, :]),
		numpy.mean(mfccs[3, :]),
		numpy.std(mfccs[3, :]),
		numpy.mean(mfccs[4, :]),
		numpy.std(mfccs[4, :]),
		numpy.mean(mfccs[5, :]),
		numpy.std(mfccs[5, :]),
		numpy.mean(mfccs[6, :]),
		numpy.std(mfccs[6, :]),
		numpy.mean(mfccs[7, :]),
		numpy.std(mfccs[7, :]),
		numpy.mean(mfccs[8, :]),
		numpy.std(mfccs[8, :]),
		numpy.mean(mfccs[9, :]),
		numpy.std(mfccs[9, :]),
		numpy.mean(mfccs[10, :]),
		numpy.std(mfccs[10, :]),
		numpy.mean(mfccs[11, :]),
		numpy.std(mfccs[11, :]),
		numpy.mean(mfccs[12, :]),
		numpy.std(mfccs[12, :]),
		numpy.mean(mfccs[13, :]),
		numpy.std(mfccs[13, :]),
	]


main()
