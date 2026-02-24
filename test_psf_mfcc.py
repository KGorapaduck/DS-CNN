import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
import python_speech_features as psf

# Generate dummy audio 1 second
SAMPLE_RATE = 16000
audio = np.random.uniform(-1.0, 1.0, size=(16000,)).astype(np.float32)

# --- TF MFCC ---
sess = tf.Session()
wav_placeholder = tf.placeholder(tf.float32, [16000, 1])
spectrogram = contrib_audio.audio_spectrogram(
    wav_placeholder, window_size=int(16000 * 0.04), stride=int(16000 * 0.02), magnitude_squared=True)
mfcc_op = contrib_audio.mfcc(spectrogram, 16000, dct_coefficient_count=10)
tf_mfcc = sess.run(mfcc_op, feed_dict={wav_placeholder: audio.reshape(-1, 1)})
sess.close()

# --- PSF (python_speech_features) MFCC ---
# contrib_audio.mfcc properties: 
# window=40ms, stride=20ms, sample_rate=16kHz, dct=10
# Note context: TF's mfcc internally uses a 40-channel mel filterbank by default and extracts coefficients 0..9 (or 1..10?) in DCT.
psf_mfcc = psf.mfcc(audio, samplerate=SAMPLE_RATE, winlen=0.04, winstep=0.02,
                    numcep=10, nfilt=40, nfft=1024, lowfreq=20, highfreq=4000, preemph=0.0)

print(f"TF shape: {tf_mfcc.shape}")
print(f"PSF shape: {psf_mfcc.shape}")

# Print first few rows to see if ranges match
print("TF MFCC first row:\n", tf_mfcc[0, 0, :])
print("PSF MFCC first row:\n", psf_mfcc[0, :])

# Test similarity (just mean/max difference)
diff = np.abs(tf_mfcc.reshape(psf_mfcc.shape) - psf_mfcc)
print(f"Mean Abs Error: {np.mean(diff)}")
print(f"Max Abs Error: {np.max(diff)}")
