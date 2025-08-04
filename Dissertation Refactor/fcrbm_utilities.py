import tensorflow as tf
import numpy as np
import librosa
import os

class FCRBM(tf.Module):
    def __init__(self, visible_dim, hidden_dim, style_dim, history_dim, k=1):
        super(FCRBM, self).__init__()

        self.visible_dim = visible_dim # this is the number of frequency bins in spectrogram, (n_fft / 2) + 1 because
        # FFT is symmetric, so only need half... e.g if n_fft is 2048 visible_dim = (2048 / 2) + 1 = 1025
        self.hidden_dim = hidden_dim
        self.style_dim = style_dim
        self.history_dim = history_dim
        self.k = k

        # initialize weights and biases
        self.visible_bias = tf.Variable(tf.zeros([visible_dim]), dtype=tf.float32, name='visible_bias')
        self.hidden_bias = tf.Variable(tf.zeros([hidden_dim]), dtype=tf.float32, name='hidden_bias')

        self.W_vh = tf.Variable(tf.random.normal([visible_dim, hidden_dim], stddev=0.01), dtype=tf.float32, name='W_vh')
        self.W_vu = tf.Variable(tf.random.normal([visible_dim, history_dim], stddev=0.01), dtype=tf.float32, name='W_vu')
        self.W_hu = tf.Variable(tf.random.normal([hidden_dim, history_dim], stddev=0.01), dtype=tf.float32, name='W_hu')
        self.W_vh_style = tf.Variable(tf.random.normal([visible_dim, hidden_dim, style_dim], stddev=0.01), dtype=tf.float32, name='W_vh_style')

    @tf.function
    def _hidden_prob(self, v, u, y):
        # calculate dynamic hidden biases
        dynamic_hidden_bias = self.hidden_bias + tf.linalg.matmul(u, self.W_hu, transpose_b=True)
        
        # calculate modulated weights based on style
        modulated_weights = self.W_vh + tf.einsum('bs,vhs->bvh', y, self.W_vh_style)
        
        # calculate hidden probabilities
        energy_term = tf.einsum('bv,bvh->bh', v, modulated_weights)
        
        return tf.sigmoid(energy_term + dynamic_hidden_bias)

    @tf.function
    def _visible_mean(self, h, u):
        # calculate dynamic visible biases
        dynamic_visible_bias = self.visible_bias + tf.linalg.matmul(h, self.W_vh, transpose_b=True)
        return dynamic_visible_bias # mean for a gaussian visible layer

    @tf.function
    def gibbs_sampling_k_steps(self, v_init, u, y):
        v_current = v_init
        for _ in tf.range(self.k):
            h_prob = self._hidden_prob(v_current, u, y)
            h_sample = tf.cast(tf.random.uniform(tf.shape(h_prob)) < h_prob, tf.float32)
            
            v_mean = self._visible_mean(h_sample, u)
            v_current = v_mean
            
        return v_current, h_sample

    @tf.function
    def train_step(self, v_pos, u, y, optimizer):
        with tf.GradientTape() as tape:
            # --- Positive Phase ---
            h_pos_prob = self._hidden_prob(v_pos, u, y)
            # sample hidden units from the probabilities
            h_pos_sample = tf.cast(tf.random.uniform(tf.shape(h_pos_prob)) < h_pos_prob, tf.float32)

            # positive phase association term
            # this is the "ideal" state of the model.
            pos_v_h = tf.linalg.matmul(v_pos, h_pos_prob, transpose_a=True)
            pos_v_u = tf.linalg.matmul(v_pos, u, transpose_a=True)
            pos_h_u = tf.linalg.matmul(h_pos_prob, u, transpose_a=True)
            
            # --- Negative Phase ---
            v_neg, h_neg_sample = self.gibbs_sampling_k_steps(v_pos, u, y)

            # negative phase association term
            # this is the "undesirable" state of the model.
            neg_v_h = tf.linalg.matmul(v_neg, h_neg_sample, transpose_a=True)
            neg_v_u = tf.linalg.matmul(v_neg, u, transpose_a=True)
            neg_h_u = tf.linalg.matmul(h_neg_sample, u, transpose_a=True)
            
            # --- Loss Calculation ---
            # The "loss" is the difference between the energy terms of the positive and negative phases
            # reconstruction error as proxy
            # for an RBM, the gradient is important so compute the gradients based on the difference of associationsdirectly.
            
            # The "loss" for the tape is not a single value but a set of updates
            # that is manually applied. This is a common pattern for RBMs.
            
            # Calculate the gradient for each parameter based on the difference of associations
            # Note: The gradient for W_vh_style is more complex due to the einsum.
            grad_W_vh = (pos_v_h - neg_v_h)
            grad_W_vu = (pos_v_u - neg_v_u)
            grad_W_hu = (pos_h_u - neg_h_u)
            grad_visible_bias = tf.reduce_mean(v_pos - v_neg, axis=0)
            grad_hidden_bias = tf.reduce_mean(h_pos_prob - h_neg_sample, axis=0)
            
            # The gradient for the three-way interaction term is more complex.
            # We need to compute the association for each style dimension.
            pos_association_style = tf.einsum('bv,bh,bs->vhs', v_pos, h_pos_prob, y)
            neg_association_style = tf.einsum('bv,bh,bs->vhs', v_neg, h_neg_sample, y)
            grad_W_vh_style = pos_association_style - neg_association_style
            
            # Collect all gradients
            gradients = [grad_W_vh, grad_W_vu, grad_W_hu, grad_visible_bias, grad_hidden_bias, grad_W_vh_style]
            
        # Manually apply gradients
        # The optimizer takes care of learning rate and momentum
        trainable_vars = [self.W_vh, self.W_vu, self.W_hu, self.visible_bias, self.hidden_bias, self.W_vh_style]
        optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # return a value for logging purposes, e.g., reconstruction error
        reconstruction_error = tf.reduce_mean(tf.square(v_pos - v_neg))
        return reconstruction_error

def synthesize_audio(model, u_seed, y_seed, num_frames_to_generate, n_fft, hop_length, sr):
    """
    Synthesizes a new audio spectrogram using the trained FCRBM.

    Args:
        model (FCRBM): The trained FCRBM model.
        u_seed (tf.Tensor): The initial spectrogram frame to start generation (history seed).
        y_seed (tf.Tensor): The style vector to maintain throughout the synthesis.
        num_frames_to_generate (int): The number of new spectrogram frames to create.
        n_fft (int): FFT window size.
        hop_length (int): Hop length for STFT.
        sr (int): Sampling rate for audio reconstruction.

    Returns:
        np.array: The generated audio waveform.
    """
    generated_spectrogram_frames = []
    
    # Initialize the current visible state with the history seed.
    # The visible state is the spectrogram frame at the current time step (v_t).
    # The history is the spectrogram frame from the previous time step (v_{t-1}).
    # So, for the first generation step, the history is the seed you provide.
    v_current = tf.expand_dims(u_seed, 0)
    
    # Expand style seed to match batch dimensions.
    # We want to maintain a constant style throughout the generation.
    y_current = tf.expand_dims(y_seed, 0)

    for _ in range(num_frames_to_generate):
        # 1. Sample the hidden state (h) from the current visible state (v)
        #    and the conditional inputs (u, y).
        #    Here, the history for the current step is the previous visible state.
        h_prob = model._hidden_prob(v_current, v_current, y_current)
        h_sample = tf.cast(tf.random.uniform(tf.shape(h_prob)) < h_prob, tf.float32)

        # 2. Reconstruct a new visible state (v_new) from the sampled hidden state.
        #    This v_new is the new spectrogram frame.
        v_mean = model._visible_mean(h_sample, v_current)
        
        # 3. Use the reconstructed visible state as the input for the next time step.
        v_current = v_mean
        
        # 4. Store the newly generated spectrogram frame.
        generated_spectrogram_frames.append(v_current.numpy()[0])
        
    generated_spectrogram = np.array(generated_spectrogram_frames).T

    # Convert the spectrogram back to an audio waveform using the Griffin-Lim algorithm.
    # This algorithm is used to estimate the phase information that was lost during STFT.
    phase = np.zeros_like(generated_spectrogram)
    D = generated_spectrogram * np.exp(1j * phase)
    audio = librosa.griffinlim(D, n_fft=n_fft, hop_length=hop_length)
    
    return audio

# convert spectrogram to audio
def spectrogram_to_audio(spectrogram, sr, n_fft, hop_length):
    phase = np.zeros_like(spectrogram)
    D = spectrogram * np.exp(1j * phase)
    return librosa.griffinlim(D, n_fft=n_fft, hop_length=hop_length)

def process_audio_files(audio_dir, n_fft, hop_length, sr):
    """
    Loads all .wav files from a directory, computes their spectrograms,
    and pads them to a common length for training.
    
    Args:
        audio_dir (str): Path to the directory containing audio files.
        n_fft (int): Number of FFT components.
        hop_length (int): Number of samples between successive FFTs.
        sr (int): Sampling rate of the audio.
    
    Returns:
        tuple: A tuple containing:
            - spectrogram_data (np.array): A concatenated array of all spectrogram frames.
            - style_data (np.array): A concatenated array of one-hot style vectors.
            - visible_dim (int): The number of frequency bins (FFT components).
    """
    all_spectrograms = []
    file_list = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])
    
    if not file_list:
        raise ValueError(f"No .wav files found in directory: {audio_dir}")
        
    num_audio_files = len(file_list)
    if num_audio_files != 9:
        print(f"Warning: Expected 9 audio files, but found {num_audio_files}. Proceeding anyway.")
    
    # Process each audio file
    for filename in file_list:
        filepath = os.path.join(audio_dir, filename)
        y, _ = librosa.load(filepath, sr=sr)
        S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        S_mag = np.abs(S)
        all_spectrograms.append(S_mag)

    # Pad all spectrograms to the same length for batching
    max_len = max(s.shape[1] for s in all_spectrograms)
    padded_spectrograms = []
    for s in all_spectrograms:
        padded = np.pad(s, ((0, 0), (0, max_len - s.shape[1])), 'constant')
        padded_spectrograms.append(padded)

    # Combine all data into a single numpy array
    spectrogram_data = np.concatenate([s.T for s in padded_spectrograms], axis=0)
    visible_dim = spectrogram_data.shape[1]

    # Create a style vector for each audio file
    style_vectors = np.eye(num_audio_files)

    # Expand the style vectors to match the number of frames
    style_data = []
    for i, s in enumerate(padded_spectrograms):
        style_data.append(np.tile(style_vectors[i], (s.shape[1], 1)))

    style_data = np.concatenate(style_data, axis=0)
    
    return spectrogram_data, style_data, visible_dim