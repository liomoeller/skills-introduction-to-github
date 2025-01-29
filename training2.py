import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense, Input, Masking
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

# Generate sequences of different lengths
def generate_data(samples=200, min_len=50, max_len=60):
    X, Y = [], []
    for _ in range(samples):
        seq_len = np.random.randint(min_len, max_len)
        x = np.arange(seq_len)  # Time steps
        y = 2 * x + 1  # Linear function: y = 2x + 1
        X.append(x)
        Y.append(y)
    return X, Y

# Generate training data
X, Y = generate_data()

# Plot an example sequence
plt.plot(X[0], Y[0], label="Example Sequence", color="red")
plt.legend()
plt.show()

# Prepare decoder inputs by shifting the Y values to the right
decoder_inputs_shifted = [np.concatenate([[0], y[:-1]]) for y in Y]

# Determine maximum sequence length
max_len = max(len(seq) for seq in Y)

# Pad sequences to make them equal in length
X_padded = pad_sequences(X, maxlen=max_len, padding='post', dtype='float32')
decoder_inputs_shifted_padded = pad_sequences(decoder_inputs_shifted, maxlen=max_len, padding='post', dtype='float32', value=-1)
Y_padded = pad_sequences(Y, maxlen=max_len, padding='post', dtype='float32', value=0)

# Reshape for LSTM input (batch_size, time_steps, features)
X_padded = np.expand_dims(X_padded, axis=-1)
decoder_inputs_shifted_padded = np.expand_dims(decoder_inputs_shifted_padded, axis=-1)
Y_padded = np.expand_dims(Y_padded, axis=-1)

# Encoder
encoder_inputs = Input(shape=(None, 1))  
encoder_masking = Masking(mask_value=-1)(encoder_inputs)  # Ignore padding (0)
encoder_lstm = LSTM(64, return_state=True)  
encoder_outputs, state_h, state_c = encoder_lstm(encoder_masking)

# Decoder
decoder_inputs = Input(shape=(None, 1))
decoder_masking = Masking(mask_value=-1)(decoder_inputs)
decoder_lstm = LSTM(64, return_sequences=True)
decoder_outputs = decoder_lstm(decoder_masking, initial_state=[state_h, state_c])
decoder_dense = Dense(1)  # Predict one value per timestep
decoder_outputs = decoder_dense(decoder_outputs)

# Define and compile the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer="adam", loss="mse")

# Show model summary
model.summary()

# Use early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)

# Train the model
history = model.fit(
    [X_padded, decoder_inputs_shifted_padded],  
    Y_padded,  
    epochs=1000,  
    batch_size=16,  
    validation_split=0.1,  # Hold out 10% of data for validation
    callbacks=[early_stopping]  
)

# Plot the training loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history.get('val_loss', []), label='Validation Loss', linestyle='dashed')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.title('Training Progress')
plt.show()

# TESTING THE MODEL
test_seq = np.arange(55)  # Example sequence of length 55
test_seq_padded = pad_sequences([test_seq], maxlen=max_len, padding='post', dtype='float32')
test_seq_padded = np.expand_dims(test_seq_padded, axis=-1)  # Reshape for LSTM

# Sequentially generate predictions
decoder_input = np.zeros((1, max_len, 1))  # Start with all zeros
predicted_seq = np.zeros((max_len,))

# Iteratively generate predictions timestep by timestep
for t in range(max_len):
    pred = model.predict([test_seq_padded, decoder_input])
    predicted_seq[t] = pred[0, t, 0]
    if t < max_len - 1:
        decoder_input[0, t + 1, 0] = predicted_seq[t]  # Feed prediction as next input

# Remove padding
predicted_seq = predicted_seq[:len(test_seq)]

# True values for comparison
true_values = 2 * test_seq + 1  

# Plot predictions vs actual values
plt.figure(figsize=(10, 5))
plt.plot(test_seq, true_values, label="True Values (y=2x+1)", linestyle="dashed", color="blue")
plt.plot(test_seq, predicted_seq, label="Model Prediction", color="red")
plt.xlabel("Timesteps")
plt.ylabel("Values")
plt.legend()
plt.title("Predictions vs True Values")
plt.show()
