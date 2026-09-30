"""
model.py
--------
Turns the raw event list from preprocess.py into training sequences, and
defines the LSTM network that learns to predict the next note event.
"""
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation, BatchNormalization


SEQUENCE_LENGTH = 100  # how many previous events the model looks at to predict the next one


def build_vocab(events):
    """Map each unique event string to an integer id (and back)."""
    pitch_names = sorted(set(events))
    note_to_int = {note: i for i, note in enumerate(pitch_names)}
    int_to_note = {i: note for note, i in note_to_int.items()}
    return note_to_int, int_to_note


def prepare_sequences(events, note_to_int, sequence_length=SEQUENCE_LENGTH):
    """Build (X, y) training pairs.

    X: overlapping windows of `sequence_length` consecutive events (normalized to [0,1])
    y: the event immediately following each window, one-hot encoded
    """
    n_vocab = len(note_to_int)
    network_input = []
    network_output = []

    for i in range(len(events) - sequence_length):
        seq_in = events[i:i + sequence_length]
        seq_out = events[i + sequence_length]
        network_input.append([note_to_int[e] for e in seq_in])
        network_output.append(note_to_int[seq_out])

    n_patterns = len(network_input)
    if n_patterns == 0:
        raise ValueError(
            f"Not enough events ({len(events)}) for sequence_length={sequence_length}. "
            f"Add more MIDI data or lower sequence_length."
        )

    # Reshape for LSTM: (samples, timesteps, features) and normalize
    X = np.reshape(network_input, (n_patterns, sequence_length, 1))
    X = X / float(n_vocab)

    # One-hot encode the targets
    y = np.zeros((n_patterns, n_vocab), dtype=np.float32)
    for idx, val in enumerate(network_output):
        y[idx, val] = 1.0

    return X, y


def build_model(sequence_length, n_vocab):
    """3-layer stacked LSTM, similar to the well-known Skuli architecture for
    symbolic music generation. Works reasonably well on a single GPU."""
    model = Sequential([
        LSTM(512, input_shape=(sequence_length, 1), return_sequences=True),
        Dropout(0.3),
        LSTM(512, return_sequences=True),
        Dropout(0.3),
        LSTM(512),
        BatchNormalization(),
        Dense(256),
        Dropout(0.3),
        Activation("relu"),
        Dense(n_vocab),
        Activation("softmax"),
    ])
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
    return model
