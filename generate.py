"""
generate.py
-----------
Loads a trained model's weights and the saved vocabulary, seeds the network
with a random window from the training data, and autoregressively samples
a new sequence of note/chord/rest events. The sequence is then converted
back into a music21 Stream and written out as a .mid file.

Usage:
    python generate.py --weights checkpoints/final.weights.h5 --notes notes.pkl \
                        --vocab vocab.pkl --length 500 --out output/generated.mid
"""
import argparse
import pickle
import random

import numpy as np
from music21 import stream, note, chord, instrument

from model import build_model, SEQUENCE_LENGTH


def sample_with_temperature(probabilities, temperature=1.0):
    """Sample an index from a probability distribution, with temperature control.
    temperature < 1.0 -> more conservative / repetitive
    temperature > 1.0 -> more random / adventurous
    """
    probabilities = np.asarray(probabilities).astype("float64")
    probabilities = np.log(probabilities + 1e-9) / temperature
    exp_probs = np.exp(probabilities)
    probabilities = exp_probs / np.sum(exp_probs)
    return np.random.choice(len(probabilities), p=probabilities)


def generate_events(model, network_input, int_to_note, n_vocab, length, temperature):
    """Autoregressively generate `length` new events starting from a random seed window."""
    start = random.randint(0, len(network_input) - 1)
    pattern = list(network_input[start])  # list of ints, length == sequence_length

    prediction_output = []
    for _ in range(length):
        input_seq = np.reshape(pattern, (1, len(pattern), 1)) / float(n_vocab)
        prediction = model.predict(input_seq, verbose=0)[0]

        index = sample_with_temperature(prediction, temperature)
        result = int_to_note[index]
        prediction_output.append(result)

        pattern.append(index)
        pattern = pattern[1:]  # slide the window forward

    return prediction_output


def events_to_midi(events, out_path, step_duration=0.5):
    """Convert a list of event strings ("C4", "4.8.11", "REST") into a MIDI file."""
    output_stream = stream.Stream()
    output_stream.append(instrument.Piano())

    offset = 0.0
    for event in events:
        if event == "REST":
            new_note = note.Rest()
            new_note.offset = offset
            output_stream.append(new_note)
        elif "." in event:
            # chord: dot-joined pitch class ids
            chord_notes = [note.Note(int(n)) for n in event.split(".")]
            new_chord = chord.Chord(chord_notes)
            new_chord.offset = offset
            output_stream.append(new_chord)
        else:
            new_note = note.Note(event)
            new_note.offset = offset
            output_stream.append(new_note)
        offset += step_duration

    output_stream.write("midi", fp=out_path)
    print(f"Saved generated MIDI to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate new music from a trained model.")
    parser.add_argument("--weights", required=True, help="Path to trained .weights.h5 file")
    parser.add_argument("--notes", default="notes.pkl", help="Pickled event list (used to seed generation)")
    parser.add_argument("--vocab", default="vocab.pkl", help="Pickled (note_to_int, int_to_note) mappings")
    parser.add_argument("--length", type=int, default=500, help="Number of events to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--sequence_length", type=int, default=SEQUENCE_LENGTH)
    parser.add_argument("--out", default="output/generated.mid")
    args = parser.parse_args()

    with open(args.notes, "rb") as f:
        events = pickle.load(f)
    with open(args.vocab, "rb") as f:
        note_to_int, int_to_note = pickle.load(f)

    n_vocab = len(note_to_int)

    # Rebuild the same windows used at training time, just to get seed material
    network_input = [
        [note_to_int[e] for e in events[i:i + args.sequence_length]]
        for i in range(len(events) - args.sequence_length)
    ]

    model = build_model(args.sequence_length, n_vocab)
    model.load_weights(args.weights)

    generated = generate_events(model, network_input, int_to_note, n_vocab, args.length, args.temperature)
    events_to_midi(generated, args.out)


if __name__ == "__main__":
    main()
