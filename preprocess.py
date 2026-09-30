"""
preprocess.py
-------------
Step 1 of the pipeline: turn a folder of MIDI files into a flat sequence of
"note events" that a neural network can learn from, using music21.

Each event is a string:
  - a single note is represented by its pitch, e.g. "C4" -> stored as "C4"
  - a chord is represented as dot-joined pitch ids, e.g. "4.8.11"
  - a rest is represented as "REST"

Usage:
    python preprocess.py --midi_dir data/midi_songs --out notes.pkl
"""
import argparse
import glob
import os
import pickle

from music21 import converter, instrument, note, chord


def get_notes_from_file(midi_path):
    """Extract a list of note/chord/rest event strings from a single MIDI file."""
    events = []
    try:
        midi = converter.parse(midi_path)
    except Exception as e:
        print(f"  [skip] could not parse {midi_path}: {e}")
        return events

    parts = instrument.partitionByInstrument(midi)
    notes_to_parse = parts.parts[0].recurse() if parts else midi.flat.notes

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            events.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            events.append(".".join(str(n) for n in element.normalOrder))
        elif isinstance(element, note.Rest):
            events.append("REST")

    return events


def build_corpus(midi_dir):
    """Walk every .mid/.midi file in midi_dir and concatenate their event sequences.

    Songs are kept back-to-back in one long list for simplicity (good enough
    for a first model). For higher quality, keep them as separate sequences
    and pad/truncate per-song in prepare_sequences().
    """
    midi_files = sorted(
        glob.glob(os.path.join(midi_dir, "**", "*.mid"), recursive=True)
        + glob.glob(os.path.join(midi_dir, "**", "*.midi"), recursive=True)
    )

    if not midi_files:
        raise FileNotFoundError(
            f"No .mid/.midi files found under '{midi_dir}'. "
            f"Add some MIDI files there first (see README.md)."
        )

    all_events = []
    print(f"Found {len(midi_files)} MIDI file(s). Parsing...")
    for i, path in enumerate(midi_files, 1):
        print(f"[{i}/{len(midi_files)}] {os.path.basename(path)}")
        all_events.extend(get_notes_from_file(path))

    print(f"Total events extracted: {len(all_events)}")
    return all_events


def main():
    parser = argparse.ArgumentParser(description="Preprocess MIDI files into note events.")
    parser.add_argument("--midi_dir", default="data/midi_songs", help="Folder containing .mid/.midi files")
    parser.add_argument("--out", default="notes.pkl", help="Output pickle file for the event list")
    args = parser.parse_args()

    events = build_corpus(args.midi_dir)

    with open(args.out, "wb") as f:
        pickle.dump(events, f)
    print(f"Saved {len(events)} events to {args.out}")


if __name__ == "__main__":
    main()
