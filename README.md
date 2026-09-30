# AI Music Generation (LSTM + music21)

Generates new music by learning note/chord patterns from a folder of MIDI
files, using an LSTM sequence model, then writes the result back out as a
playable `.mid` file.

## Pipeline

```
MIDI files  --preprocess.py-->  notes.pkl (event sequence)
notes.pkl   --train.py-------->  checkpoints/*.weights.h5, vocab.pkl
weights     --generate.py----->  output/generated.mid
```

## 1. Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

A GPU is strongly recommended for training (CPU training on a few hundred
songs can take hours per epoch).

## 2. Get MIDI data

Put `.mid` / `.midi` files into `data/midi_songs/` (subfolders are fine).
Good free sources for classical piano MIDI:
- The "Classical Piano Midi Page" (Bach, Beethoven, Chopin, etc.)
- The Lakh MIDI Dataset (large, multi-genre)
- MAESTRO dataset (Google Magenta) — high quality solo piano performances

A few hundred songs of one genre/composer works best; mixing wildly
different styles will make the model's output less coherent.

## 3. Preprocess

```bash
python preprocess.py --midi_dir data/midi_songs --out notes.pkl
```

This walks every MIDI file, extracts notes, chords and rests with
`music21`, and saves them as one flat list of event strings.

## 4. Train

```bash
python train.py --notes notes.pkl --epochs 100 --batch_size 64
```

- Builds a vocabulary from every unique event (`vocab.pkl`).
- Slices the event list into overlapping 100-event windows (`X`) with the
  following event as the label (`y`).
- Trains a 3-layer stacked LSTM (`model.py`) with dropout and batch
  normalization, checkpointing the best weights to `checkpoints/`.

Useful flags: `--sequence_length` (context window size), `--epochs`,
`--batch_size`.

## 5. Generate new music

```bash
python generate.py --weights checkpoints/final.weights.h5 \
                    --length 500 --temperature 1.0 \
                    --out output/generated.mid
```

- Seeds the model with a random 100-event window from the training data.
- Autoregressively predicts one event at a time, feeding each prediction
  back in as context for the next (sliding window).
- `--temperature` controls randomness: <1.0 = safer/more repetitive,
  >1.0 = more adventurous (and more likely to sound chaotic).
- Converts the generated event list back into notes/chords/rests with
  `music21` and writes a standard MIDI file you can open in any DAW,
  GarageBand, MuseScore, or play with `python -c "import music21;
  music21.converter.parse('output/generated.mid').show('midi')"` in a
  Jupyter notebook.

## Tips for better results

- **More data, one style**: 100+ MIDI files of a consistent style (e.g.
  solo piano, one composer) generalizes much better than a tiny or very
  mixed dataset.
- **Longer training**: loss should keep dropping for many epochs;
  `EarlyStopping` (patience=10) is already wired in to stop once it plateaus.
- **Sequence length**: longer context (e.g. 150–200) captures more musical
  structure but needs more data and memory.
- **Swap in a GAN**: this project uses an LSTM (predict-the-next-note),
  which is simpler to train and a good starting point. A GAN (e.g. a
  MidiNet/MuseGAN-style architecture) can produce more diverse results but
  is considerably harder to get training stably — a good "v2" upgrade once
  the LSTM pipeline works end-to-end.
- **Multi-track**: this pipeline flattens everything to a single
  instrument line. For multi-instrument generation, keep each instrument's
  events as a separate channel/vocabulary and train per-instrument or with
  a multi-hot representation.

## File overview

| File | Purpose |
|---|---|
| `preprocess.py` | Parse MIDI → flat list of note/chord/rest event strings |
| `model.py` | Vocabulary building, sequence windowing, LSTM architecture |
| `train.py` | Train the LSTM on the preprocessed events, checkpoint weights |
| `generate.py` | Sample a new event sequence from trained weights → write `.mid` |
