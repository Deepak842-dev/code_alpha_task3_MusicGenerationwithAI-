"""
train.py
--------
Loads the pickled event list produced by preprocess.py, builds training
sequences, and trains the LSTM model, checkpointing weights after epochs
that improve on the best-seen loss.

Usage:
    python train.py --notes notes.pkl --epochs 100 --batch_size 64
"""
import argparse
import pickle
import os

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from model import build_vocab, prepare_sequences, build_model, SEQUENCE_LENGTH


def main():
    parser = argparse.ArgumentParser(description="Train the LSTM music generation model.")
    parser.add_argument("--notes", default="notes.pkl", help="Pickled event list from preprocess.py")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--sequence_length", type=int, default=SEQUENCE_LENGTH)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--vocab_out", default="vocab.pkl", help="Where to save the note<->int mappings")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    with open(args.notes, "rb") as f:
        events = pickle.load(f)
    print(f"Loaded {len(events)} events from {args.notes}")

    note_to_int, int_to_note = build_vocab(events)
    n_vocab = len(note_to_int)
    print(f"Vocabulary size: {n_vocab} unique note/chord/rest events")

    with open(args.vocab_out, "wb") as f:
        pickle.dump((note_to_int, int_to_note), f)

    X, y = prepare_sequences(events, note_to_int, args.sequence_length)
    print(f"Training samples: {X.shape[0]}")

    model = build_model(args.sequence_length, n_vocab)
    model.summary()

    checkpoint_path = os.path.join(args.checkpoint_dir, "weights-{epoch:02d}-{loss:.4f}.weights.h5")
    callbacks = [
        ModelCheckpoint(checkpoint_path, monitor="loss", save_best_only=True, save_weights_only=True, verbose=1),
        EarlyStopping(monitor="loss", patience=10, restore_best_weights=True),
    ]

    model.fit(X, y, epochs=args.epochs, batch_size=args.batch_size, callbacks=callbacks)

    model.save_weights(os.path.join(args.checkpoint_dir, "final.weights.h5"))
    print("Training complete. Final weights saved.")


if __name__ == "__main__":
    main()
