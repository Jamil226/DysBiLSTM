"""Experiment-wide constants.

These values reproduce the protocol described in the manuscript
(Section III): 16 kHz mono audio, fixed 3 s window, 40 MFCCs,
stratified 80/10/10 split and a global seed of 42.
"""

SEED = 42

# ─── Dataset layout (Kaggle "Dysarthria and Non Dysarthria" release of TORGO) ──
DYSARTHRIC_DIRS = ("Female_dysarthria", "Male_Dysarthria")
NON_DYSARTHRIC_DIRS = ("Female_Non_Dysarthria", "Male_Non_Dysarthria")
CLASS_NAMES = ("Non-Dysarthric", "Dysarthric")  # label 0, label 1

# ─── Signal pre-processing ─────────────────────────────────────────────────────
SAMPLE_RATE = 16_000
DURATION = 3.0  # seconds; shorter clips are zero-padded, longer ones truncated
N_SAMPLES = int(SAMPLE_RATE * DURATION)

# ─── MFCC matrix (BiLSTM, LSTM, CNN, DT, RF, SVM, NB-flat) ─────────────────────
N_MFCC = 40
N_FFT = 1024
HOP_LENGTH = 512
N_FRAMES = 1 + N_SAMPLES // HOP_LENGTH  # 94 frames for a 3 s clip

# ─── Data partitioning ─────────────────────────────────────────────────────────
TEST_SIZE = 0.10
VAL_SIZE = 0.1111  # of the remaining 90 % -> 10 % of the total
