import time
import csv
import random
from datetime import datetime
from threading import Thread, Event
from queue import Queue, Empty

import numpy as np
from psychopy import visual, event, core

from brainflow.board_shim import BoardShim, BrainFlowInputParams

# --------------------------
# Settings
# --------------------------
N_SINGLE = 10
N_DOUBLE = 10

FIX_SEC = 1.5
CUE_SEC = 1.5
REST_SEC = 1.5

SYNC_BLINKS = 3
SYNC_GAP_SEC = 2.0

STREAM_PULL_SEC = 0.05  # how often we pull from Cyton buffer in background
CYTON_ENABLED = True  # set True in the lab
# --------------------------
# Cyton utilities
# --------------------------
def find_openbci_port():
    """
    Simplest version: BrainFlow can often auto-find if serial_port is empty.
    
    """
    return ""  # try auto first

def start_cyton():
    BoardShim.enable_dev_board_logger()  # optional logging

    params = BrainFlowInputParams()
    params.serial_port = find_openbci_port()

    board_id = 0 #CYTON BOARD ID
    board = BoardShim(board_id, params)

    board.prepare_session()
    board.start_stream(45000)
    return board, board_id

def stop_cyton(board):
    try:
        board.stop_stream()
    except Exception:
        pass
    try:
        board.release_session()
    except Exception:
        pass

def cyton_reader(board, board_id, stop_event, q: Queue):
    """
    Continuously pulls any available samples and pushes to a queue.
    Each queue item: (timestamps, eeg_matrix)
    - timestamps: shape (n_samples,)
    - eeg_matrix: shape (n_channels, n_samples)
    """
    ts_ch = board.get_timestamp_channel(board_id)
    eeg_chs = board.get_eeg_channels(board_id)

    while not stop_event.is_set():
        data = board.get_board_data()  # returns ALL currently buffered samples
        if data.size != 0 and data.shape[1] > 0:
            timestamps = data[ts_ch, :].astype(float)
            eeg = data[eeg_chs, :].astype(float)
            q.put((timestamps, eeg))
        time.sleep(STREAM_PULL_SEC)

# --------------------------
# Main experiment
# --------------------------
def main():
    # Timestamped output 
    print("Script started")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eeg_path = f"eeg_{stamp}.csv"
    events_path = f"events_{stamp}.csv" 

    # Create trial list
    trials = (["single"] * N_SINGLE) + (["double"] * N_DOUBLE)
    random.shuffle(trials)

    # PsychoPy window + 
    print("Creating window...")
    WIN = visual.Window(size=(1000, 650), color="black", units="norm")
    textStim = visual.TextStim(WIN, text="", color="white", height=0.12)

    # Corner indicator dot (top-right)
    indicator = visual.Circle(
        WIN, radius=0.03, pos=(0.9, -0.85),
        fillColor="red", lineColor="red"
    )

    # A helper: draw text + indicator and flip (this is the "highlighted benefit")
    def show(text):
        textStim.text = text
        textStim.draw()
        indicator.draw()
        WIN.flip()

    # Show ready screen
    show("READY\n\nPress SPACE to start recording")
    event.waitKeys(keyList=["space"])

    # Start Cyton
    board = None
    board_id = None
    if CYTON_ENABLED:
        board, board_id = start_cyton()
    else:
        print("SIM MODE: Cyton disabled (no headset). Running paradigm only.")

    # Turn dot green (recording active) and show it
    indicator.fillColor = "green"
    indicator.lineColor = "green"
    show("RECORDING...\n\nPress SPACE to begin SYNC")
    event.waitKeys(keyList=["space"])

    # Start background streaming thread
    stop_event = Event()
    q = Queue()
    t = Thread(target=cyton_reader, args=(board, board_id, stop_event, q), daemon=True)
    t.start()

    # Open CSV writers
    with open(eeg_path, "w", newline="") as eeg_f, open(events_path, "w", newline="") as ev_f:
        eeg_writer = csv.writer(eeg_f)
        ev_writer = csv.writer(ev_f)

        eeg_writer.writerow(["timestamp_sec"] + [f"eeg_ch{i+1}" for i in range(len(board.get_eeg_channels(board_id)))])
        ev_writer.writerow(["event", "trial_index", "trial_type", "t_wall_sec"])

        exp_start_wall = time.time()
        ev_writer.writerow(["experiment_start", -1, "", exp_start_wall])

        # Helper to drain queue into EEG csv
        def drain_eeg():
            if not CYTON_ENABLED:
                return 0
            wrote = 0
            while True:
                try:
                    timestamps, eeg = q.get_nowait()
                except Empty:
                    break

                # write sample-by-sample rows
                # eeg shape: (n_ch, n_samp)
                for k in range(eeg.shape[1]):
                    row = [timestamps[k]] + [float(eeg[ch, k]) for ch in range(eeg.shape[0])]
                    eeg_writer.writerow(row)
                    wrote += 1
            return wrote

        # -------------------
        # SYNC BLINK BLOCK
        # -------------------
        show("SYNC:\nDo 3 exaggerated blinks\n(Wait for BLINK NOW)")
        core.wait(1.5)

        for i in range(SYNC_BLINKS):
            show("BLINK NOW (SYNC)")
            ev_writer.writerow(["sync_blink", -1, "sync", time.time()])
            core.wait(SYNC_GAP_SEC)
            drain_eeg()

        # -------------------
        # MAIN TRIALS
        # -------------------
        for idx, trial_type in enumerate(trials, start=1):
            show("+")
            core.wait(FIX_SEC)
            drain_eeg()

            cue_text = "SINGLE BLINK" if trial_type == "single" else "DOUBLE BLINK"
            show(cue_text)
            ev_writer.writerow(["cue_on", idx, trial_type, time.time()])
            core.wait(CUE_SEC)
            drain_eeg()

            show("rest...")
            core.wait(REST_SEC)
            drain_eeg()

            # allow abort
            if "escape" in event.getKeys():
                ev_writer.writerow(["aborted_escape", -1, "", time.time()])
                break

        ev_writer.writerow(["experiment_end", -1, "", time.time()])
        drain_eeg()

    # Stop streaming
    stop_event.set()
    core.wait(0.2)
    if CYTON_ENABLED:
        stop_cyton(board)

    # Dot red (not recording)
    indicator.fillColor = "red"
    indicator.lineColor = "red"
    show(f"DONE\nSaved:\n{eeg_path}\n{events_path}\n\nPress SPACE to exit")
    event.waitKeys(keyList=["space"])

    WIN.close()
    core.quit()

if __name__ == "__main__":
    main()