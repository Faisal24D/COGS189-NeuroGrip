import time
import csv
import random
import sys
import glob
import serial
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

STREAM_PULL_SEC = 0.05
BAUD_RATE = 115200
CYTON_ENABLED = False  # set False to run paradigm only (no headset)

# --------------------------
# Cyton utilities
# --------------------------
def find_openbci_port():
    """
    Simon's method: actively handshakes with each serial port by sending 'v'
    and checking for 'OpenBCI' in the response. Works on Windows, Linux, Mac.
    """
    if sys.platform.startswith('win'):
        ports = ['COM%s' % (i + 1) for i in range(256)]
    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
        ports = glob.glob('/dev/ttyUSB*')
    elif sys.platform.startswith('darwin'):
        ports = glob.glob('/dev/cu.usbserial*')
    else:
        raise EnvironmentError('Error finding ports on your operating system')

    openbci_port = ''
    for port in ports:
        try:
            s = serial.Serial(port=port, baudrate=BAUD_RATE, timeout=None)
            s.write(b'v')
            line = ''
            time.sleep(2)
            if s.inWaiting():
                line = ''
                c = ''
                while '$$$' not in line:
                    c = s.read().decode('utf-8', errors='replace')
                    line += c
                if 'OpenBCI' in line:
                    openbci_port = port
            s.close()
        except (OSError, serial.SerialException):
            pass

    if openbci_port == '':
        raise OSError('Cannot find OpenBCI port.')
    else:
        return openbci_port

def start_cyton():
    """
    Connect to Cyton board. Returns (board, board_id) or raises on failure.
    """
    BoardShim.enable_dev_board_logger()

    params = BrainFlowInputParams()
    params.serial_port = find_openbci_port()
    print(f"Attempting Cyton connection on port: '{params.serial_port}'")

    board_id = 0  # Cyton board ID
    board = BoardShim(board_id, params)
    board.prepare_session()
    board.start_stream(45000)
    print("Cyton connected and streaming.")
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

def cyton_reader(board, board_id, stop_event, q):
    """
    Background thread: continuously pulls buffered samples and pushes to queue.
    Each queue item: (timestamps, eeg_matrix)
      - timestamps: shape (n_samples,)
      - eeg_matrix: shape (n_channels, n_samples)
    Exits safely if board is None.
    """
    if board is None:
        print("cyton_reader: board is None, thread exiting.")
        return

    ts_ch = board.get_timestamp_channel(board_id)
    eeg_chs = board.get_eeg_channels(board_id)

    while not stop_event.is_set():
        try:
            data = board.get_board_data()
            if data.size != 0 and data.shape[1] > 0:
                timestamps = data[ts_ch, :].astype(float)
                eeg = data[eeg_chs, :].astype(float)
                q.put((timestamps, eeg))
        except Exception as e:
            print(f"cyton_reader error: {e}")
        time.sleep(STREAM_PULL_SEC)

# --------------------------
# Main experiment
# --------------------------
def main():
    print("Script started")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eeg_path = f"eeg_{stamp}.csv"
    events_path = f"events_{stamp}.csv"

    # Create randomized trial list
    trials = (["single"] * N_SINGLE) + (["double"] * N_DOUBLE)
    random.shuffle(trials)

    # PsychoPy window
    print("Creating window...")
    WIN = visual.Window(size=(1000, 650), color="black", units="norm")
    textStim = visual.TextStim(WIN, text="", color="white", height=0.12)

    # Corner indicator dot: red = not recording, green = recording
    indicator = visual.Circle(
        WIN, radius=0.03, pos=(0.9, -0.85),
        fillColor="red", lineColor="red"
    )

    def show(text):
        textStim.text = text
        textStim.draw()
        indicator.draw()
        WIN.flip()

    # --- Ready screen ---
    show("READY\n\nPress SPACE to start recording")
    event.waitKeys(keyList=["space"])

    # --- Connect to Cyton ---
    board = None
    board_id = None
    n_eeg_ch = 8  # default fallback for CSV header

    if CYTON_ENABLED:
        try:
            board, board_id = start_cyton()
            n_eeg_ch = len(board.get_eeg_channels(board_id))
            print(f"EEG channels detected: {n_eeg_ch}")
        except Exception as e:
            print(f"ERROR: Could not connect to Cyton: {e}")
            print("Falling back to SIM MODE (no EEG data will be saved).")
            board = None
            board_id = None
    else:
        print("SIM MODE: Cyton disabled. Running paradigm only.")

    # --- Turn dot green and confirm recording ---
    indicator.fillColor = "green"
    indicator.lineColor = "green"
    status = "RECORDING" if (board is not None) else "SIM MODE (no board)"
    show(f"{status}\n\nPress SPACE to begin SYNC")
    event.waitKeys(keyList=["space"])

    # --- Start background reader thread ---
    stop_event = Event()
    q = Queue()
    t = Thread(target=cyton_reader, args=(board, board_id, stop_event, q), daemon=True)
    t.start()

    # --- Open CSV files and run experiment ---
    with open(eeg_path, "w", newline="") as eeg_f, open(events_path, "w", newline="") as ev_f:
        eeg_writer = csv.writer(eeg_f)
        ev_writer = csv.writer(ev_f)

        # Write headers (n_eeg_ch determined safely above)
        eeg_writer.writerow(["timestamp_sec"] + [f"eeg_ch{i+1}" for i in range(n_eeg_ch)])
        ev_writer.writerow(["event", "trial_index", "trial_type", "t_wall_sec"])

        exp_start_wall = time.time()
        ev_writer.writerow(["experiment_start", -1, "", exp_start_wall])

        def drain_eeg():
            """Flush queue to CSV. No-op if board not connected."""
            if board is None:
                return 0
            wrote = 0
            while True:
                try:
                    timestamps, eeg = q.get_nowait()
                except Empty:
                    break
                for k in range(eeg.shape[1]):
                    row = [timestamps[k]] + [float(eeg[ch, k]) for ch in range(eeg.shape[0])]
                    eeg_writer.writerow(row)
                    wrote += 1
            return wrote

        # --- SYNC BLINK BLOCK ---
        show("SYNC:\nDo 3 exaggerated blinks\n(Wait for BLINK NOW)")
        core.wait(1.5)

        for i in range(SYNC_BLINKS):
            show("BLINK NOW (SYNC)")
            ev_writer.writerow(["sync_blink", -1, "sync", time.time()])
            core.wait(SYNC_GAP_SEC)
            drain_eeg()

        # --- MAIN TRIALS ---
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

            if "escape" in event.getKeys():
                ev_writer.writerow(["aborted_escape", -1, "", time.time()])
                break

        ev_writer.writerow(["experiment_end", -1, "", time.time()])
        drain_eeg()

    # --- Cleanup ---
    stop_event.set()
    core.wait(0.2)
    if board is not None:
        stop_cyton(board)

    indicator.fillColor = "red"
    indicator.lineColor = "red"
    show(f"DONE\nSaved:\n{eeg_path}\n{events_path}\n\nPress SPACE to exit")
    event.waitKeys(keyList=["space"])

    WIN.close()
    core.quit()

if __name__ == "__main__":
    main()