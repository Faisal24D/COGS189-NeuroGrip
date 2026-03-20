"""
COGS 189 - BCI Hand Demo
Single vs Double blink classification -> hand open/close

LIVE MODE:  CYTON_ENABLED = True  -> scans for board, reads live EEG indefinitely
SIM MODE:   CYTON_ENABLED = False -> replays one recorded session, stops when done
"""

import time
import os
import sys
import glob
import numpy as np
import threading
from queue import Queue, Empty
from collections import deque

import pygame
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
CYTON_ENABLED  = False       # set True when board is plugged in

DATA_DIR       = r"C:\Users\f1f99\OneDrive\Documents\Blink_Project"
FS             = 250
N_CH           = 8
EPOCH_T0       = -0.2
EPOCH_T1       =  2.5
BP_LOW         = 1.0
BP_HIGH        = 40.0
NOTCH_HZ       = 60.0
THRESHOLD      = 150         # uV artifact rejection (applied after filtering)
BAUD_RATE      = 115200
BLINK_COOLDOWN = 3.5         # seconds between classifications

REPLAY_FILE    = "eeg_20260306_163614.csv"

SESSIONS = [
    ("eeg_20260306_160721.csv",  "events_20260306_160721.csv"),
    ("eeg_20260306_161822.csv",  "events_20260306_161822.csv"),
    ("eeg_20260306_162212.csv",  "events_20260306_162212.csv"),
    ("eeg_20260306_162529.csv",  "events_20260306_162529.csv"),
    ("eeg_20260306_162907.csv",  "events_20260306_162907.csv"),
    ("eeg_20260306_163227.csv",  "events_20260306_163227.csv"),
    ("eeg_20260306_163614.csv",  "events_20260306_163614.csv"),
    ("eeg_20260306_163927.csv",  "events_20260306_163927.csv"),
    ("eeg_20260306_164236.csv",  "events_20260306_164236.csv"),
    ("eeg_20260306_165141.csv",  "events_20260306_165141.csv"),
]

# ─────────────────────────────────────────
# SIGNAL PROCESSING
# ─────────────────────────────────────────
def bandpass(data, lo, hi, fs, order=4):
    nyq = fs / 2.0
    b, a = butter(order, [lo/nyq, hi/nyq], btype='band')
    return filtfilt(b, a, data, axis=-1)

def notch_filter(data, freq, fs, Q=30):
    b, a = iirnotch(freq / (fs / 2), Q)
    return filtfilt(b, a, data, axis=-1)

def baseline_correct(epoch, times, t0=-0.2, t1=0.0):
    mask = (times >= t0) & (times <= t1)
    return epoch - epoch[:, mask].mean(axis=1, keepdims=True)

def extract_features(epoch, times):
    mask = (times >= 0.0) & (times <= 2.0)
    win  = epoch[:, mask]
    return np.concatenate([
        np.max(np.abs(win), axis=1),
        np.max(win, axis=1) - np.min(win, axis=1),
        np.mean(np.abs(win), axis=1),
        np.var(win, axis=1),
    ])

# ─────────────────────────────────────────
# CLASSIFIER TRAINING
# ─────────────────────────────────────────
def load_and_train(status_ref):
    all_epochs, all_labels = [], []
    times_ref = None

    for i, (eeg_f, ev_f) in enumerate(SESSIONS):
        status_ref[0] = f"Loading session {i+1}/{len(SESSIONS)}..."
        try:
            eeg_df = pd.read_csv(os.path.join(DATA_DIR, eeg_f))
            ev_df  = pd.read_csv(os.path.join(DATA_DIR, ev_f))

            ts     = eeg_df['timestamp_sec'].values
            eeg    = eeg_df[[f'eeg_ch{j+1}' for j in range(N_CH)]].values.T
            fs_est = round((len(ts)-1) / (ts[-1] - ts[0]))

            eeg = bandpass(eeg, BP_LOW, BP_HIGH, fs_est)
            eeg = notch_filter(eeg, NOTCH_HZ, fs_est)

            cues      = ev_df[ev_df['event'] == 'cue_on']
            epoch_len = int((EPOCH_T1 - EPOCH_T0) * fs_est)
            times     = np.linspace(EPOCH_T0, EPOCH_T1, epoch_len)
            if times_ref is None:
                times_ref = times

            for _, row in cues.iterrows():
                t_cue = row['t_wall_sec']
                label = 0 if row['trial_type'] == 'single' else 1
                idx0  = np.searchsorted(ts, t_cue + EPOCH_T0)
                idx1  = idx0 + epoch_len
                if idx1 > eeg.shape[1]:
                    continue
                epoch = eeg[:, idx0:idx1].copy()
                if np.all(epoch == 0):
                    continue
                epoch = baseline_correct(epoch, times)
                all_epochs.append(epoch)
                all_labels.append(label)
        except Exception as e:
            print(f"  Skip {eeg_f}: {e}")

    all_epochs = np.array(all_epochs)
    all_labels = np.array(all_labels)

    keep       = np.max(np.abs(all_epochs), axis=(1, 2)) < THRESHOLD
    all_epochs = all_epochs[keep]
    all_labels = all_labels[keep]

    X = np.array([extract_features(ep, times_ref) for ep in all_epochs])
    y = all_labels

    status_ref[0] = "Fitting classifier..."
    clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
    clf.fit(X, y)
    print(f"Classifier trained: {len(y)} epochs ({(y==0).sum()} single, {(y==1).sum()} double)")
    return clf, times_ref

# ─────────────────────────────────────────
# BOARD CONNECTION
# ─────────────────────────────────────────
def find_openbci_port():
    """Same method as blink_experiment.py — proven to work with Cyton."""
    import serial
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
    return openbci_port

def try_connect_board():
    """Try to connect to Cyton. Returns (board, ts_ch, eeg_chs) or None."""
    try:
        from brainflow.board_shim import BoardShim, BrainFlowInputParams
        BoardShim.enable_dev_board_logger()
        params = BrainFlowInputParams()
        params.serial_port = find_openbci_port()
        print(f"Cyton found on {params.serial_port}, connecting...")
        board = BoardShim(0, params)
        board.prepare_session()
        board.start_stream(45000)
        ts_ch   = BoardShim.get_timestamp_channel(0)
        eeg_chs = BoardShim.get_eeg_channels(0)
        print("Cyton connected and streaming.")
        return board, ts_ch, eeg_chs
    except Exception as e:
        print(f"Board connection failed: {e}")
        return None

# ─────────────────────────────────────────
# COLORS
# ─────────────────────────────────────────
BG         = (  8,  10,  20)
PANEL      = ( 14,  18,  32)
CYAN       = (  0, 210, 240)
CYAN_DIM   = (  0,  80, 110)
GREEN      = ( 50, 210, 110)
RED        = (240,  55,  75)
WHITE      = (215, 225, 245)
GRAY       = ( 75,  85, 110)
HAND_FILL  = ( 28,  48,  88)
HAND_EDGE  = ( 55,  95, 165)
HAND_JOINT = (  0, 155, 210)

def lerp_col(c1, c2, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))

# ─────────────────────────────────────────
# HAND DRAWING
# ─────────────────────────────────────────
def draw_finger(surf, bx, by, width, segments, max_angles, open_frac):
    x, y  = bx, by
    angle = 0.0
    pts   = [(x, y)]

    for seg_len, max_ang in zip(segments, max_angles):
        angle += max_ang * (1.0 - open_frac)
        rad    = np.radians(angle)
        x     += np.sin(rad) * seg_len
        y     -= np.cos(rad) * seg_len
        pts.append((x, y))

    for i in range(len(pts) - 1):
        p1 = (int(pts[i][0]),   int(pts[i][1]))
        p2 = (int(pts[i+1][0]), int(pts[i+1][1]))
        pygame.draw.line(surf, HAND_EDGE, p1, p2, width + 4)
        pygame.draw.line(surf, HAND_FILL, p1, p2, width)

    for (jx, jy) in pts:
        pygame.draw.circle(surf, HAND_JOINT, (int(jx), int(jy)), max(3, width // 2))
        pygame.draw.circle(surf, HAND_EDGE,  (int(jx), int(jy)), max(3, width // 2), 1)

def draw_hand(surf, cx, cy, open_frac):
    pw, ph = 110, 100
    px, py = cx - pw // 2, cy - ph // 4

    pygame.draw.rect(surf, HAND_EDGE, (px-2, py-2, pw+4, ph+4), border_radius=16)
    pygame.draw.rect(surf, HAND_FILL, (px,   py,   pw,   ph),   border_radius=14)

    fingers = [
        (-33, 0,  13, [32, 26, 20], [0, 72, 60]),
        ( -9, -5, 13, [36, 28, 22], [0, 72, 60]),
        ( 15, 0,  12, [33, 26, 20], [0, 72, 60]),
        ( 37, 8,  10, [26, 20, 16], [0, 65, 55]),
    ]
    for ox, oy, w, segs, angs in fingers:
        draw_finger(surf, cx + ox, py + oy, w, segs, angs, open_frac)

    px0, py0 = px + 6, py + 40
    curl     = 68 * (1.0 - open_frac)
    ang0     = np.radians(-150 + curl * 0.7)
    tx1      = px0 + np.cos(ang0) * 32
    ty1      = py0 + np.sin(ang0) * 32
    ang1     = ang0 + np.radians(curl * 0.5)
    tx2      = tx1 + np.cos(ang1) * 26
    ty2      = ty1 + np.sin(ang1) * 26

    for p1, p2, w in [
        ((px0, py0), (int(tx1), int(ty1)), 15),
        ((int(tx1), int(ty1)), (int(tx2), int(ty2)), 13),
    ]:
        pygame.draw.line(surf, HAND_EDGE, p1, p2, w + 4)
        pygame.draw.line(surf, HAND_FILL, p1, p2, w)

    for (jx, jy) in [(tx1, ty1), (tx2, ty2)]:
        pygame.draw.circle(surf, HAND_JOINT, (int(jx), int(jy)), 7)
        pygame.draw.circle(surf, HAND_EDGE,  (int(jx), int(jy)), 7, 1)

def draw_panel(surf, x, y, w, h, title, font):
    pygame.draw.rect(surf, PANEL,    (x, y, w, h), border_radius=6)
    pygame.draw.rect(surf, CYAN_DIM, (x, y, w, h), 1, border_radius=6)
    surf.blit(font.render(title, True, CYAN), (x + 10, y + 10))
    pygame.draw.line(surf, CYAN_DIM, (x+10, y+28), (x+w-10, y+28), 1)

# ─────────────────────────────────────────
# LOADING SCREEN
# ─────────────────────────────────────────
def show_loading(screen, W, H, fonts, status_ref, tick):
    fs_title, fs_large, fs_med, fs_small = fonts
    screen.fill(BG)
    pygame.draw.rect(screen, PANEL, (0, 0, W, 50))
    pygame.draw.line(screen, CYAN_DIM, (0, 50), (W, 50), 1)
    screen.blit(fs_title.render("COGS 189  //  BCI HAND DEMO", True, CYAN), (18, 15))

    for i in range(8):
        ang = np.radians(i * 45 - tick * 5)
        sx  = int(W//2 + np.cos(ang) * 40)
        sy  = int(H//2 - 30 + np.sin(ang) * 40)
        col = lerp_col(GRAY, CYAN, (i/8 + tick*0.025) % 1.0)
        pygame.draw.circle(screen, col, (sx, sy), 5)

    t1 = fs_large.render("INITIALIZING...", True, CYAN)
    screen.blit(t1, (W//2 - t1.get_width()//2, H//2 + 30))
    t2 = fs_med.render(status_ref[0], True, WHITE)
    screen.blit(t2, (W//2 - t2.get_width()//2, H//2 + 85))
    pygame.display.flip()

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main():
    pygame.init()
    W, H   = 980, 660
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("COGS 189 - BCI Hand Demo")
    clock  = pygame.time.Clock()

    fs_title = pygame.font.SysFont("Consolas", 20, bold=True)
    fs_large = pygame.font.SysFont("Consolas", 44, bold=True)
    fs_med   = pygame.font.SysFont("Consolas", 19)
    fs_small = pygame.font.SysFont("Consolas", 14)
    fonts    = (fs_title, fs_large, fs_med, fs_small)

    # ── Step 1: Train classifier in background ──
    status_ref   = ["Starting..."]
    train_result = [None]
    train_error  = [None]

    def train_bg():
        try:
            clf, times_ref  = load_and_train(status_ref)
            train_result[0] = (clf, times_ref)
            status_ref[0]   = "Classifier ready."
        except Exception as e:
            train_error[0] = str(e)
            status_ref[0]  = f"ERROR: {e}"

    train_thread = threading.Thread(target=train_bg, daemon=True)
    train_thread.start()

    tick = 0
    while train_thread.is_alive():
        clock.tick(60)
        tick += 1
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (
                    ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                pygame.quit()
                return
        show_loading(screen, W, H, fonts, status_ref, tick)

    if train_error[0]:
        print(f"Training failed: {train_error[0]}")
        pygame.quit()
        return

    clf, times_ref = train_result[0]
    epoch_len      = len(times_ref)

    # Hold "Ready" for 1 second
    t0 = time.time()
    while time.time() - t0 < 1.0:
        clock.tick(60)
        tick += 1
        show_loading(screen, W, H, fonts, status_ref, tick)

    # ── Step 2: Connect to board (only if CYTON_ENABLED) ──
    board_result = None

    if CYTON_ENABLED:
        status_ref[0] = "Scanning for Cyton board..."
        scan_done     = [False]
        scan_result   = [None]

        def scan_bg():
            scan_result[0] = try_connect_board()
            scan_done[0]   = True
            if scan_result[0]:
                status_ref[0] = "Board found!"
            else:
                status_ref[0] = "No board found — SIM MODE"

        scan_thread = threading.Thread(target=scan_bg, daemon=True)
        scan_thread.start()

        while not scan_done[0]:
            clock.tick(60)
            tick += 1
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT or (
                        ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                    pygame.quit()
                    return
            show_loading(screen, W, H, fonts, status_ref, tick)

        board_result = scan_result[0]

        # Hold result message for 1 second
        t0 = time.time()
        while time.time() - t0 < 1.0:
            clock.tick(60)
            tick += 1
            show_loading(screen, W, H, fonts, status_ref, tick)
    else:
        status_ref[0] = "SIM MODE"

    if board_result is not None:
        board, ts_ch, eeg_chs = board_result
        live_mode = True
        print("LIVE MODE: Cyton connected.")
    else:
        board     = None
        live_mode = False
        print("SIM MODE: Replaying recorded session.")

    # ── Shared buffers ──
    eeg_buf  = deque(maxlen=FS * 12)
    ts_buf   = deque(maxlen=FS * 12)
    buf_lock = threading.Lock()
    result_q = Queue()
    stop_ev  = threading.Event()

    sim_done  = [False]
    sim_total = [0]
    sim_fed   = [0]

    # ── Reader thread ──
    def live_reader():
        """Same pattern as cyton_reader in blink_experiment.py."""
        while not stop_ev.is_set():
            try:
                data = board.get_board_data()
                if data.size != 0 and data.shape[1] > 0:
                    timestamps = data[ts_ch, :].astype(float)
                    eeg        = data[eeg_chs, :].astype(float)
                    with buf_lock:
                        for i in range(eeg.shape[1]):
                            ts_buf.append(timestamps[i])
                            eeg_buf.append(eeg[:, i])
            except Exception as e:
                print(f"Live reader error: {e}")
            time.sleep(0.05)

    def sim_reader():
        """Feed recorded CSV into buffer at real-time speed. Stops when file ends."""
        print(f"SIM: Loading {REPLAY_FILE}...")
        try:
            df           = pd.read_csv(os.path.join(DATA_DIR, REPLAY_FILE))
            eeg_data     = df[[f'eeg_ch{i+1}' for i in range(N_CH)]].values
            sim_total[0] = len(eeg_data)
            print(f"SIM: {sim_total[0]} samples loaded. Feeding at {FS} Hz.")
        except Exception as e:
            print(f"SIM: Could not load file: {e}")
            sim_done[0] = True
            return

        batch = 25
        idx   = 0
        while not stop_ev.is_set() and idx < len(eeg_data):
            end = min(idx + batch, len(eeg_data))
            with buf_lock:
                for j in range(idx, end):
                    ts_buf.append(time.time())
                    eeg_buf.append(eeg_data[j])
            sim_fed[0] = end
            idx        = end
            time.sleep(batch / FS)

        print("SIM: Replay complete.")
        sim_done[0] = True

    # ── Classifier thread ──
    last_classify_time = [0.0]

    def classifier_thread():
        while not stop_ev.is_set():
            now = time.time()
            if now - last_classify_time[0] < BLINK_COOLDOWN:
                time.sleep(0.05)
                continue

            # Snapshot buffer without holding lock during processing
            with buf_lock:
                buf_len  = len(ts_buf)
                eeg_snap = np.array(eeg_buf).T.copy() if buf_len >= epoch_len else None

            if eeg_snap is None:
                time.sleep(0.1)
                continue

            ep = eeg_snap[:, -epoch_len:]

            try:
                ep      = bandpass(ep, BP_LOW, BP_HIGH, FS)
                ep      = notch_filter(ep, NOTCH_HZ, FS)
                ep      = baseline_correct(ep, times_ref)
                max_amp = np.max(np.abs(ep))

                if max_amp > THRESHOLD:
                    print(f"Epoch rejected: {max_amp:.1f} uV")
                    time.sleep(0.1)
                    continue

                feats      = extract_features(ep, times_ref).reshape(1, -1)
                pred       = clf.predict(feats)[0]
                prob       = clf.predict_proba(feats)[0]
                confidence = max(prob)
                print(f"Prediction: {'SINGLE' if pred==0 else 'DOUBLE'}  conf={confidence*100:.1f}%")
                result_q.put((pred, confidence, now))
                last_classify_time[0] = now

            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Classifier error: {e}")

            time.sleep(0.05)

    reader = threading.Thread(
        target=live_reader if live_mode else sim_reader, daemon=True)
    reader.start()

    clf_t = threading.Thread(target=classifier_thread, daemon=True)
    clf_t.start()

    # ── Animation state ──
    open_frac      = 1.0
    target_frac    = 1.0
    ANIM_SPEED     = 0.035
    last_pred      = None
    last_conf      = 0.0
    last_pred_time = 0.0
    history        = []
    flash_timer    = 0.0
    flash_col      = GREEN
    paused         = False

    # ── Main loop ──
    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        flash_timer = max(0.0, flash_timer - dt)

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_SPACE:
                    paused = not paused

        # Auto-stop when sim replay finishes and queue is drained
        if not live_mode and sim_done[0] and result_q.empty():
            time.sleep(2.0)
            running = False

        if not paused:
            try:
                pred, conf, t  = result_q.get_nowait()
                last_pred      = pred
                last_conf      = conf
                last_pred_time = t
                target_frac    = 1.0 if pred == 0 else 0.0
                flash_col      = GREEN if pred == 0 else RED
                flash_timer    = 0.5
                lstr = "SINGLE BLINK -> OPEN" if pred == 0 else "DOUBLE BLINK -> CLOSE"
                history.append((lstr, conf, time.time()))
                if len(history) > 6:
                    history.pop(0)
            except Empty:
                pass

        if open_frac < target_frac:
            open_frac = min(open_frac + ANIM_SPEED, target_frac)
        elif open_frac > target_frac:
            open_frac = max(open_frac - ANIM_SPEED, target_frac)

        # ── DRAW ──
        screen.fill(BG)

        for gx in range(0, W, 50):
            pygame.draw.line(screen, (12, 18, 34), (gx, 0), (gx, H), 1)
        for gy in range(0, H, 50):
            pygame.draw.line(screen, (12, 18, 34), (0, gy), (W, gy), 1)

        if flash_timer > 0:
            s = pygame.Surface((W, H), pygame.SRCALPHA)
            s.fill(flash_col + (int(25 * flash_timer / 0.5),))
            screen.blit(s, (0, 0))

        # Top bar
        pygame.draw.rect(screen, PANEL, (0, 0, W, 50))
        pygame.draw.line(screen, CYAN_DIM, (0, 50), (W, 50), 1)
        screen.blit(fs_title.render("COGS 189  //  BCI HAND DEMO", True, CYAN), (18, 15))

        mode_str = "LIVE EEG" if live_mode else "SIM MODE"
        mode_col = GREEN      if live_mode else RED
        ms = fs_small.render(f"* {mode_str}", True, mode_col)
        mx = W - ms.get_width() - 18
        pygame.draw.rect(screen, PANEL,    (mx-8, 13, ms.get_width()+16, 24), border_radius=4)
        pygame.draw.rect(screen, mode_col, (mx-8, 13, ms.get_width()+16, 24), 1, border_radius=4)
        screen.blit(ms, (mx, 18))

        # Left panel: Detection log
        lx, ly, lw, lh = 16, 62, 255, 310
        draw_panel(screen, lx, ly, lw, lh, "// DETECTION LOG", fs_small)
        for i, (lstr, conf_val, t_val) in enumerate(reversed(history)):
            age  = time.time() - t_val
            fade = max(0.2, 1.0 - age / 20.0)
            col  = lerp_col(GREEN if "OPEN" in lstr else RED, GRAY, 1.0 - fade)
            icon = "^ " if "OPEN" in lstr else "v "
            screen.blit(fs_small.render(icon + lstr, True, col),
                        (lx+10, ly+38 + i*40))
            screen.blit(fs_small.render(f"  conf: {conf_val*100:.0f}%", True,
                        lerp_col(GRAY, BG, 1.0-fade)),
                        (lx+10, ly+54 + i*40))

        # Left panel 2: Status
        sx, sy, sw, sh = 16, 385, 255, 185
        draw_panel(screen, sx, sy, sw, sh, "// STATUS", fs_small)

        with buf_lock:
            buf_now = len(ts_buf)
        buf_frac  = min(1.0, buf_now / epoch_len)
        buf_ready = buf_frac >= 1.0
        buf_col   = GREEN if buf_ready else CYAN
        screen.blit(fs_small.render("BUFFER", True, GRAY), (sx+10, sy+38))
        pygame.draw.rect(screen, (20,30,50), (sx+85, sy+40, sw-100, 12), border_radius=4)
        pygame.draw.rect(screen, buf_col,
                         (sx+85, sy+40, int((sw-100)*buf_frac), 12), border_radius=4)
        screen.blit(fs_small.render(
            "READY" if buf_ready else f"{buf_now}/{epoch_len}",
            True, buf_col), (sx+85, sy+56))

        cd_elapsed = time.time() - last_classify_time[0]
        cd_frac    = min(1.0, cd_elapsed / BLINK_COOLDOWN)
        cd_ready   = cd_frac >= 1.0
        cd_col     = GREEN if cd_ready else CYAN
        screen.blit(fs_small.render("COOLDOWN", True, GRAY), (sx+10, sy+80))
        pygame.draw.rect(screen, (20,30,50), (sx+85, sy+82, sw-100, 12), border_radius=4)
        pygame.draw.rect(screen, cd_col,
                         (sx+85, sy+82, int((sw-100)*cd_frac), 12), border_radius=4)
        screen.blit(fs_small.render(
            "READY" if cd_ready else f"{BLINK_COOLDOWN*(1-cd_frac):.1f}s",
            True, cd_col), (sx+85, sy+98))

        if not live_mode:
            sim_frac = min(1.0, sim_fed[0] / max(1, sim_total[0]))
            sim_col  = GREEN if sim_done[0] else CYAN
            screen.blit(fs_small.render("REPLAY", True, GRAY), (sx+10, sy+122))
            pygame.draw.rect(screen, (20,30,50), (sx+85, sy+124, sw-100, 12), border_radius=4)
            pygame.draw.rect(screen, sim_col,
                             (sx+85, sy+124, int((sw-100)*sim_frac), 12), border_radius=4)
            screen.blit(fs_small.render(
                "DONE" if sim_done[0] else f"{sim_frac*100:.0f}%",
                True, sim_col), (sx+85, sy+140))

        # Center: Hand
        hcx, hcy  = W//2 + 20, H//2 + 35
        state_col = GREEN if open_frac > 0.5 else RED
        rr = 125
        rs = pygame.Surface((rr*2+20, rr*2+20), pygame.SRCALPHA)
        pygame.draw.circle(rs, state_col+(15,), (rr+10, rr+10), rr)
        pygame.draw.circle(rs, state_col+(40,), (rr+10, rr+10), rr, 2)
        screen.blit(rs, (hcx-rr-10, hcy-rr-10))

        draw_hand(screen, hcx, hcy, open_frac)

        ss = fs_large.render("OPEN" if open_frac > 0.5 else "CLOSED", True, state_col)
        screen.blit(ss, (hcx - ss.get_width()//2, hcy + 158))

        if last_pred is not None:
            elapsed = time.time() - last_pred_time
            fade    = max(0.0, 1.0 - elapsed / 4.0)
            col     = lerp_col(GREEN if last_pred==0 else RED, GRAY, 1.0-fade)
            ds = fs_med.render(
                "SINGLE BLINK DETECTED" if last_pred==0 else "DOUBLE BLINK DETECTED",
                True, col)
            screen.blit(ds, (hcx - ds.get_width()//2, hcy + 208))

        # Right panel: Classifier info
        rx, ry, rw, rh = W-272, 62, 256, 195
        draw_panel(screen, rx, ry, rw, rh, "// CLASSIFIER", fs_small)
        for i, (k, v) in enumerate([
            ("MODEL",    "LDA + StandardScaler"),
            ("ACCURACY", "77.0% +/- 7.3%"),
            ("AUC",      "0.81 +/- 0.05"),
            ("EPOCHS",   "176  (89S / 87D)"),
            ("FEATURES", "32  (4 x 8ch)"),
            ("COOLDOWN", f"{BLINK_COOLDOWN}s"),
        ]):
            screen.blit(fs_small.render(k, True, GRAY),  (rx+10,  ry+38+i*25))
            screen.blit(fs_small.render(v, True, WHITE), (rx+100, ry+38+i*25))

        # Right panel 2: Confidence
        cx2, cy2, cw2, ch2 = W-272, 270, 256, 110
        draw_panel(screen, cx2, cy2, cw2, ch2, "// CONFIDENCE", fs_small)
        conf_val = last_conf if last_pred is not None else 0.0
        conf_col = lerp_col(RED, GREEN, conf_val)
        pygame.draw.rect(screen, (20,30,50),
                         (cx2+10, cy2+42, cw2-20, 16), border_radius=4)
        pygame.draw.rect(screen, conf_col,
                         (cx2+10, cy2+42, int((cw2-20)*conf_val), 16), border_radius=4)
        ct = fs_med.render(f"{conf_val*100:.1f}%", True, WHITE)
        screen.blit(ct, (cx2+cw2//2 - ct.get_width()//2, cy2+66))

        hint = fs_small.render("ESC: quit    SPACE: pause", True, GRAY)
        screen.blit(hint, (W//2 - hint.get_width()//2, H-20))

        if paused:
            ov = pygame.Surface((W, H), pygame.SRCALPHA)
            ov.fill((0, 0, 0, 150))
            screen.blit(ov, (0, 0))
            ps = fs_large.render("PAUSED", True, CYAN)
            screen.blit(ps, (W//2 - ps.get_width()//2, H//2 - 25))
            ps2 = fs_med.render("SPACE to resume", True, GRAY)
            screen.blit(ps2, (W//2 - ps2.get_width()//2, H//2 + 30))

        pygame.display.flip()

    # ── Cleanup ──
    stop_ev.set()
    if board is not None:
        try:
            board.stop_stream()
            board.release_session()
        except Exception:
            pass
    pygame.quit()
    print("Demo closed.")


if __name__ == "__main__":
    main()