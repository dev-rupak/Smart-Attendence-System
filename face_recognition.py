"""
face_recognition.py
────────────────────
Face recognition module.
Gives the user up to 3 blink-trials before returning "DENIED".

API:
    recognize_user(user_id, status_cb=None, cancel_event=None) -> str
    Returns: "GRANTED" | "DENIED" | "ERROR"
"""

import cv2
import mediapipe as mp
import time, os, json, math, threading
import numpy as np
import warnings
from collections import deque
from deepface import DeepFace

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────
USER_DB           = "User_Database"
GLOBAL_THRESHOLD  = 0.55   # tightened: must be below fraud-check threshold (0.55)
CAM_W, CAM_H      = 320, 240
ANTISPOOF_THRESH  = 10.0
BLINK_EAR         = 0.26
BLINK_TIMEOUT_S   = 25.0   # per trial
MAX_FACE_TRIALS   = 3


# ── Camera ──────────────────────────────────────────────────────

def _find_cam():
    try:
        import subprocess
        out = subprocess.check_output(
            ['v4l2-ctl', '--list-devices'], text=True, stderr=subprocess.DEVNULL)
        for block in out.split('\n\n'):
            if 'USB' in block or 'FINGERS' in block:
                for line in block.split('\n'):
                    if '/dev/video' in line:
                        return int(line.strip().replace('/dev/video', ''))
    except Exception:
        pass
    return 0

def _open_cam(st_fn=None):
    idx = _find_cam()
    for attempt in range(3):
        cap = cv2.VideoCapture(idx)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        if cap.isOpened():
            for _ in range(5):
                ret, _ = cap.read()
                if ret:
                    print(f"[RECOG] Camera ok on index {idx}")
                    return cap
                time.sleep(0.2)
        cap.release()
        if st_fn: st_fn(f"Cam retry {attempt+1}", "")
        time.sleep(1)
    for i in range(4):
        if i == idx: continue
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
                print(f"[RECOG] Camera fallback on index {i}")
                return cap
        cap.release()
    return None


# ── Helpers ──────────────────────────────────────────────────────

def _crop(frame, lms):
    h, w, _ = frame.shape
    xs = [int(lm.x * w) for lm in lms]
    ys = [int(lm.y * h) for lm in lms]
    x1, x2 = max(0, min(xs)), min(w, max(xs))
    y1, y2 = max(0, min(ys)), min(h, max(ys))
    
    pad_x, pad_y = int((x2 - x1) * 0.15), int((y2 - y1) * 0.15)
    c = frame[max(0, y1 - pad_y):min(h, y2 + pad_y), max(0, x1 - pad_x):min(w, x2 + pad_x)]
    return c if c.size > 0 else frame

def _ear(idx, lms, w, h):
    p = [(int(lms[i].x*w), int(lms[i].y*h)) for i in idx]
    d = lambda a, b: math.hypot(b[0]-a[0], b[1]-a[1])
    hz = d(p[0], p[3])
    return 0.0 if hz == 0 else (d(p[1],p[5]) + d(p[2],p[4])) / (2*hz)

def _l2(v):
    n = np.sqrt(np.dot(v, v))
    return v/n if n > 0 else v

def _embed(path):
    return np.array(DeepFace.represent(
        img_path=path, model_name="Facenet512",
        detector_backend="skip", enforce_detection=False
    )[0]['embedding'])

def _spoof_check(frame):
    if frame.size == 0: return 0.0, False
    var = cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    return var, var >= ANTISPOOF_THRESH

def _load_user(uid):
    vecs, names = [], []
    mp_p = os.path.join(USER_DB, uid, f"{uid}_masters.json")
    if os.path.exists(mp_p):
        try:
            for pose, meta in json.load(open(mp_p)).items():
                vecs.append(_l2(np.array(meta["vector"])))
                names.append(pose)
        except Exception as e:
            print(f"[RECOG] load masters err: {e}")
    thr = GLOBAL_THRESHOLD
    cp = os.path.join(USER_DB, uid, f"{uid}_threshold.json")
    if os.path.exists(cp):
        try: thr = float(json.load(open(cp))["threshold"])
        except Exception: pass
    return vecs, names, thr


def _load_all_others(claimed_uid):
    others = {}
    try:
        for u in os.listdir(USER_DB):
            if u == claimed_uid: continue
            p     = os.path.join(USER_DB, u)
            mp_p  = os.path.join(p, f"{u}_masters.json")
            thr_p = os.path.join(p, f"{u}_threshold.json")
            if not (os.path.isdir(p) and os.path.exists(mp_p) and os.path.exists(thr_p)):
                continue
            try:
                vecs = [_l2(np.array(m["vector"]))
                        for m in json.load(open(mp_p)).values()]
                thr  = GLOBAL_THRESHOLD
                try: thr = float(json.load(open(thr_p))["threshold"])
                except Exception: pass
                if vecs:
                    others[u] = {"vecs": vecs, "thr": thr}
            except Exception as e:
                print(f"[RECOG] load others err ({u}): {e}")
    except Exception as e:
        print(f"[RECOG] load_all_others err: {e}")
    return others


# ══════════════════════════════════════════════════════════════
def recognize_user(user_id, status_cb=None, cancel_event=None):
    """
    Verify a user's face with up to MAX_FACE_TRIALS blink attempts.
    Returns: "GRANTED" | "DENIED" | "ERROR"
    """
    if cancel_event is None: cancel_event = threading.Event()

    def st(l1, l2=""):
        l1, l2 = l1[:16], l2[:16]
        if status_cb: status_cb(l1, l2)
        print(f"  [LCD] {l1} | {l2}")

    if not os.path.exists(os.path.join(USER_DB, user_id)):
        st("User NOT FOUND", user_id[:16])
        return "DENIED"

    # Load claimed user + all others in background
    master_vecs = []; master_names = []; thr_box = [GLOBAL_THRESHOLD]
    other_users = {}   
    data_ev = threading.Event()

    def _load():
        v, n, t = _load_user(user_id)
        master_vecs.extend(v)
        master_names.extend(n)
        thr_box[0] = t
        other_users.update(_load_all_others(user_id))
        data_ev.set()

    threading.Thread(target=_load, daemon=True).start()

    st("Face Auth", "Opening camera")
    cap = _open_cam(st)
    if cap is None:
        st("Camera FAIL!", "No camera found")
        return "ERROR"

    fm = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
    for _ in range(5): cap.grab()

    try:
        for trial in range(1, MAX_FACE_TRIALS + 1):
            if cancel_event.is_set(): st("Cancelled", ""); return "DENIED"

            st(f"Face try {trial}/{MAX_FACE_TRIALS}", "Blink slowly")
            buf = deque(maxlen=10)
            frames_to_process = []
            deadline  = time.time() + BLINK_TIMEOUT_S

            while True:
                if cancel_event.is_set(): st("Cancelled", ""); return "DENIED"
                if time.time() > deadline:
                    st(f"Try {trial} timeout", ""); break

                ret, frame = cap.read()
                if not ret: time.sleep(0.1); continue

                res = fm.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if not res.multi_face_landmarks:
                    st("No Face Found", "Move closer")
                    time.sleep(0.1)
                    continue

                for lm in res.multi_face_landmarks:
                    h, w, _ = frame.shape
                    ev = (_ear([33,160,158,133,153,144], lm.landmark, w, h) +
                          _ear([362,385,387,263,373,380], lm.landmark, w, h)) / 2
                    buf.append((frame.copy(), list(lm.landmark), ev))
                    rdy = "READY" if data_ev.is_set() else "Loading"
                    st(f"EAR:{ev:.2f} {rdy}", f"Try {trial}/{MAX_FACE_TRIALS}")

                    if ev < BLINK_EAR and len(buf) >= 4:
                        buf_list = list(buf)
                        offset = min(6, len(buf_list) - 3)
                        max_i = len(buf_list) - 1
                        
                        frames_to_process = [
                            buf_list[max_i - offset][0:2],
                            buf_list[max_i - offset - 1][0:2],
                            buf_list[max_i - offset - 2][0:2]
                        ]
                        break

                if frames_to_process: break

            if not frames_to_process:
                continue   

            if not data_ev.is_set():
                st("Loading data", "Please wait")
                data_ev.wait(timeout=15.0)
            if not master_vecs:
                st("No face data!", "Re-enroll")
                return "ERROR"

            cropped = _crop(frames_to_process[0][0], frames_to_process[0][1])
            lap, real = _spoof_check(cropped)
            if not real:
                st("Too Blurry!", "Trying again...")
                print(f"[RECOG] Spoof/Blur rejected. Laplacian={lap:.1f}")
                time.sleep(2.0)
                continue

            st(f"Check face {trial}/{MAX_FACE_TRIALS}", "Please wait...")
            try:
                vecs = []
                for i, (f, lms) in enumerate(frames_to_process):
                    tmp = f"/tmp/auth_{user_id}_{i}_{int(time.time()*1000)}.jpg"
                    crp = _crop(f, lms)
                    cv2.imwrite(tmp, crp)
                    try:
                        v = _embed(tmp)
                        vecs.append(np.array(v))
                    except Exception as e:
                        print(f"[RECOG] Error embedding frame {i}: {e}")
                    finally:
                        if os.path.exists(tmp): os.remove(tmp)
                        
                if not vecs:
                    raise ValueError("Failed to extract embeddings")
                    
                lv = _l2(np.mean(vecs, axis=0))
                dists = [np.linalg.norm(mv - lv) for mv in master_vecs]
                best  = min(dists)
                pose  = master_names[dists.index(best)]
                thr   = thr_box[0]

                # ── Cross-user fraud check ────────────────────────
                print(f"[RECOG] trial={trial} dist={best:.3f} ({pose}) thr={thr} | all_dists: {dict(zip(master_names, [round(d,3) for d in dists]))}")

                impostor_uid = None
                ambiguous = False
                for other_uid, other_data in other_users.items():
                    other_vecs = other_data["vecs"]
                    other_best = min(np.linalg.norm(v - lv) for v in other_vecs)
                    print(f"[RECOG] cross-check vs {other_uid}: dist={other_best:.3f}")
                    
                    # FIXED LOGIC: Decouple fraud threshold from login threshold.
                    # Lowered to 0.40 to allow for shared genetics (Mom/Child).
                    FRAUD_THRESHOLD = 0.40
                    
                    # 1. HARD IMPOSTOR: Match is below absolute 0.40 family fraud line.
                    if other_best < FRAUD_THRESHOLD and other_best <= (best - 0.08):
                        impostor_uid = other_uid
                        break
                        
                    # 2. AMBIGUOUS FAMILY MEMBER: We use a tight 0.05 margin.
                    if best < thr and other_best < FRAUD_THRESHOLD and abs(best - other_best) < 0.05:
                        ambiguous = True

                if impostor_uid:
                    st("IDENTITY FRAUD!", f"Matches ID:{impostor_uid}")
                    print(f"[RECOG] BLOCKED: face matches user {impostor_uid}")
                    time.sleep(2.0)
                    continue 
                    
                if ambiguous:
                    st("AMBIGUOUS FACE", "Move/Try again")
                    print(f"[RECOG] AMBIGUOUS: Margin to another user is < 0.05. Forcing retry.")
                    time.sleep(2.0)
                    continue

                if best < thr:
                    st("Face MATCH!", f"dist:{best:.3f}")
                    return "GRANTED"
                else:
                    st(f"No match {trial}/{MAX_FACE_TRIALS}", f"dist:{best:.3f}")
                    time.sleep(2.0) 

            except Exception as e:
                import traceback; traceback.print_exc()
                st("Embed Error", str(e)[:16])
                return "ERROR"

        st("Face DENIED", "3 tries done")
        time.sleep(2.5)
        return "DENIED"

    except Exception as e:
        import traceback; traceback.print_exc()
        st("ERROR", str(e)[:16])
        return "ERROR"
    finally:
        cap.release()


# ── Standalone ──────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    print("[SYSTEM] Loading Facenet512...")
    DeepFace.build_model("Facenet512")
    print("[SYSTEM] Ready.\n")
    uid = input("Enter User ID: ").strip()
    result = recognize_user(uid, status_cb=lambda l1, l2: print(f"  >>> {l1} | {l2}"))
    print(f"\n[RESULT] {result}")
    sys.exit(0 if result == "GRANTED" else 1)
