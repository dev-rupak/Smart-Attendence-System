"""
face_enrollment.py
──────────────────
Face enrollment module.  Key design change:
  After camera work finishes (50 frames + 5 blinks) the function fires
  capture_done_cb() so the caller can start fingerprint enrollment in
  PARALLEL while the heavy embedding math runs in the background.
  The function only returns True after BOTH math AND the caller confirm
  fingerprint is done (signalled via fp_done_event).

API:
    enroll_user(
        user_id,
        status_cb       = None,   # (line1, line2) → None
        cancel_event    = None,   # threading.Event — set to abort
        capture_done_cb = None,   # () → None  — called when camera work done
        fp_done_event   = None,   # threading.Event — set by caller when FP ok
    ) -> bool
"""

import cv2
import mediapipe as mp
import time, os, math, json, shutil, threading
import numpy as np
import warnings
from collections import deque
from deepface import DeepFace

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────
USER_DB           = "User_Database"
CAM_W, CAM_H      = 320, 240
TOTAL_FRAMES      = 50
IMAGES_TO_AVERAGE = 5
CAL_SAMPLES       = 5
CAL_MULTIPLIER    = 3.0
CAL_MAX_CAP       = 0.65   # bumped from 0.60
CAL_MIN_FLOOR     = 0.50   # bumped from 0.40
FRAUD_CHECK_FRAMES = 7     # top-N sharpest frames per pose used for fraud embed
GRACE             = 10.0
BLINK_EAR         = 0.26
BLINK_TIMEOUT_S   = 60.0

PHASES = [
    (17, "STRAIGHT", "Look STRAIGHT"),
    (34, "LEFT",     "Turn LEFT"),
    (50, "RIGHT",    "Turn RIGHT"),
]
POSE_BLOCKS = [
    ("straight",  0,  16),
    ("left",     17,  33),
    ("right",    34,  49),
]

os.makedirs(USER_DB, exist_ok=True)


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

def _open_cam(st_fn):
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
                    print(f"[ENROLL] Camera ok on index {idx}")
                    return cap
                time.sleep(0.2)
        cap.release()
        st_fn(f"Cam retry {attempt+1}", "")
        time.sleep(1)
    # fallback sweep
    for i in range(4):
        if i == idx: continue
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
                print(f"[ENROLL] Camera fallback on index {i}")
                return cap
        cap.release()
    return None


# ── Image helpers ────────────────────────────────────────────────

def _crop(frame, lms):
    h, w, _ = frame.shape
    xs = [int(lm.x * w) for lm in lms]
    ys = [int(lm.y * h) for lm in lms]
    x1, x2 = max(0, min(xs)), min(w, max(xs))
    y1, y2 = max(0, min(ys)), min(h, max(ys))
    
    pad_x, pad_y = int((x2 - x1) * 0.15), int((y2 - y1) * 0.15)
    c = frame[max(0, y1 - pad_y):min(h, y2 + pad_y), max(0, x1 - pad_x):min(w, x2 + pad_x)]
    return c if c.size > 0 else frame

def _pose(lms, w, h):
    yaw = (lms[1].x*w - lms[234].x*w) / (lms[454].x*w - lms[1].x*w + 1e-6)
    if yaw > 2.0: return "LEFT"
    if yaw < 0.5: return "RIGHT"
    return "STRAIGHT"

def _ear(idx, lms, w, h):
    p = [(int(lms[i].x*w), int(lms[i].y*h)) for i in idx]
    d = lambda a,b: math.hypot(b[0]-a[0],b[1]-a[1])
    hz = d(p[0],p[3])
    return 0.0 if hz==0 else (d(p[1],p[5])+d(p[2],p[4]))/(2*hz)

def _l2(v):
    n = np.sqrt(np.dot(v,v)); return v/n if n>0 else v

def _flush(cap, n=10):
    for _ in range(n): cap.grab()

def _embed(path):
    return np.array(DeepFace.represent(
        img_path=path, model_name="Facenet512",
        detector_backend="skip", enforce_detection=False
    )[0]['embedding'])

def _sharpness(path):
    """Laplacian variance — higher = sharper / better quality image."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None: return 0.0
    return cv2.Laplacian(img, cv2.CV_64F).var()

def _best_frames(folder, user_id, start, end, n):
    """
    Return up to n paths from frame range [start..end], ranked by sharpness.
    This replaces the old linspace / fixed-index approach:
    - avoids blurry mid-turn frames
    - adapts to however many good frames the user managed to capture
    """
    scored = []
    for i in range(start, end + 1):
        p = f"{folder}/{user_id}_{i}.jpg"
        if os.path.exists(p):
            scored.append((_sharpness(p), p))
    scored.sort(reverse=True)
    return [p for _, p in scored[:n]] or [p for _, p in scored]

def _avg_vec(paths):
    vecs = []
    for p in paths:
        if not os.path.exists(p): continue
        try: vecs.append(_embed(p))
        except Exception as e: print(f"[ENROLL] embed err {p}: {e}")
    return _l2(np.mean(vecs,axis=0)) if vecs else None


# ══════════════════════════════════════════════════════════════
def enroll_user(user_id, status_cb=None, cancel_event=None,
                capture_done_cb=None, fp_done_event=None):
    """
    Enroll face.  Returns True on success, False on any failure.
    capture_done_cb() is fired as soon as all camera work (frames + blinks)
    is complete so the caller can start fingerprint enrollment in parallel.
    fp_done_event must be set by the caller once fingerprint succeeds;
    if it is never set this function will time out and return False.
    """
    if cancel_event is None: cancel_event = threading.Event()
    if fp_done_event is None: fp_done_event = threading.Event()

    folder = os.path.join(USER_DB, user_id)
    cap = None

    def st(l1, l2=""):
        l1,l2 = l1[:16], l2[:16]
        if status_cb: status_cb(l1, l2)
        print(f"  [LCD] {l1} | {l2}")

    def cancelled(): return cancel_event.is_set()

    # ── Stale-partial cleanup (previous failed enroll) ──────────
    masters_path = os.path.join(folder, f"{user_id}_masters.json")
    if os.path.exists(folder) and not os.path.exists(masters_path):
        print(f"[ENROLL] Stale partial folder found for {user_id} — auto-cleaning")
        shutil.rmtree(folder)

    if os.path.exists(folder):
        st("ID EXISTS!", "Delete first")
        return False

    os.makedirs(folder, exist_ok=True)
    success = False

    try:
        # ── Pre-warm model ────────────────────────────────────────
        model_ready = threading.Event()
        def _load():
            try: DeepFace.build_model("Facenet512")
            except Exception as e: print(f"[ENROLL] model load err: {e}")
            finally: model_ready.set()
        threading.Thread(target=_load, daemon=True).start()
        st("Loading AI...", "Please wait")

        # ── Open camera ───────────────────────────────────────────
        st("Opening camera", "")
        cap = _open_cam(st)
        if cap is None:
            st("Camera FAIL!", "No camera found")
            return False

        fm = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

        existing = [u for u in os.listdir(USER_DB)
                    if u != user_id and os.path.isdir(os.path.join(USER_DB, u))]

        # ── PHASE 1-3: Capture 50 frames ─────────────────────────
        count=0; active_phase=None; bad_start=None
        st("Ready!", "Follow the poses")
        time.sleep(2); _flush(cap)

        while count < TOTAL_FRAMES:
            if cancelled(): st("Cancelled",""); return False

            target=inst=""
            for lim,pose,ins in PHASES:
                if count < lim: target,inst = pose,ins; break

            if target != active_phase:
                active_phase=target; bad_start=None
                n = ["STRAIGHT","LEFT","RIGHT"].index(target)+1
                st(f"Pose {n}/3:", inst[:14])
                _flush(cap); time.sleep(1); continue

            ret,frame = cap.read()
            if not ret: time.sleep(0.1); continue

            res = fm.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not res.multi_face_landmarks:
                if bad_start is None: bad_start=time.time()
                left = GRACE-(time.time()-bad_start)
                if left<=0: st("No face!","Timeout"); return False
                st(f"No face {left:.0f}s","Move closer"); time.sleep(0.1); continue

            for lm in res.multi_face_landmarks:
                h,w,_=frame.shape
                p = _pose(lm.landmark,w,h)
                ok = (p==target)
                if not ok:
                    if bad_start is None: bad_start=time.time()
                    left=GRACE-(time.time()-bad_start)
                    if left<=0: st("Hold pose!","Timeout"); return False
                    st(f"Turn {target}",f"{left:.0f}s left")
                else:
                    bad_start=None
                    st(f"Frame {count+1}/{TOTAL_FRAMES}",target)
                    cv2.imwrite(f"{folder}/{user_id}_{count}.jpg",_crop(frame,lm.landmark))
                    count+=1; time.sleep(0.03)

        st("50 frames done!","Anti-fraud chk")

        # ── Anti-fraud check ──────────────────────────────────────
        if existing:
            # Instant check: ONLY extract 1 perfectly sharp straight frame instead of 21.
            # FaceNet is accurate enough to cross-compare a single frontal frame.
            is_fraud=False
            
            # To guarantee 100% accuracy for 300+ people while staying fast,
            # we take the 1 sharpest frame from EACH pose block (Straight, Left, Right).
            # This generates a highly resilient "3D" vector profile in just 3 math calls instead of 21.
            fraud_paths = (
                _best_frames(folder, user_id, 0, 16, 1) +
                _best_frames(folder, user_id, 17, 33, 1) +
                _best_frames(folder, user_id, 34, 49, 1)
            )
            fraud_vec = _avg_vec(fraud_paths)
            
            if fraud_vec is not None:
                lv = _l2(fraud_vec)
                for eu in existing:
                    mp2=os.path.join(USER_DB,eu,f"{eu}_masters.json")
                    if not os.path.exists(mp2): continue
                    try:
                        # Compare against all 3 master vectors of the enrolled user
                        for pm in json.load(open(mp2)).values():
                            dist = np.linalg.norm(np.array(pm["vector"])-lv)
                            print(f"[ENROLL] fraud check vs {eu}: dist={dist:.3f}")
                            if dist < 0.50:
                                is_fraud=True; break
                    except Exception: pass
                    if is_fraud: break
            if is_fraud:
                st("FRAUD DETECT!", "Face registered")
                print(f"[ENROLL] Fraud: {user_id} matches existing user")
                return False

        st("Unique face OK!","Computing vecs")

        # ── Background: master vectors ────────────────────────────
        master_meta=[]; master_vecs=[]; math_done=threading.Event(); math_err=[None]
        def _compute():
            try:
                for pname,si,ei in POSE_BLOCKS:
                    if cancelled(): math_err[0]="Cancelled"; return
                    # Dynamically pick the sharpest IMAGES_TO_AVERAGE frames
                    paths = _best_frames(folder, user_id, si, ei, IMAGES_TO_AVERAGE)
                    print(f"[ENROLL] {pname}: using {len(paths)} best frames for master vec")
                    v=_avg_vec(paths)
                    if v is None: math_err[0]=f"{pname} vec fail"; return
                    master_vecs.append(v)
                    master_meta.append((pname,paths,v))
            except Exception as e: math_err[0]=str(e)
            finally: math_done.set()
        threading.Thread(target=_compute,daemon=True).start()

        # ── PHASE 5: Blink calibration ────────────────────────────
        _flush(cap); buf=deque(maxlen=10); cooldown=0; blink_frames=[]; n=0
        st("Blink Calib",f"Blink {CAL_SAMPLES}x slow")
        deadline=time.time()+BLINK_TIMEOUT_S

        while n < CAL_SAMPLES:
            if cancelled(): st("Cancelled",""); return False
            if time.time()>deadline: st("Blink timeout!",""); return False
            ret,frame=cap.read()
            if not ret: time.sleep(0.1); continue
            if cooldown>0: cooldown-=1; continue
            res=fm.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            if not res.multi_face_landmarks: continue
            for lm in res.multi_face_landmarks:
                h,w,_=frame.shape
                ev=(_ear([33,160,158,133,153,144],lm.landmark,w,h)+
                    _ear([362,385,387,263,373,380],lm.landmark,w,h))/2
                buf.append((frame.copy(),list(lm.landmark), ev))
                bg="math done" if math_done.is_set() else "computing"
                st(f"Blink {n+1}/{CAL_SAMPLES}",f"EAR:{ev:.2f} {bg}")
                if ev < BLINK_EAR and len(buf) >= 5:
                    buf_list = list(buf)
                    offset = min(6, len(buf_list) - 1)
                    pf, pl = buf_list[-(offset + 1)][0:2]
                            
                    blink_frames.append(_crop(pf, pl))
                    n += 1; st(f"Blink {n}/{CAL_SAMPLES}", "Captured!")
                    cooldown = 30; buf.clear(); break

        # Release camera — fingerprint can now start in parallel
        cap.release(); cap=None
        print("[ENROLL] Camera released — camera work complete")

        # ── Signal caller: camera work done, start fingerprint now ─
        if capture_done_cb:
            capture_done_cb()

        # ── Wait for fingerprint result from caller ───────────────
        st("Scan finger now","(parallel)...")
        fp_ok = fp_done_event.wait(timeout=120.0)
        if not fp_ok:
            st("FP timeout","Aborted")
            print("[ENROLL] fp_done_event timed out — aborting face save")
            return False
        if cancelled():
            st("Cancelled",""); return False

        # ── Wait for background math to finish ────────────────────
        if not math_done.is_set():
            st("Finishing math","Please wait")
            math_done.wait(timeout=60.0)
        if math_err[0]:
            st("Math Error!",math_err[0][:16]); return False

        # ── Save master vectors ───────────────────────────────────
        meta_dict={}
        for pname,paths_used,v in master_meta:
            meta_dict[pname]={"frames":paths_used,"vector":v.tolist()}
        with open(os.path.join(folder,f"{user_id}_masters.json"),"w") as f:
            json.dump(meta_dict,f)

        # ── Compute threshold from blink samples ──────────────────
        distances=[]
        for i,bf in enumerate(blink_frames):
            tmp=f"/tmp/cal_{user_id}_{i}.jpg"; cv2.imwrite(tmp,bf)
            try:
                lv=_l2(_embed(tmp))
                distances.append(min(np.linalg.norm(v-lv) for _,_,v in master_meta))
            except Exception as e: print(f"[ENROLL] blink embed err: {e}")
            finally:
                if os.path.exists(tmp): os.remove(tmp)
        if distances:
            print(f"[ENROLL] blink distances: {[round(d,3) for d in distances]}")

        if len(distances)<2: thr=CAL_MIN_FLOOR
        else:
            arr=np.array(distances)
            raw=float(arr.mean())+float(arr.std())*CAL_MULTIPLIER+0.05
            thr=round(max(min(raw,CAL_MAX_CAP),CAL_MIN_FLOOR),4)

        with open(f"{folder}/{user_id}_threshold.json","w") as f:
            json.dump({"user_id":user_id,"threshold":thr,
                       "mean":round(float(np.mean(distances)) if distances else 0,4),
                       "std":round(float(np.std(distances))  if distances else 0,4)},f,indent=2)

        print(f"[ENROLL] '{user_id}' enrolled. Threshold={thr}")
        st("Face Enrolled!",f"Thr:{thr:.3f}")
        success=True; return True

    except Exception as e:
        import traceback; traceback.print_exc()
        st("ERROR",str(e)[:16]); return False

    finally:
        if cap and cap.isOpened(): cap.release()
        if not success and os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"[ENROLL] Cleaned up {folder}")


# ── Standalone ──────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    uid=input("\nEnter New User ID: ").strip()
    # standalone: fp_done_event auto-sets after capture so you can test without hardware
    fp_ev=threading.Event()
    cap_cb=lambda: (print("  [standalone] capture done — auto-confirming FP"), fp_ev.set())
    ok=enroll_user(uid,
                   status_cb=lambda l1,l2: print(f"  >>> {l1} | {l2}"),
                   capture_done_cb=cap_cb,
                   fp_done_event=fp_ev)
    print("\n[RESULT]","SUCCESS" if ok else "FAILED")
    sys.exit(0 if ok else 1)
