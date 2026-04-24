"""
attendance_daemon.py — Dual-Biometric Attendance Tracker
Runs natively on the Arduino UNO Q Linux environment.
"""
import sys, os, time, shutil, threading, sqlite3, datetime
from arduino.app_utils import Bridge, App

# Import your existing, unmodified AI modules
import face_enrollment as fe
import face_recognition as fr

USER_DB = "Attendance_Database"
os.makedirs(USER_DB, exist_ok=True)
DB_FILE = os.path.join(USER_DB, "attendance_ledger.db")

_lock = threading.Lock()
_state = { "lcd1": "System Booting", "lcd2": "Please Wait", "result": "IDLE", "busy": False }
_cancel = threading.Event()
_fp_done = threading.Event()

def _set(lcd1=None, lcd2=None, result=None):
    with _lock:
        if lcd1 is not None: _state["lcd1"] = lcd1[:16]
        if lcd2 is not None: _state["lcd2"] = lcd2[:16]
        if result is not None: _state["result"] = result

def _status_cb(l1, l2): 
    _set(lcd1=l1, lcd2=l2)

def _init_db():
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS users (uid TEXT PRIMARY KEY, finger_name TEXT)")
        conn.execute("CREATE TABLE IF NOT EXISTS logs (id INTEGER PRIMARY KEY AUTOINCREMENT, uid TEXT, timestamp DATETIME, log_type TEXT)")
_init_db()

# --- Core Bridge Functions ---

def check_admin_exists():
    with sqlite3.connect(DB_FILE) as conn:
        res = conn.execute("SELECT 1 FROM users WHERE uid='1'").fetchone()
        return "YES" if res else "NO"

def get_user_count():
    with sqlite3.connect(DB_FILE) as conn:
        res = conn.execute("SELECT COUNT(*) FROM users").fetchone()
        return str(res[0]) if res else "0"

def set_voter_finger(data):
    uid, f_name = data.split("|")
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("INSERT OR REPLACE INTO users (uid, finger_name) VALUES (?, ?)", (str(uid), f_name))
    return "OK"

def start_enrollment(uid):
    with _lock:
        if _state["busy"]: return "BUSY"
        _state["busy"] = True
        _state["result"] = "WORKING"
    _cancel.clear()
    _fp_done.clear()

    def _on_capture_done():
        _set(lcd1="Scan finger now", lcd2="(3 tries)", result="CAPTURE_DONE")

    def _run():
        try:
            ok = fe.enroll_user(uid, status_cb=_status_cb, cancel_event=_cancel, capture_done_cb=_on_capture_done, fp_done_event=_fp_done)
            if ok: 
                _set(lcd1="Enroll Success!", lcd2=f"ID:{uid} Saved", result="FACE_DONE")
            else:  
                _set(lcd1="Enroll FAILED", lcd2="Try again", result="FACE_ERROR")
        finally:
            with _lock: _state["busy"] = False

    threading.Thread(target=_run, daemon=True).start()
    return "STARTED"

def fp_success(uid):
    _fp_done.set()
    _set(lcd1="FP done!", lcd2="Finishing math")
    return "OK"

def fp_failed(uid):
    _cancel.set()
    _fp_done.set()
    _set(lcd1="FP FAILED", lcd2="Rolled back", result="FACE_ERROR")
    return "OK"

def log_attendance(uid):
    """Determines if the user is checking IN or OUT, logs it, and returns LCD string."""
    now = datetime.datetime.now()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    time_str = now.strftime("%I:%M %p")

    with sqlite3.connect(DB_FILE) as conn:
        # Find the last log for today
        last_log = conn.execute("SELECT log_type FROM logs WHERE uid=? AND timestamp >= ? ORDER BY timestamp DESC LIMIT 1", (str(uid), today_start)).fetchone()
        
        # Toggle logic: If last was IN, log OUT. Otherwise log IN.
        new_type = "OUT" if last_log and last_log[0] == "IN" else "IN"
        conn.execute("INSERT INTO logs (uid, timestamp, log_type) VALUES (?, ?, ?)", (str(uid), now, new_type))
        
    return f"{new_type}: {time_str}"

def start_recognition(uid):
    with _lock:
        if _state["busy"]: return "BUSY"
        _state["busy"] = True
        _state["result"] = "WORKING"
    _cancel.clear()

    def _run():
        try:
            with sqlite3.connect(DB_FILE) as conn:
                res = conn.execute("SELECT 1 FROM users WHERE uid=?", (str(uid),)).fetchone()
                if not res:
                    _set(lcd1="ID Not Found", lcd2="Enroll First", result="FACE_DENIED")
                    return

            _set(lcd1="Camera Active", lcd2="Look at Lens...")
            res = fr.recognize_user(uid, status_cb=_status_cb, cancel_event=_cancel)
            
            if res == "GRANTED":
                log_msg = log_attendance(uid)
                _set(lcd1="Access GRANTED", lcd2=log_msg, result="FACE_GRANTED")
            else:
                _set(lcd1="Mismatch/Spoof", lcd2="Access Denied", result="FACE_DENIED")
        finally:
            with _lock: _state["busy"] = False

    threading.Thread(target=_run, daemon=True).start()
    return "STARTED"

def delete_user(uid):
    if str(uid) == "1": return "DENIED"
    try:
        p = os.path.join(USER_DB, str(uid))
        if os.path.exists(p): shutil.rmtree(p)
        with sqlite3.connect(DB_FILE) as conn:
            conn.execute("DELETE FROM users WHERE uid=?", (str(uid),))
            conn.execute("DELETE FROM logs WHERE uid=?", (str(uid),)) # Optional: clear their logs too
        return "OK"
    except: return "ERROR"

def full_reset():
    try:
        for u in os.listdir(USER_DB):
            if u != "attendance_ledger.db" and os.path.isdir(os.path.join(USER_DB, u)):
                shutil.rmtree(os.path.join(USER_DB, u))
        with sqlite3.connect(DB_FILE) as conn:
            conn.execute("DELETE FROM users")
            conn.execute("DELETE FROM logs")
        return "OK"
    except: return "ERROR"

def cancel_operation():
    _cancel.set()
    _fp_done.set()
    _set(lcd1="Cancelled", lcd2="", result="IDLE")
    with _lock: _state["busy"] = False
    return "OK"

Bridge.provide("check_admin_exists", check_admin_exists)
Bridge.provide("get_user_count", get_user_count)
Bridge.provide("set_voter_finger", set_voter_finger)
Bridge.provide("start_enrollment", start_enrollment)
Bridge.provide("fp_success", fp_success)
Bridge.provide("fp_failed", fp_failed)
Bridge.provide("start_recognition", start_recognition)
Bridge.provide("delete_user", delete_user)
Bridge.provide("full_reset", full_reset)
Bridge.provide("cancel_operation", cancel_operation)

Bridge.provide("get_lcd1", lambda: _state["lcd1"])
Bridge.provide("get_lcd2", lambda: _state["lcd2"])
Bridge.provide("get_result", lambda: _state["result"])

print("--- SMART ATTENDANCE DAEMON RUNNING ---")
def loop(): time.sleep(0.1)
App.run(user_loop=loop)
