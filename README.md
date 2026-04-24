# 🏢 Smart Biometric Attendance System (Zero-Spoof)

An enterprise-grade, touchless Smart Attendance System powered by the **Arduino UNO Q**. 

To completely eliminate "buddy punching" and spoofing, this system enforces a strict **Dual-Factor Biometric Architecture** (Fingerprint + 3D Facial Recognition). It leverages advanced Machine Learning models running directly on the UNO Q's Linux microprocessor, while logging dynamic Check-In/Check-Out timestamps locally to an SQLite ledger.

## 🚀 Key Features

* **Dual-Biometric Verification:** Employees must provide a registered fingerprint, followed immediately by a facial scan. This makes the system virtually impossible to spoof.
* **Intelligent Auto-Toggle:** The system automatically determines if an employee is checking in or out based on their last log entry for the day.
* **Zero-Spoof Face Authentication:** This system is immune to photo or video playback attacks. It enforces strict liveness detection (requiring a physical blink) and runs Laplacian texture checking to analyze image depth.
* **Edge-AI Processing:** No cloud connectivity is required. The heavy `Facenet512` model runs entirely locally on the UNO Q's Linux processor, ensuring absolute privacy.
* **Master Admin Governance:** The system cannot be modified without the physical presence of the Master Administrator (ID: 1). Adding or deleting users requires an active, verified Admin dual-factor scan.

## 📋 The Database Ledger
All attendance logs are stored on the Linux environment in an SQLite database (`attendance_ledger.db`). The system tracks:
* `id`: The log sequence number.
* `uid`: The employee's identification number.
* `timestamp`: The exact DateTime the biometric scan was approved.
* `log_type`: Automatically tagged as either `IN` or `OUT`.

## 🕹️ Keypad Administrative Menu

The 4x4 matrix keypad operates as a strict administrative menu:

* `A` - **Add Employee:** Prompts the Admin to verify their identity, then initiates the camera and fingerprint enrollment process for a new employee ID.
* `B` - **Log Attendance:** The standard entry mode. The employee types their ID, scans their finger, and looks at the camera. Upon success, the LCD displays their IN/OUT timestamp.
* `C` - **Count Employees:** Displays the total number of registered employees currently stored in the database.
* `D` - **Remove Employee:** Prompts the Admin to verify their identity, then prompts for a target ID to permanently wipe from the database.
* `*0#` - **Factory Reset:** A hidden command that drops all SQLite tables, deletes all stored facial embeddings, and clears the fingerprint sensor memory.

*(Note: On a fresh boot, if the database is empty, the system will bypass the menu and immediately force the enrollment of the Admin (ID: 1).*

## 🛠️ Hardware Requirements

* 1x Arduino UNO Q (SBC)
* 1x R307S Optical Fingerprint Sensor
* 1x USB Web Camera
* 1x 4x4 Matrix Keypad
* 1x 16x2 I2C LCD Display

## 🔌 Circuit Configuration

* **USB Camera:** Plugged directly into the UNO Q's USB host port.
* **Fingerprint Sensor (R307S):** Hardware Serial Pins `0 (RX)` and `1 (TX)`.
  * *(Important: Disconnect these pins when uploading firmware via USB to avoid serial conflicts).*
* **Keypad:** Rows connected to Digital Pins `2, 3, 4, 5`. Columns connected to Analog Pins `A2, A3, A4, A5`.
* **LCD Display:** SDA and SCL connected to the UNO Q's I2C bus.

## 💻 Firmware & Software Setup

1.  **Dependencies:** Access the UNO Q's Linux terminal and install the required AI libraries:
    ```bash
    pip3 install opencv-python mediapipe deepface numpy
    ```
2.  **MCU Firmware:** Upload `unoQ_attendance.ino` to the MCU using the Arduino IDE.
3.  **Linux Daemon:** Ensure `attendance_daemon.py`, `face_enrollment.py`, and `face_recognition.py` are all in the same folder on the UNO Q's Linux environment.
4.  **Launch:** Execute `python3 attendance_daemon.py` on the Linux side to start the attendance ledger and await commands from the keypad.
