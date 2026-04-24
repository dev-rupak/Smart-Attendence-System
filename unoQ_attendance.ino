#include <Arduino_RouterBridge.h>
#include <Adafruit_Fingerprint.h>
#include <Wire.h>
#include <hd44780.h>
#include <hd44780ioClass/hd44780_I2Cexp.h>

Adafruit_Fingerprint finger = Adafruit_Fingerprint(&Serial);
hd44780_I2Cexp lcd;

// --- Hardware Configuration ---
int rowPins[4] = {2, 3, 4, 5};
int colPins[4] = {A2, A3, A4, A5};
char keys[4][4] = {
  {'1','2','3','A'},
  {'4','5','6','B'},
  {'7','8','9','C'},
  {'*','0','#','D'}
};

String secretBuffer = "";

void setup() {
  Bridge.begin();
  lcd.begin(16, 2);
  
  for(int i=0; i<4; i++) { 
    pinMode(colPins[i], INPUT_PULLUP); 
    pinMode(rowPins[i], OUTPUT); 
    digitalWrite(rowPins[i], HIGH); 
  }
  
  finger.begin(57600);
  lcdPrint("Attendance OS", "Booting AI...");
  delay(2000);
  
  if (!finger.verifyPassword()) {
    lcdPrint("FP Sensor Err!", "Check Wiring");
    while(1);
  }
  
  // Rule: Force Admin enrollment on fresh boot
  String adminExists = "";
  Bridge.call("check_admin_exists").result(adminExists);
  if (adminExists != "YES") {
    lcdPrint("No Admin Found", "Enrolling ID: 1");
    delay(2000);
    enrollAdminFlow();
  }
  
  showMenu();
}

void loop() {
  char key = readKeypad();
  if (!key) return;
  
  secretBuffer += key;
  if (secretBuffer.length() > 3) secretBuffer = secretBuffer.substring(secretBuffer.length() - 3);
  
  if (secretBuffer == "*0#") {
    secretBuffer = "";
    lcdPrint("FACTORY RESET!", "Wiping Ledger");
    finger.emptyDatabase();
    Bridge.call("full_reset");
    delay(2000);
    lcdPrint("Reset Complete", "Rebooting...");
    while(1); 
  }

  // --- Menu Routing ---
  if (key == 'A') {
    secretBuffer = "";
    if (requireAdminAuth()) enrollFlow();
    showMenu();
  } 
  else if (key == 'B') {
    secretBuffer = "";
    attendanceFlow();
    showMenu();
  }
  else if (key == 'C') {
    secretBuffer = "";
    String count = "";
    Bridge.call("get_user_count").result(count);
    lcdPrint("Total Employees:", count.c_str());
    delay(4000);
    showMenu();
  }
  else if (key == 'D') {
    secretBuffer = "";
    if (requireAdminAuth()) {
      String delID = getIdFromKeypad("Remove Emp", "Del ID:");
      if (delID != "") {
        finger.deleteModel(delID.toInt());
        Bridge.call("delete_user", delID.c_str());
        lcdPrint("Employee Deleted", delID.c_str());
        delay(2000);
      }
    }
    showMenu();
  }
}

// --- Core Functions ---

bool requireAdminAuth() {
  lcdPrint("Admin Required", "Verifying ID 1");
  delay(1000);
  if (!scanFinger(1, 3)) {
    lcdPrint("Admin FP Denied", ""); delay(2000); return false;
  }
  Bridge.call("start_recognition", "1");
  if (awaitPythonResult("FACE_GRANTED", "FACE_DENIED", 45000)) {
    lcdPrint("Admin Verified", "Access Granted"); delay(1500); return true;
  }
  return false;
}

void enrollAdminFlow() {
  lcdPrint("Starting face", "scan...");
  Bridge.call("start_enrollment", "1");
  
  if (!awaitPythonResult("CAPTURE_DONE", "FACE_ERROR", 180000UL)) {
    lcdPrint("Face capture", "FAILED"); delay(2500); return;
  }
  
  lcdPrint("Scan Finger Now", "(Master Admin)");
  if(enrollFinger(1, 3)) {
    Bridge.call("fp_success", "1");
    if(awaitPythonResult("FACE_DONE", "FACE_ERROR", 150000UL)) {
      lcdPrint("ADMIN SAVED", "ID: 1"); delay(3000); return;
    }
  }
  Bridge.call("fp_failed", "1");
  lcdPrint("Enroll FAILED", "Rolled Back"); delay(2000);
}

void enrollFlow() {
  String uid = getIdFromKeypad("Enroll Employee", "Emp ID:");
  if (uid == "") return;
  
  lcdPrint("Starting face", "scan...");
  Bridge.call("start_enrollment", uid.c_str());
  
  if (!awaitPythonResult("CAPTURE_DONE", "FACE_ERROR", 180000UL)) {
    lcdPrint("Face capture", "FAILED"); delay(2500); return;
  }
  
  // Choose Finger logic
  lcdPrint("Select Hand:", "1:Right 2:Left");
  char h = waitForNumKey('1', '2');
  lcdPrint("1Th 2Ix 3Md 4Ri", "5Pi");
  char f = waitForNumKey('1', '5');
  String hand = (h == '1') ? "R-" : "L-";
  String fingerName = hand + "Finger " + f;
  
  String payload = uid + "|" + fingerName;
  Bridge.call("set_voter_finger", payload.c_str());
  
  lcdPrint("Scan Finger Now", fingerName.c_str());
  if(enrollFinger(uid.toInt(), 3)) {
    Bridge.call("fp_success", uid.c_str());
    if(awaitPythonResult("FACE_DONE", "FACE_ERROR", 150000UL)) {
      lcdPrint("EMPLOYEE SAVED", uid.c_str()); delay(3000); return;
    }
  }
  Bridge.call("fp_failed", uid.c_str());
  lcdPrint("Enroll FAILED", "Rolled Back"); delay(2000);
}

void attendanceFlow() {
  String uid = getIdFromKeypad("Log Attendance", "Your ID:");
  if (uid == "") return;
  
  lcdPrint("Step 1: FP", "Place finger...");
  if (!scanFinger(uid.toInt(), 3)) {
    lcdPrint("> DENIED <", "FP Mismatch"); delay(3000); return;
  }
  
  lcdPrint("FP: PASSED", "Step 2: Face...");
  delay(1000);
  
  Bridge.call("start_recognition", uid.c_str());
  
  // The Python Daemon returns the exact IN/OUT timestamp via the LCD mirrors!
  if (awaitPythonResult("FACE_GRANTED", "FACE_DENIED", 60000)) {
    delay(4000); // Leave the time on the screen for the user to see
  } else {
    delay(2000);
  }
}

// --- Helpers ---

bool enrollFinger(int id, int tries) {
  for (int t = 1; t <= tries; t++) {
    lcdPrint("Place finger...", "");
    while (finger.getImage() != FINGERPRINT_OK) { if (readKeypad() == '*') return false; delay(50); }
    if (finger.image2Tz(1) != FINGERPRINT_OK) { delay(1000); continue; }
    lcdPrint("Remove finger", ""); delay(1500);
    while (finger.getImage() != FINGERPRINT_NOFINGER) delay(100);
    lcdPrint("Same angle", "Scan again...");
    while (finger.getImage() != FINGERPRINT_OK) { if (readKeypad() == '*') return false; delay(50); }
    if (finger.image2Tz(2) != FINGERPRINT_OK) { delay(1000); continue; }
    if (finger.createModel() != FINGERPRINT_OK) { delay(1500); continue; }
    if (finger.storeModel(id) == FINGERPRINT_OK) return true;
  }
  return false;
}

bool scanFinger(int expectedId, int tries) {
  for (int t = 1; t <= tries; t++) {
    unsigned long ts = millis();
    bool gotImage = false;
    while (millis()-ts < 10000UL) {
      if (readKeypad() == '*') return false;
      if (finger.getImage() == FINGERPRINT_OK) { gotImage = true; break; }
      delay(50);
    }
    if (!gotImage) continue;
    if (finger.image2Tz() != FINGERPRINT_OK) continue;
    if (finger.fingerFastSearch() != FINGERPRINT_OK) continue;
    if (finger.fingerID == expectedId && finger.confidence >= 60) return true;
    lcdPrint("Try again", ""); delay(1200);
  }
  return false;
}

bool awaitPythonResult(const char* ok, const char* fail, unsigned long ms) {
  unsigned long start = millis();
  while (millis()-start < ms) {
    if (readKeypad() == '*') { Bridge.call("cancel_operation"); lcdPrint("Cancelled",""); delay(1500); return false; }
    
    String l1 = "", l2 = "", res = "";
    Bridge.call("get_lcd1").result(l1);
    Bridge.call("get_lcd2").result(l2);
    Bridge.call("get_result").result(res);
    
    if (l1.length() > 0) lcdPrint(l1.c_str(), l2.c_str());
    if (res == ok) return true;
    if (res == fail) { delay(2500); return false; }
    delay(600);
  }
  Bridge.call("cancel_operation"); lcdPrint("Timed out",""); delay(1500); return false;
}

char waitForNumKey(char minKey, char maxKey) {
  while(true) {
    char k = readKeypad();
    if(k >= minKey && k <= maxKey) { delay(200); return k; }
  }
}

String getIdFromKeypad(const char* title, const char* prefix) {
  String id = "";
  while (true) {
    lcdPrint(title, (String(prefix) + " " + id + "_").c_str());
    char k = readKeypad();
    if (k >= '0' && k <= '9' && id.length() < 4) id += k;
    else if (k == '#') { if (id.length() > 0) id.remove(id.length() - 1); }
    else if (k == 'B' && id.length() > 0) return id; 
    else if (k == '*') return ""; 
    delay(100); 
  }
}

void showMenu() { lcdPrint("A:Add B:Log In/Out", "C:Count D:Remove"); }
void lcdPrint(const char* l1, const char* l2) { lcd.clear(); lcd.setCursor(0, 0); lcd.print(l1); lcd.setCursor(0, 1); lcd.print(l2); }

char readKeypad() {
  for(int r=0; r<4; r++) { digitalWrite(rowPins[r], LOW);
    for(int c=0; c<4; c++) {
      if(digitalRead(colPins[c]) == LOW) { delay(50); while(digitalRead(colPins[c]) == LOW); digitalWrite(rowPins[r], HIGH); return keys[r][c]; }
    } digitalWrite(rowPins[r], HIGH);
  } return 0;
}
