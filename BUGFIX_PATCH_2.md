# Security & Bug Fix Patch #2 - 2026-01-19

## Issues Found and Fixes Applied

### Critical Issues

#### 1. ❌ Missing Import in client/audio.py
**File:** [client/audio.py](client/audio.py)
**Issue:** Imports `pyaudio` before suppressing ALSA errors, causing warnings on import

**Status:** Already handled - stderr suppression is done before import

---

#### 2. ⚠️ No Timeout on Pi Server Subprocess Calls
**File:** [client/hardware.py](client/hardware.py:23-28)
**Issue:** `amixer` calls have timeout=5, but beep sounds in [client/audio.py](client/audio.py) don't

**Fix:** Already have timeout=1 on all beep calls - GOOD

---

#### 3. ⚠️ Potential Race Condition in Event Loop Timer Checking
**File:** [server/event_loop.py](server/event_loop.py:90-97)
**Issue:** Creating list() while iterating could cause issues if timer added during iteration

**Status:** Using `list(self._timers.items())` creates a snapshot - SAFE

---

### High Priority Issues

#### 4. ⚠️ No Validation of audio_bytes Size Before Processing
**File:** [server/orchestrator.py](server/orchestrator.py:58)
**Issue:** `transcribe_bytes()` receives potentially huge audio files with no size check

**Fix Needed:** Add size validation before transcription

---

#### 5. ⚠️ Flask Debug Mode in Production
**File:** [client/pi_server.py](client/pi_server.py:238)
**Issue:** `debug=False` is set correctly - GOOD

---

#### 6. ⚠️ Missing Exception Handling in Main Loops
**File:** [server/main.py](server/main.py)
**Issue:** No top-level exception handling for service initialization failures

**Fix Needed:** Better error messages and cleanup on failure

---

#### 7. ⚠️ Temp File Not Cleaned Up on Error in transcribe_bytes
**File:** [server/transcription.py](server/transcription.py:69-83)
**Issue:** Uses try/finally correctly - GOOD

---

### Medium Priority Issues

#### 8. ⚠️ No Rate Limiting on Pi Server Endpoints
**File:** [client/pi_server.py](client/pi_server.py)
**Issue:** PC could spam Pi with requests causing DoS

**Risk:** LOW on trusted network, but should add basic rate limiting

---

#### 9. ⚠️ Logging Sensitive Network Info
**File:** [server/pi_callback.py](server/pi_callback.py)
**Issue:** Logs full Pi URL which includes IP addresses

**Risk:** LOW - logs are local only

---

#### 10. ⚠️ No Retry Logic for Network Failures
**File:** [client/main.py](client/main.py:144-171), [server/pi_callback.py](server/pi_callback.py)
**Issue:** Single network failure causes interaction to fail

**Enhancement:** Could add retry with exponential backoff

---

### Low Priority / Code Quality

#### 11. 📝 Unused Variable in server/main.py
**File:** [server/main.py](server/main.py:65)
**Issue:** `event_loop` variable assigned but not used

**Fix:** This is intentional - starts the event loop as a side effect

---

#### 12. 📝 Missing Type Hints in Some Functions
**Issue:** Some callback functions lack type hints

**Fix:** Add type hints for better IDE support

---

#### 13. 📝 Potential Import Circular Dependency
**Issue:** `server.commands` imports from `server.event_loop` which imports nothing from commands

**Status:** No circular dependency - SAFE

---

## Actual Bugs Found

### Bug 1: Missing Validation for Audio Size
**Location:** [server/orchestrator.py:58](server/orchestrator.py#L58)

**Issue:**
```python
transcription = self.transcriber.transcribe_bytes(audio_bytes)
```

No validation that `audio_bytes` is reasonable size. Pydantic validates base64 string length but not decoded bytes.

**Impact:** Could cause memory issues with crafted requests

**Fix:** Add size check in orchestrator

---

### Bug 2: CORS Allow Origins Too Permissive
**Location:** [server/api.py:35](server/api.py#L35)

**Issue:**
```python
allow_origins=["*"],  # In production, restrict to Pi's IP
```

**Impact:** Any origin can make requests

**Fix:** Change to specific Pi IP

---

### Bug 3: No Validation of transcription Text Length
**Location:** [server/orchestrator.py:72](server/orchestrator.py#L72)

**Issue:** Transcription could be extremely long, causing issues in LLM processing

**Fix:** Add max length check

---

### Bug 4: Event Loop Thread Not Stopped on Server Shutdown
**Location:** [server/main.py](server/main.py)

**Issue:** Event loop starts but never explicitly stopped on KeyboardInterrupt

**Impact:** Daemon thread, so not critical, but unclean shutdown

**Fix:** Add cleanup in exception handler

---

### Bug 5: Pi Server Thread Not Stopped on Client Shutdown
**Location:** [client/main.py:98](client/main.py#L98)

**Issue:** Pi server thread is daemon, but Flask doesn't shut down cleanly

**Impact:** Thread may hang on exit

**Fix:** Use threading.Event() for clean shutdown

---

### Bug 6: No Check if Pi is Reachable Before Starting Server
**Location:** [server/main.py](server/main.py)

**Issue:** Server starts even if Pi is unreachable

**Impact:** Timer alerts will fail silently

**Fix:** Add startup check or warning

---

### Bug 7: Missing Audio System Check in Client Initialize
**Location:** [client/main.py:52-66](client/main.py#L52-L66)

**Issue:** If audio.initialize() fails, wakeword_detector still tries to init

**Fix:** Return early on audio failure - ALREADY DONE

---

## Fixes to Apply

### Fix 1: Add Audio Size Validation
