# Igor → Proxmox LXC + Home Assistant Migration

Started 2026-04-18. Delete this file when migration is complete and notes are absorbed into CLAUDE.md / wiki.

## Why

1. **Move Igor off the user's PC** to dedicated always-on infra (LXC on Proxmox node `Snap` at 10.0.30.10).
2. **Delegate device actions to Home Assistant** (running at 10.0.40.5, VLAN 40 IoT). HA already integrates LIFX / Sonos / androidtv — drop ~5 Python deps and stop maintaining per-device code.

## Architecture Decisions

- **Igor stays the brain.** Custom prompt, living memory (identity narrative + episodic + consolidation), quality gate, intent router, multi-client routing, Claude tool loop — all stays. We considered HA Voice (Wyoming protocol, Assist) and rejected it: it would replace the brain, not just the mouth.
- **STT/TTS stay local in Igor.** Faster-whisper small (int8) + Kokoro `am_onyx` are tuned and the voice is part of Igor's identity. HA's Piper voices and Wyoming pipeline aren't worth the loss.
- **HA owns device actions.** All `lifx_cmd / sonos_cmd / tv_cmd / adb_cmd` get replaced by a thin `ha_client.py` + intent-shaped `Command` classes that call HA REST services.
- **HA Areas are the source of truth for room → entity mapping.** No more hardcoded device lists in `rooms.yaml`. We query HA at startup (templating API or WS area_registry).
- **`rooms.yaml` shrinks to client identity only**: `client_id → ha_area + callback_url + prefer_sonos_output + indicator_light`. Everything else comes from HA.
- **LXC, not VM.** Per ADR-002 (wirenest wiki): no kernel needs, no hardware passthrough. **4 cores / 4GB RAM** to start, resize live if needed.

## HA Inventory (snapshot 2026-04-18, HA 2026.4.3)

64 entities, key ones for Igor:
- **Lights (4)**: `light.tall_lamp`, `light.table_lamp`, `light.office_lamp`, `light.corner_lamp`
- **Media (3)**: `media_player.giant_ass_teevee` (×2 — likely Cast + androidtv), `media_player.living_room` (Sonos)
- **Remote**: `remote.giant_ass_teevee` (TV nav keys)
- **Sonos feature switches (6)**: `switch.living_room_{crossfade,loudness,surround_music_full_volume,night_sound,speech_enhancement,surround_enabled}`
- **Presence**: `person.sam`, `device_tracker.sams_pphone`
- **Other usable**: `todo` domain (HA shopping/tasks list), `notify.*` services (push to phone via Companion app)
- **Areas**: confirmed configured by user — use these for room scoping

## Sequenced Plan

Order matters: each step is independently testable. A bug in step 2 should not raise doubt about step 1.

### Step 1 — Bare migration to LXC (no refactor)

Goal: Igor running on the LXC with the existing PC code, Pi clients pointing at the new IP, end-to-end interaction working.

- [ ] Provision Debian 12 LXC on Snap, 4C / 4GB / ~20GB disk, VLAN 30 (Servers)
- [ ] Assign static IP — **TODO: pick IP, e.g. 10.0.30.30** — then update `wirenest` services-registry + Pi-hole DNS entry (`igor.kingdahm.com`)
- [ ] Install system deps: `python3.12 python3-venv git ffmpeg sox alsa-utils` (+ whatever Kokoro needs)
- [ ] Clone repo, create venv, `pip install -r requirements.txt` (or pyproject)
- [ ] Copy `data/brain.json` from PC (`~/OneDrive/.../igor/data/`) — preserves memory + feedback
- [ ] Copy `oww_models/igor.onnx` and `oww_models/igor_verifier.pkl` — wakeword models (only needed by Pi clients, but keep on server for training reruns)
- [ ] Copy `kokoro/kokoro-v1.0.onnx` + `kokoro/voices-v1.0.bin`
- [ ] Set env vars: `ANTHROPIC_API_KEY`, `SERVER_HOST=<new LXC IP>`, `SERVER_EXTERNAL_HOST=<same>`, `PI_HOST=192.168.0.3` (or whatever)
- [ ] Create systemd unit `igor-server.service` — `ExecStart=python -m server.main`, `Restart=on-failure`, `User=igor`
- [ ] Update each Pi/PC client's `client/config.py` (or env): `SERVER_HOST=<new LXC IP>`
- [ ] Update `data/rooms.yaml` callback_urls if Pi IPs need to change
- [ ] Health check: `curl http://<lxc-ip>:8000/api/health`
- [ ] End-to-end: wake word → "what time is it" → "set a 30 second timer"
- [ ] Add to Uptime Kuma

### Step 2 — HA action layer refactor (do on the LXC, not the PC)

Goal: replace per-device commands with HA-backed intent commands. Same LLM tool surface, new implementation.

- [ ] **Get fresh HA long-lived token on LXC** — Windows token at `C:\Users\samda\.ha_token.txt` (UTF-16 BOM) won't move. Generate new one in HA UI at `http://10.0.40.5:8123/profile/security` and store in `/etc/igor/ha_token` (root-only) or as systemd `EnvironmentFile=` secret.
- [ ] `server/ha_client.py` — single REST client:
  - auth, GET `/api/states`, POST `/api/services/{domain}/{service}`
  - cached entity registry refreshed on startup + every N min (or WS subscribe later)
  - area lookup via templating API: `POST /api/template` with `{{ area_entities('Office') }}`
- [ ] Replace `server/commands/lifx_cmd.py` → `light_cmd.py` (HA-backed):
  - `set_light`, `set_brightness`, `set_color`, `set_color_temp`, `adjust_brightness`, `adjust_color_temp`, `shift_hue`, `list_lights`, `list_scenes`, `set_scene`
  - All resolve entity_ids by area from `_ctx.room.ha_area`
- [ ] Replace `sonos_cmd.py` + part of `system_cmd.py` → `media_cmd.py`:
  - `set_volume / adjust_volume / sonos_mute` for Sonos (and TV media_player)
  - `play / pause / next / previous`
  - Entity resolution: room's `media_player` entities, prefer Sonos for music intent
- [ ] Replace `tv_cmd.py` + `adb_cmd.py` → `tv_cmd.py` (HA-backed):
  - `tv_power`, `tv_key` → `remote.send_command` on `remote.giant_ass_teevee`
  - `tv_launch`, `tv_playback`, `tv_skip`, `tv_search_youtube` → `media_player.*` services + `play_media` for YouTube
  - **Drop the TV state poller entirely** — read `media_player.giant_ass_teevee.state` from HA on demand. Update `_get_tv_playback_state()` to query HA.
- [ ] **NEW**: `todo_cmd.py`:
  - `add_todo(item, list)`, `complete_todo(item, list)`, `list_todo(list)`
  - HA `todo.add_item / update_item / get_items`
  - List defaults to user's primary HA todo list
- [ ] **NEW**: `notify_cmd.py`:
  - `notify(message, target=None, title=None)` — push to phone via HA Companion app
  - Discover available `notify.*` services on startup
- [ ] Remove deps: `lifxlan`, `soco`, `androidtvremote2`, `adb-shell` (if no longer used)
- [ ] Remove `server/pair_google_tv.py` (TV pairing now HA's problem)

### Step 3 — Shrink `rooms.yaml`, switch to HA areas

Goal: stop maintaining device lists in two places.

- [ ] New `RoomConfig` shape: `client_id, ha_area, callback_url, prefer_sonos_output, indicator_light` only
- [ ] Migrate existing `rooms.yaml` (one-time edit, will write a script if there are multiple rooms)
- [ ] `RoomStateManager`: TV state cache → just call HA on demand (or cache for ≤1s); drop the polling thread
- [ ] Update `data/rooms.yaml.example` and CLAUDE.md to reflect new shape

## Open Questions / TODOs

- LXC IP address — needs to be picked + reserved in pfSense and added to Pi-hole DNS (`igor.kingdahm.com` → IP)
- The `device_tracker.sams_pphone` data corruption (`sam`s pphone` with replacement char) suggests an encoding issue somewhere — check phone's HA Companion config name
- HA marked "Planned" in `wirenest` wiki but is actually live — update `services-registry.md` and `adr-002` after Igor migration succeeds
- Decide where to store Igor secrets on LXC: SOPS-encrypted in `/etc/igor/`, or systemd `EnvironmentFile`, or both. Wiki convention says SOPS — follow that.

## Where We Left Off

Done in this session:
- Trained new wakeword model (`igor.onnx`) with template-matched windowing — committed `6c1631d`
- Verified HA is live at 10.0.40.5, enumerated entities, picked architecture
- Wrote this plan

Not yet started:
- Step 1 (LXC provisioning) — waiting on IP + token regeneration
