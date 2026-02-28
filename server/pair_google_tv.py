#!/usr/bin/env python3
"""One-time Google TV pairing script.

Run this once before using tv_power / tv_key commands:
    python server/pair_google_tv.py

The TV will display a PIN code. Enter it here to complete pairing.
Certificates are saved to data/ and reused by tv_cmd.py automatically.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from server.config import (
    GOOGLE_TV_HOST,
    GOOGLE_TV_CERT_FILE,
    GOOGLE_TV_KEY_FILE,
    GOOGLE_TV_CLIENT_NAME,
)

try:
    from androidtvremote2 import AndroidTVRemote, CannotConnect, ConnectionClosed, InvalidAuth
except ImportError:
    print("ERROR: androidtvremote2 not installed. Run: pip install androidtvremote2")
    sys.exit(1)


async def main():
    if not GOOGLE_TV_HOST or "0.X" in GOOGLE_TV_HOST:
        print("ERROR: Set GOOGLE_TV_HOST in server/config.py to your TV's LAN IP first.")
        sys.exit(1)

    remote = AndroidTVRemote(
        client_name=GOOGLE_TV_CLIENT_NAME,
        certfile=GOOGLE_TV_CERT_FILE,
        keyfile=GOOGLE_TV_KEY_FILE,
        host=GOOGLE_TV_HOST,
    )

    generated = await remote.async_generate_cert_if_missing()
    if generated:
        print(f"Generated certificates:\n  {GOOGLE_TV_CERT_FILE}\n  {GOOGLE_TV_KEY_FILE}")
    else:
        print("Certificates already exist.")

    print(f"\nConnecting to Google TV at {GOOGLE_TV_HOST} as '{GOOGLE_TV_CLIENT_NAME}'...")
    print("Your TV should prompt you to allow the connection and show a PIN.\n")

    while True:
        try:
            await remote.async_start_pairing()
            pin = input("Enter the PIN shown on your TV: ").strip()
            await remote.async_finish_pairing(pin)
            print("\nPairing successful. tv_power and tv_key commands are ready.")
            break
        except InvalidAuth:
            print("Wrong PIN — try again.")
        except (CannotConnect, ConnectionClosed, OSError) as e:
            print(f"\nConnection failed: {e}")
            print("Make sure the TV is on and GOOGLE_TV_HOST is correct.")
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
