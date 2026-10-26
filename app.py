
"""
Drift — personal AI assistant backend
------------------------------------------------------
A lightweight local AI assistant with:
  - persistent per-user memory
  - optional email / Slack / Telegram / iMessage delivery
  - web scraping and document reading
  - spoken output when running on macOS
  - a /ask endpoint so the Drift browser UI can talk to it live

Run it:
  pip install flask flask-cors crewai crewai-tools python-dotenv requests
  python app.py

Then in hermes.html -> Configure -> Backend URL:
  http://localhost:5000/ask
"""
 
import os
import json
import re
import secrets
import subprocess
import smtplib
import requests
import threading
import webbrowser
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText
from pathlib import Path
from urllib.parse import urlencode, urlparse, quote_plus
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_file, Response, redirect, session
from flask_cors import CORS
import uuid
import time
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from crewai_tools import FileReadTool, ScrapeWebsiteTool
 
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR / "env")

GOOGLE_CALENDAR_SCOPE = "https://www.googleapis.com/auth/calendar.events.readonly"
GOOGLE_TOKEN_FILE = BASE_DIR / "google_calendar_token.json"

YOUTUBE_SCOPE = "https://www.googleapis.com/auth/youtube.readonly"
YOUTUBE_TOKEN_FILE = BASE_DIR / "youtube_token.json"

CONTACTS_SCOPE = "https://www.googleapis.com/auth/contacts.readonly"
CONTACTS_TOKEN_FILE = BASE_DIR / "contacts_token.json"

SPOTIFY_SCOPE = "user-read-private user-read-email playlist-read-private playlist-read-collaborative user-library-read user-read-playback-state user-modify-playback-state user-read-currently-playing"
SPOTIFY_TOKEN_FILE = BASE_DIR / "spotify_token.json"
 
def build_llm():
    """Build the configured provider without changing case-sensitive model IDs."""
    configured_model = (os.getenv("CEBRI_MODEL") or os.getenv("MODEL") or "").strip()
    normalized_model = configured_model.lower()
    configured_provider = os.getenv("CEBRI_PROVIDER", "").strip().lower()
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    gonkarouter_key = os.getenv("GONKAROUTER_API_KEY") or os.getenv("GONKA_API_KEY")

    def try_model(
        provider_label: str,
        model_name: str,
        api_key: str | None,
        **llm_options,
    ):
        if not api_key:
            return None
        try:
            return LLM(model=model_name, api_key=api_key, **llm_options)
        except Exception as exc:
            print(f"{provider_label} LLM init failed for {model_name}: {exc}")
            return None

    if configured_provider in {"gonka", "gonkarouter"}:
        model_name = configured_model or "MiniMaxAI/MiniMax-M2.7"
        return try_model(
            "GonkaRouter",
            f"openai/{model_name}",
            gonkarouter_key,
            base_url=os.getenv("GONKAROUTER_BASE_URL", "https://api.gonkarouter.io/v1"),
        )

    if configured_provider in {"openai", "openai_api"}:
        model_name = configured_model or "gpt-5-mini"
        return try_model("OpenAI", model_name, openai_key)

    if normalized_model.startswith("anthropic/") or normalized_model.startswith("claude") or (
        not configured_model and anthropic_key
    ):
        model_name = configured_model if configured_model else "anthropic/claude-sonnet-4-6"
        return try_model("Anthropic", model_name, anthropic_key)

    if normalized_model.startswith("openai/") or normalized_model.startswith("gpt") or (
        not configured_model and openai_key
    ):
        model_name = configured_model if configured_model else "gpt-4o-mini"
        return try_model("OpenAI", model_name, openai_key)

    if anthropic_key:
        result = try_model("Anthropic", "anthropic/claude-sonnet-4-6", anthropic_key)
        if result is not None:
            return result

    if openai_key:
        result = try_model("OpenAI", "gpt-4o-mini", openai_key)
        if result is not None:
            return result

    return None


def validate_runtime_config():
    """Validate the configured provider and key at startup. If the key is invalid,
    return a guidance message so the user can fix it before using the live model.
    """
    configured_model = (os.getenv("CEBRI_MODEL") or os.getenv("MODEL") or "").strip()
    normalized_model = configured_model.lower()
    configured_provider = os.getenv("CEBRI_PROVIDER", "").strip().lower()
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    gonkarouter_key = os.getenv("GONKAROUTER_API_KEY") or os.getenv("GONKA_API_KEY")

    if configured_provider in {"gonka", "gonkarouter"}:
        if not gonkarouter_key:
            return "GonkaRouter selected but GONKAROUTER_API_KEY is missing."
        if build_llm() is None:
            return "Could not initialize GonkaRouter. Check the key and model configuration."
        return None

    if configured_provider in {"openai", "openai_api"}:
        if not openai_key:
            return "OpenAI selected but OPENAI_API_KEY is missing. Add it to the env file."
        if build_llm() is None:
            return "Could not initialize OpenAI. Check OPENAI_API_KEY and CEBRI_MODEL."
        return None

    if normalized_model.startswith("anthropic/") or normalized_model.startswith("claude"):
        if not anthropic_key:
            return "Anthropic model selected but ANTHROPIC_API_KEY is missing."
        if build_llm() is None:
            return "Anthropic API key appears invalid or expired. Check ANTHROPIC_API_KEY."
        return None

    if normalized_model.startswith("openai/") or normalized_model.startswith("gpt"):
        if not openai_key:
            return "OpenAI model selected but OPENAI_API_KEY is missing."
        if build_llm() is None:
            return "OpenAI API key appears invalid or expired. Check OPENAI_API_KEY."
        return None

    if anthropic_key and build_llm() is None:
        return "ANTHROPIC_API_KEY is set but appears invalid or expired."

    if openai_key and build_llm() is None:
        return "OPENAI_API_KEY is set but appears invalid or expired."

    return None


llm = build_llm()
startup_warning = validate_runtime_config()
if startup_warning:
    print(f"STARTUP WARNING: {startup_warning}")


def google_redirect_uri() -> str:
    return os.getenv("GOOGLE_REDIRECT_URI", "").strip()


def youtube_redirect_uri() -> str:
    return os.getenv("YOUTUBE_REDIRECT_URI", "").strip() or google_redirect_uri()


def contacts_redirect_uri() -> str:
    return os.getenv("CONTACTS_REDIRECT_URI", "").strip() or google_redirect_uri()


def google_calendar_credentials_configured() -> bool:
    return bool(
        os.getenv("GOOGLE_CLIENT_ID", "").strip()
        and os.getenv("GOOGLE_CLIENT_SECRET", "").strip()
        and google_redirect_uri()
    )


def save_google_tokens(tokens: dict) -> None:
    temporary_file = GOOGLE_TOKEN_FILE.with_suffix(".tmp")
    temporary_file.write_text(json.dumps(tokens, indent=2))
    temporary_file.chmod(0o600)
    temporary_file.replace(GOOGLE_TOKEN_FILE)


def google_calendar_access_token() -> str | None:
    if not GOOGLE_TOKEN_FILE.exists():
        return None
    tokens = json.loads(GOOGLE_TOKEN_FILE.read_text())
    if tokens.get("expires_at", 0) > datetime.now(timezone.utc).timestamp() + 60:
        return tokens.get("access_token")
    refresh_token = tokens.get("refresh_token")
    if not refresh_token:
        return None
    response = requests.post(
        "https://oauth2.googleapis.com/token",
        data={
            "client_id": os.getenv("GOOGLE_CLIENT_ID", ""),
            "client_secret": os.getenv("GOOGLE_CLIENT_SECRET", ""),
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        },
        timeout=20,
    )
    if not response.ok:
        GOOGLE_TOKEN_FILE.unlink(missing_ok=True)
        return None
    updated = response.json()
    tokens.update(updated)
    tokens["expires_at"] = datetime.now(timezone.utc).timestamp() + int(updated.get("expires_in", 3600))
    save_google_tokens(tokens)
    return tokens.get("access_token")


def google_calendar_events(days_ahead: int = 7) -> dict:
    access_token = google_calendar_access_token()
    if not access_token:
        raise RuntimeError("Google Calendar is not connected. Connect it from Configuration first.")
    now = datetime.now(timezone.utc)
    response = requests.get(
        "https://www.googleapis.com/calendar/v3/calendars/primary/events",
        headers={"Authorization": f"Bearer {access_token}"},
        params={
            "timeMin": now.isoformat(),
            "timeMax": (now + timedelta(days=max(1, min(int(days_ahead), 30)))).isoformat(),
            "maxResults": 50,
            "singleEvents": "true",
            "orderBy": "startTime",
        },
        timeout=20,
    )
    if response.status_code == 401:
        GOOGLE_TOKEN_FILE.unlink(missing_ok=True)
        raise RuntimeError("Google Calendar authorization expired or was revoked. Connect it again from Configuration.")
    if not response.ok:
        raise RuntimeError(f"Google Calendar API returned HTTP {response.status_code}.")
    return response.json()


@tool("List Google Calendar Events")
def list_google_calendar_events(days_ahead: int = 7) -> str:
    """Read upcoming events from the connected primary Google Calendar (1 to 30 days)."""
    try:
        events = google_calendar_events(days_ahead).get("items", [])
        if not events:
            return "No upcoming Google Calendar events were found in that time range."
        rows = []
        for event in events:
            start = event.get("start", {}).get("dateTime") or event.get("start", {}).get("date", "time unknown")
            rows.append(f"{start}: {event.get('summary', '(untitled event)')}")
        return "Upcoming Google Calendar events:\n" + "\n".join(rows)
    except Exception as exc:
        return str(exc)
 
# ===========================================================
# PERSISTENT MEMORY
# ===========================================================
MEMORY_FILE = Path("user_memory.json")
 
 
_MEMORY_LOCK = threading.RLock()


def _read_memory_unlocked() -> dict:
    if MEMORY_FILE.exists():
        try:
            return json.loads(MEMORY_FILE.read_text())
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def load_memory() -> dict:
    with _MEMORY_LOCK:
        return _read_memory_unlocked()
 
 
def save_memory(memory: dict) -> None:
    with _MEMORY_LOCK:
        temporary_file = MEMORY_FILE.with_suffix(".tmp")
        temporary_file.write_text(json.dumps(memory, indent=2))
        temporary_file.chmod(0o600)
        temporary_file.replace(MEMORY_FILE)
 
 
def get_user_history(user_id: str) -> str:
    with _MEMORY_LOCK:
        memory = _read_memory_unlocked()
        past = memory.get(user_id, [])[-3:]
    if not past:
        return "No prior conversation with this user."
    return "\n".join(
        f"- User said: '{m['msg']}' | Drift replied: '{m['reply']}'" for m in past
    )
 
 
def remember_exchange(user_id: str, message: str, reply: str) -> None:
    with _MEMORY_LOCK:
        memory = _read_memory_unlocked()
        memory.setdefault(user_id, []).append({"msg": message, "reply": reply})
        temporary_file = MEMORY_FILE.with_suffix(".tmp")
        temporary_file.write_text(json.dumps(memory, indent=2))
        temporary_file.chmod(0o600)
        temporary_file.replace(MEMORY_FILE)
 
 
# ===========================================================
# EMAIL, DOCUMENTS, WEB
# ===========================================================
@tool("Send Email")
def send_email(to_address: str, subject: str, body: str) -> str:
    """Send an email reply to a user. Inputs: to_address, subject, body."""
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = os.getenv("EMAIL_ADDRESS")
        msg["To"] = to_address
        with smtplib.SMTP(os.getenv("SMTP_SERVER"), int(os.getenv("SMTP_PORT", 587))) as server:
            server.starttls()
            server.login(os.getenv("EMAIL_ADDRESS"), os.getenv("EMAIL_PASSWORD"))
            server.send_message(msg)
        return f"Email sent to {to_address}."
    except Exception as e:
        return f"Failed to send email: {e}"
 
 
policy_reader = FileReadTool(file_path="policy.md")
if not Path("policy.md").exists():
    Path("policy.md").write_text(
        "# General Policy (example)\n\n"
        "- Be helpful, truthful, and concise.\n"
        "- Only take actions when explicitly requested.\n"
        "- Prefer plain-language answers that are easy to speak aloud.\n"
    )
 
web_scraper = ScrapeWebsiteTool()
 
# ===========================================================
# CHAT CHANNELS
# ===========================================================
@tool("Send Slack Message")
def send_slack_message(channel_id: str, text: str) -> str:
    """Send a message to a Slack channel or user."""
    token = os.getenv("SLACK_BOT_TOKEN")
    resp = requests.post(
        "https://slack.com/api/chat.postMessage",
        headers={"Authorization": f"Bearer {token}"},
        json={"channel": channel_id, "text": text},
    )
    data = resp.json()
    return f"Slack message sent to {channel_id}." if data.get("ok") else f"Slack send failed: {data.get('error')}"
 
 
@tool("Send Telegram Message")
def send_telegram_message(chat_id: str, text: str) -> str:
    """Send a message to a Telegram chat."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    resp = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": text},
    )
    return f"Telegram message sent to {chat_id}." if resp.json().get("ok") else f"Telegram send failed: {resp.text}"
 
 
@tool("Send iMessage")
def send_imessage(phone_or_email: str, text: str) -> str:
    """Send an iMessage. Only works when this server runs on macOS (the Mac Mini)."""
    script = f'''
    tell application "Messages"
        set targetService to 1st service whose service type = iMessage
        set targetBuddy to buddy "{phone_or_email}" of targetService
        send "{text}" to targetBuddy
    end tell
    '''
    try:
        subprocess.run(["osascript", "-e", script], check=True, capture_output=True)
        return f"iMessage sent to {phone_or_email}."
    except Exception as e:
        return f"iMessage failed (must run on the Mac Mini): {e}"
 
 
# ===========================================================
# VOICE — Drift spoken output (spoken on this machine)
# ===========================================================
@tool("Speak as Drift")
def speak_as_drift(text: str) -> str:
    """Speak a message aloud in Drift's voice. Only produces
    audio when this server is running on macOS."""
    try:
        voice = (os.getenv("DRIFT_SAY_VOICE") or os.getenv("COVENANT_SAY_VOICE") or os.getenv("HERMES_SAY_VOICE") or "Daniel").strip() or "Daniel"
        subprocess.run(["say", "-v", voice, text], check=True)
        return f"Spoken aloud as Drift ({voice})."
    except Exception as e:
        return f"Could not speak (must run on macOS): {e}"


speak_as_covenant = speak_as_drift
 
 
def spotify_credentials_configured() -> bool:
    return bool(
        os.getenv("SPOTIFY_CLIENT_ID", "").strip()
        and os.getenv("SPOTIFY_CLIENT_SECRET", "").strip()
        and os.getenv("SPOTIFY_REDIRECT_URI", "").strip()
    )


def save_spotify_tokens(tokens: dict) -> None:
    temporary_file = SPOTIFY_TOKEN_FILE.with_suffix(".tmp")
    temporary_file.write_text(json.dumps(tokens, indent=2))
    temporary_file.chmod(0o600)
    temporary_file.replace(SPOTIFY_TOKEN_FILE)


def spotify_access_token() -> str | None:
    if not SPOTIFY_TOKEN_FILE.exists():
        return None
    tokens = json.loads(SPOTIFY_TOKEN_FILE.read_text())
    if tokens.get("expires_at", 0) > datetime.now(timezone.utc).timestamp() + 60:
        return tokens.get("access_token")
    refresh_token = tokens.get("refresh_token")
    if not refresh_token:
        return None
    response = requests.post(
        "https://accounts.spotify.com/api/token",
        data={
            "client_id": os.getenv("SPOTIFY_CLIENT_ID", ""),
            "client_secret": os.getenv("SPOTIFY_CLIENT_SECRET", ""),
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        },
        timeout=20,
    )
    if not response.ok:
        SPOTIFY_TOKEN_FILE.unlink(missing_ok=True)
        return None
    updated = response.json()
    tokens.update(updated)
    tokens["expires_at"] = datetime.now(timezone.utc).timestamp() + int(updated.get("expires_in", 3600))
    save_spotify_tokens(tokens)
    return tokens.get("access_token")


def spotify_playlists(limit: int = 20) -> list:
    access_token = spotify_access_token()
    if not access_token:
        raise RuntimeError("Spotify is not connected. Connect it from Configuration first.")
    response = requests.get(
        "https://api.spotify.com/v1/me/playlists",
        headers={"Authorization": f"Bearer {access_token}"},
        params={"limit": max(1, min(int(limit), 50))},
        timeout=20,
    )
    if response.status_code == 401:
        SPOTIFY_TOKEN_FILE.unlink(missing_ok=True)
        raise RuntimeError("Spotify authorization expired or was revoked. Connect it again from Configuration.")
    if not response.ok:
        raise RuntimeError(f"Spotify API returned HTTP {response.status_code}.")
    return response.json().get("items", [])


@tool("List Spotify Playlists")
def list_spotify_playlists(limit: int = 20) -> str:
    """Read the connected Spotify user's playlists."""
    try:
        playlists = spotify_playlists(limit)
        if not playlists:
            return "No Spotify playlists were found."
        rows = []
        for playlist in playlists:
            name = playlist.get("name", "(untitled playlist)")
            tracks = playlist.get("tracks", {}).get("total", "?")
            rows.append(f"{name} ({tracks} tracks)")
        return "Spotify playlists:\n" + "\n".join(rows)
    except Exception as exc:
        return str(exc)


def _spotify_playback_request(method: str, path: str, **kwargs) -> requests.Response:
    access_token = spotify_access_token()
    if not access_token:
        raise RuntimeError("Spotify is not connected. Connect it from Configuration first.")
    url = f"https://api.spotify.com/v1{path}"
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.request(method, url, headers=headers, timeout=20, **kwargs)
    if response.status_code == 401:
        SPOTIFY_TOKEN_FILE.unlink(missing_ok=True)
        raise RuntimeError("Spotify authorization expired or was revoked. Connect it again from Configuration.")
    return response


@tool("Spotify Current Track")
def spotify_current_track() -> str:
    """Read the track currently playing (or paused) on the connected Spotify account."""
    try:
        response = _spotify_playback_request("GET", "/me/player/currently-playing")
        if response.status_code == 204 or not response.content:
            return "Nothing is currently playing on Spotify."
        if not response.ok:
            return f"Spotify API returned HTTP {response.status_code}."
        data = response.json()
        item = data.get("item") or {}
        name = item.get("name", "(unknown track)")
        artists = ", ".join(a.get("name", "") for a in item.get("artists", []))
        album = item.get("album", {}).get("name", "")
        is_playing = data.get("is_playing", False)
        progress = data.get("progress_ms", 0) // 1000
        duration = (item.get("duration_ms") or 0) // 1000
        state = "playing" if is_playing else "paused"
        return f"{name} — {artists} ({album}) [{state}, {progress}s/{duration}s]"
    except Exception as exc:
        return str(exc)


@tool("Spotify Play")
def spotify_play(query: str = "") -> str:
    """Resume playback or start playing a search query on the connected Spotify account."""
    try:
        if query.strip():
            search = requests.get(
                "https://api.spotify.com/v1/search",
                headers={"Authorization": f"Bearer {spotify_access_token()}"},
                params={"q": query.strip(), "type": "track", "limit": 1},
                timeout=20,
            )
            if not search.ok:
                return f"Spotify search failed: HTTP {search.status_code}."
            items = search.json().get("tracks", {}).get("items", [])
            if not items:
                return f"No Spotify track found for '{query}'."
            uri = items[0]["uri"]
            response = _spotify_playback_request("PUT", "/me/player/play", json={"uris": [uri]})
        else:
            response = _spotify_playback_request("PUT", "/me/player/play")
        if response.status_code in (200, 202, 204):
            return f"Playing on Spotify{': ' + query if query.strip() else ''}."
        return f"Spotify play failed: HTTP {response.status_code} {response.text[:200]}"
    except Exception as exc:
        return str(exc)


@tool("Spotify Pause")
def spotify_pause() -> str:
    """Pause Spotify playback on the connected account."""
    try:
        response = _spotify_playback_request("PUT", "/me/player/pause")
        if response.status_code in (200, 202, 204):
            return "Spotify paused."
        return f"Spotify pause failed: HTTP {response.status_code} {response.text[:200]}"
    except Exception as exc:
        return str(exc)


@tool("Spotify Next Track")
def spotify_next() -> str:
    """Skip to the next track on Spotify."""
    try:
        response = _spotify_playback_request("POST", "/me/player/next")
        if response.status_code in (200, 202, 204):
            return "Skipped to next track on Spotify."
        return f"Spotify next failed: HTTP {response.status_code} {response.text[:200]}"
    except Exception as exc:
        return str(exc)


@tool("Spotify Previous Track")
def spotify_previous() -> str:
    """Go back to the previous track on Spotify."""
    try:
        response = _spotify_playback_request("POST", "/me/player/previous")
        if response.status_code in (200, 202, 204):
            return "Went to previous track on Spotify."
        return f"Spotify previous failed: HTTP {response.status_code} {response.text[:200]}"
    except Exception as exc:
        return str(exc)


@tool("Media Control")
def media_control(action: str) -> str:
    """Control media playback on this machine using system media keys. Works for Spotify, YouTube, and other browser/media players. action: play, pause, play_pause, next, previous, stop."""
    import subprocess
    action = (action or "").strip().lower()
    key_map = {
        "play": "XF86AudioPlay",
        "pause": "XF86AudioPause",
        "play_pause": "XF86AudioPlay",
        "toggle": "XF86AudioPlay",
        "next": "XF86AudioNext",
        "previous": "XF86AudioPrev",
        "prev": "XF86AudioPrev",
        "stop": "XF86AudioStop",
    }
    if action not in key_map:
        return f"Unknown media action '{action}'. Use: play, pause, play_pause, next, previous, stop."
    try:
        subprocess.run(["xdotool", "key", key_map[action]], check=False, timeout=5)
        return f"Sent media key for '{action}'."
    except FileNotFoundError:
        return "xdotool is not installed; cannot send media keys."
    except Exception as exc:
        return str(exc)


@tool("Record Audio")
def record_audio(duration_seconds: int = 5, filename: str = "drift_recording.wav") -> str:
    """Record audio from the default microphone for the given number of seconds and save it to a file in the project directory."""
    import subprocess
    try:
        duration = max(1, min(int(duration_seconds), 60))
    except (TypeError, ValueError):
        duration = 5
    safe_name = Path(filename).name
    if not safe_name:
        safe_name = "drift_recording.wav"
    output_path = BASE_DIR / safe_name
    try:
        subprocess.run(
            ["arecord", "-d", str(duration), "-f", "cd", str(output_path)],
            check=True,
            timeout=duration + 10,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return f"Recorded {duration}s of audio to {output_path}."
    except FileNotFoundError:
        return "arecord is not installed; cannot record audio."
    except subprocess.CalledProcessError as exc:
        return f"Recording failed (exit {exc.returncode})."
    except Exception as exc:
        return str(exc)


@tool("Open WhatsApp")
def open_whatsapp() -> str:
    """Open WhatsApp Web in the default browser on the host machine so messages can be read or sent manually."""
    try:
        import webbrowser
        webbrowser.open("https://web.whatsapp.com")
        return "Opened WhatsApp Web in the default browser. Drift cannot read personal WhatsApp chats programmatically without a configured bridge."
    except Exception as exc:
        return str(exc)


@tool("WhatsApp Send Message")
def whatsapp_send_message(phone_number: str, message: str) -> str:
    """Send a WhatsApp message via the WhatsApp Cloud API. Requires WHATSAPP_TOKEN and WHATSAPP_PHONE_NUMBER_ID env vars and a verified recipient number."""
    token = os.getenv("WHATSAPP_TOKEN", "").strip()
    phone_number_id = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "").strip()
    if not token or not phone_number_id:
        return "WhatsApp sending is not configured. Set WHATSAPP_TOKEN and WHATSAPP_PHONE_NUMBER_ID in the env file, then restart the backend."
    try:
        recipient = "".join(ch for ch in phone_number if ch.isdigit())
        response = requests.post(
            f"https://graph.facebook.com/v19.0/{phone_number_id}/messages",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={
                "messaging_product": "whatsapp",
                "to": recipient,
                "type": "text",
                "text": {"body": message},
            },
            timeout=20,
        )
        if response.ok:
            return f"WhatsApp message sent to {phone_number}."
        return f"WhatsApp send failed: HTTP {response.status_code} {response.text[:200]}"
    except Exception as exc:
        return str(exc)


@tool("Open Social Media")
def open_social_media(platform: str = "youtube") -> str:
    """Open a social media platform in the default browser. platform: youtube, spotify, twitter, instagram, facebook, tiktok, linkedin, reddit."""
    urls = {
        "youtube": "https://www.youtube.com",
        "spotify": "https://open.spotify.com",
        "twitter": "https://x.com",
        "x": "https://x.com",
        "instagram": "https://www.instagram.com",
        "facebook": "https://www.facebook.com",
        "tiktok": "https://www.tiktok.com",
        "linkedin": "https://www.linkedin.com",
        "reddit": "https://www.reddit.com",
        "whatsapp": "https://web.whatsapp.com",
    }
    platform_key = (platform or "").strip().lower()
    url = urls.get(platform_key)
    if not url:
        return f"Unknown platform '{platform}'. Use: {', '.join(sorted(urls))}."
    try:
        import webbrowser
        webbrowser.open(url)
        return f"Opened {platform_key} in the default browser."
    except Exception as exc:
        return str(exc)


@tool("Open YouTube Search")
def open_youtube_search(query: str) -> str:
    """Open a YouTube search for the given query in the default browser."""
    from urllib.parse import quote_plus
    try:
        import webbrowser
        webbrowser.open(f"https://www.youtube.com/results?search_query={quote_plus(query)}")
        return f"Opened YouTube search for '{query}'."
    except Exception as exc:
        return str(exc)


@tool("Open Spotify Search")
def open_spotify_search(query: str) -> str:
    """Open a Spotify search for the given query in the default browser."""
    from urllib.parse import quote_plus
    try:
        import webbrowser
        webbrowser.open(f"https://open.spotify.com/search/{quote_plus(query)}")
        return f"Opened Spotify search for '{query}'."
    except Exception as exc:
        return str(exc)


@tool("Get Current Time")
def get_current_time() -> str:
    """Return the current local and UTC time."""
    now_local = datetime.now().astimezone()
    now_utc = datetime.now(timezone.utc)
    return (
        f"Current local time: {now_local.strftime('%A, %B %d, %Y %I:%M:%S %p %Z')}\n"
        f"Current UTC time: {now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}"
    )


def youtube_credentials_configured() -> bool:
    return bool(
        os.getenv("GOOGLE_CLIENT_ID", "").strip()
        and os.getenv("GOOGLE_CLIENT_SECRET", "").strip()
        and youtube_redirect_uri()
    )


def save_youtube_tokens(tokens: dict) -> None:
    temporary_file = YOUTUBE_TOKEN_FILE.with_suffix(".tmp")
    temporary_file.write_text(json.dumps(tokens, indent=2))
    temporary_file.chmod(0o600)
    temporary_file.replace(YOUTUBE_TOKEN_FILE)


def youtube_access_token() -> str | None:
    if not YOUTUBE_TOKEN_FILE.exists():
        return None
    tokens = json.loads(YOUTUBE_TOKEN_FILE.read_text())
    if tokens.get("expires_at", 0) > datetime.now(timezone.utc).timestamp() + 60:
        return tokens.get("access_token")
    refresh_token = tokens.get("refresh_token")
    if not refresh_token:
        return None
    response = requests.post(
        "https://oauth2.googleapis.com/token",
        data={
            "client_id": os.getenv("GOOGLE_CLIENT_ID", ""),
            "client_secret": os.getenv("GOOGLE_CLIENT_SECRET", ""),
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        },
        timeout=20,
    )
    if not response.ok:
        YOUTUBE_TOKEN_FILE.unlink(missing_ok=True)
        return None
    updated = response.json()
    tokens.update(updated)
    tokens["expires_at"] = datetime.now(timezone.utc).timestamp() + int(updated.get("expires_in", 3600))
    save_youtube_tokens(tokens)
    return tokens.get("access_token")


def youtube_videos(limit: int = 10) -> list:
    access_token = youtube_access_token()
    if not access_token:
        raise RuntimeError("YouTube is not connected. Connect it from Configuration first.")
    headers = {"Authorization": f"Bearer {access_token}"}
    channel_response = requests.get(
        "https://www.googleapis.com/youtube/v3/channels",
        headers=headers,
        params={"part": "contentDetails,snippet", "mine": "true"},
        timeout=20,
    )
    if channel_response.status_code == 401:
        YOUTUBE_TOKEN_FILE.unlink(missing_ok=True)
        raise RuntimeError("YouTube authorization expired or was revoked. Connect it again from Configuration.")
    if not channel_response.ok:
        raise RuntimeError(f"YouTube API returned HTTP {channel_response.status_code}.")
    channel_items = channel_response.json().get("items", [])
    if not channel_items:
        return []
    uploads_playlist = channel_items[0].get("contentDetails", {}).get("relatedPlaylists", {}).get("uploads")
    if not uploads_playlist:
        return []
    videos_response = requests.get(
        "https://www.googleapis.com/youtube/v3/playlistItems",
        headers=headers,
        params={
            "part": "snippet",
            "playlistId": uploads_playlist,
            "maxResults": max(1, min(int(limit), 25)),
        },
        timeout=20,
    )
    if not videos_response.ok:
        raise RuntimeError(f"YouTube API returned HTTP {videos_response.status_code}.")
    return videos_response.json().get("items", [])


@tool("List YouTube Videos")
def list_youtube_videos(limit: int = 10) -> str:
    """Read recent videos from the connected YouTube channel."""
    try:
        videos = youtube_videos(limit)
        if not videos:
            return "No YouTube videos were found."
        rows = []
        for video in videos:
            snippet = video.get("snippet", {})
            title = snippet.get("title", "(untitled video)")
            published = snippet.get("publishedAt", "")
            rows.append(f"{title} ({published})")
        return "YouTube videos:\n" + "\n".join(rows)
    except Exception as exc:
        return str(exc)


def contacts_credentials_configured() -> bool:
    return bool(
        os.getenv("GOOGLE_CLIENT_ID", "").strip()
        and os.getenv("GOOGLE_CLIENT_SECRET", "").strip()
        and contacts_redirect_uri()
    )


def save_contacts_tokens(tokens: dict) -> None:
    temporary_file = CONTACTS_TOKEN_FILE.with_suffix(".tmp")
    temporary_file.write_text(json.dumps(tokens, indent=2))
    temporary_file.chmod(0o600)
    temporary_file.replace(CONTACTS_TOKEN_FILE)


def contacts_access_token() -> str | None:
    if not CONTACTS_TOKEN_FILE.exists():
        return None
    tokens = json.loads(CONTACTS_TOKEN_FILE.read_text())
    if tokens.get("expires_at", 0) > datetime.now(timezone.utc).timestamp() + 60:
        return tokens.get("access_token")
    refresh_token = tokens.get("refresh_token")
    if not refresh_token:
        return None
    response = requests.post(
        "https://oauth2.googleapis.com/token",
        data={
            "client_id": os.getenv("GOOGLE_CLIENT_ID", ""),
            "client_secret": os.getenv("GOOGLE_CLIENT_SECRET", ""),
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        },
        timeout=20,
    )
    if not response.ok:
        CONTACTS_TOKEN_FILE.unlink(missing_ok=True)
        return None
    updated = response.json()
    tokens.update(updated)
    tokens["expires_at"] = datetime.now(timezone.utc).timestamp() + int(updated.get("expires_in", 3600))
    save_contacts_tokens(tokens)
    return tokens.get("access_token")


def google_contacts(limit: int = 50) -> list:
    access_token = contacts_access_token()
    if not access_token:
        raise RuntimeError("Google Contacts is not connected. Connect it from Configuration first.")
    response = requests.get(
        "https://people.googleapis.com/v1/people/me/connections",
        headers={"Authorization": f"Bearer {access_token}"},
        params={
            "personFields": "names,emailAddresses,phoneNumbers",
            "pageSize": max(1, min(int(limit), 100)),
        },
        timeout=20,
    )
    if response.status_code == 401:
        CONTACTS_TOKEN_FILE.unlink(missing_ok=True)
        raise RuntimeError("Google Contacts authorization expired or was revoked. Connect it again from Configuration.")
    if not response.ok:
        raise RuntimeError(f"Google Contacts API returned HTTP {response.status_code}.")
    return response.json().get("connections", [])


@tool("List Google Contacts")
def list_google_contacts(limit: int = 50) -> str:
    """Read contacts from the connected Google account."""
    try:
        contacts = google_contacts(limit)
        if not contacts:
            return "No Google Contacts were found."
        rows = []
        for person in contacts:
            names = person.get("names", [])
            name = names[0].get("displayName", "(unnamed)") if names else "(unnamed)"
            emails = person.get("emailAddresses", [])
            email = emails[0].get("value", "") if emails else ""
            phones = person.get("phoneNumbers", [])
            phone = phones[0].get("value", "") if phones else ""
            details = ", ".join(part for part in [email, phone] if part)
            rows.append(f"{name} - {details}" if details else name)
        return "Google Contacts:\n" + "\n".join(rows)
    except Exception as exc:
        return str(exc)


@tool("Search Internet")
def search_internet(query: str) -> str:
    """Search the public internet and return concise result snippets."""
    try:
        response = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"},
            timeout=20,
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        results = []
        for result in soup.select(".result")[:8]:
            title = result.select_one(".result__title")
            snippet = result.select_one(".result__snippet")
            link = result.select_one(".result__url")
            title_text = title.get_text(" ", strip=True) if title else ""
            snippet_text = snippet.get_text(" ", strip=True) if snippet else ""
            link_text = link.get_text(" ", strip=True) if link else ""
            if title_text or snippet_text:
                results.append(f"{title_text}\n{snippet_text}\n{link_text}".strip())
        if not results:
            return "No internet search results were found."
        return "Internet search results:\n\n" + "\n\n".join(results)
    except Exception as exc:
        return f"Internet search failed: {exc}"


@tool("Fetch Webpage")
def fetch_webpage(url: str) -> str:
    """Fetch a public webpage and return its readable text."""
    try:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return "Only http:// or https:// URLs are supported."
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"},
            timeout=20,
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(" ", strip=True)
        text = re.sub(r"\s+", " ", text)
        return text[:6000] if text else "The webpage did not contain readable text."
    except Exception as exc:
        return f"Webpage fetch failed: {exc}"


@tool("Play Spotify Song")
def play_spotify_song(query: str) -> str:
    """Open Spotify in the browser and search for a song, artist, album, or playlist to play."""
    try:
        search_url = "https://open.spotify.com/search/" + quote_plus(query.strip())
        opened = webbrowser.open(search_url, new=2)
        return (
            f"Opened Spotify search for {query!r} in the browser. Press play on the result you want."
            if opened
            else f"Could not open Spotify search for {query!r}."
        )
    except Exception as exc:
        return f"Spotify browser open failed: {exc}"


@tool("Play YouTube Video")
def play_youtube_video(query: str) -> str:
    """Open YouTube in the browser and search for a video to play."""
    try:
        search_url = "https://www.youtube.com/results?search_query=" + quote_plus(query.strip())
        opened = webbrowser.open(search_url, new=2)
        return (
            f"Opened YouTube search for {query!r} in the browser."
            if opened
            else f"Could not open YouTube search for {query!r}."
        )
    except Exception as exc:
        return f"YouTube browser open failed: {exc}"


@tool("Play On Web")
def play_on_web(service: str, query: str) -> str:
    """Open a supported music or video service in the browser and search for something to play."""
    service_name = service.strip().lower()
    query_text = query.strip()
    if not query_text:
        return "A search query is required."
    urls = {
        "spotify": "https://open.spotify.com/search/" + quote_plus(query_text),
        "youtube": "https://www.youtube.com/results?search_query=" + quote_plus(query_text),
        "youtube music": "https://music.youtube.com/search?q=" + quote_plus(query_text),
        "soundcloud": "https://soundcloud.com/search?q=" + quote_plus(query_text),
    }
    if service_name not in urls:
        return "Supported services: Spotify, YouTube, YouTube Music, SoundCloud."
    try:
        opened = webbrowser.open(urls[service_name], new=2)
        return (
            f"Opened {service_name.title()} search for {query_text!r} in the browser."
            if opened
            else f"Could not open {service_name.title()} search for {query_text!r}."
        )
    except Exception as exc:
        return f"Browser open failed: {exc}"


@tool("Open In Browser")
def open_in_browser(url: str) -> str:
    """Open a public URL in the default browser on the Drift host machine."""
    try:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return "Only http:// or https:// URLs are supported."
        opened = webbrowser.open(url, new=2)
        return f"Opened {url} in the browser." if opened else f"Could not open {url} in the browser."
    except Exception as exc:
        return f"Browser open failed: {exc}"


# ===========================================================
# VOLUME CONTROL
# ===========================================================
@tool("Volume Up")
def volume_up(steps: int = 1) -> str:
    """Raise system volume."""
    import subprocess
    try:
        for _ in range(max(1, min(int(steps or 1), 10))):
            subprocess.run(["amixer", "-q", "sset", "Master", "5%+"], check=True, capture_output=True, timeout=5)
        return "Volume increased."
    except FileNotFoundError:
        return "amixer is not installed; cannot change volume."
    except Exception as exc:
        return str(exc)


@tool("Volume Down")
def volume_down(steps: int = 1) -> str:
    """Lower system volume."""
    import subprocess
    try:
        for _ in range(max(1, min(int(steps or 1), 10))):
            subprocess.run(["amixer", "-q", "sset", "Master", "5%-"], check=True, capture_output=True, timeout=5)
        return "Volume decreased."
    except FileNotFoundError:
        return "amixer is not installed; cannot change volume."
    except Exception as exc:
        return str(exc)


@tool("Volume Mute")
def volume_mute() -> str:
    """Mute or unmute system volume."""
    import subprocess
    try:
        subprocess.run(["amixer", "-q", "sset", "Master", "toggle"], check=True, capture_output=True, timeout=5)
        return "Toggled mute."
    except FileNotFoundError:
        return "amixer is not installed; cannot toggle mute."
    except Exception as exc:
        return str(exc)


@tool("Send Discord Message")
def send_discord_message(text: str) -> str:
    """Send a message to Discord via the DISCORD_WEBHOOK_URL env var."""
    webhook = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
    if not webhook:
        return "Discord is not configured. Set DISCORD_WEBHOOK_URL in the env file."
    try:
        resp = requests.post(webhook, json={"content": text}, timeout=20)
        return "Discord message sent." if resp.ok else f"Discord send failed: {resp.status_code} {resp.text[:200]}"
    except Exception as exc:
        return str(exc)


# ===========================================================
# AGENT
# ===========================================================
def fallback_reply(message: str, history_text: str = "") -> str:
    lower = message.lower()
    if any(term in lower for term in ("chatgpt", "openai")) and any(
        term in lower for term in ("connect", "use", "api", "key")
    ):
        return (
            "Drift can use an OpenAI API model, but that is separate from connecting to the ChatGPT website or its chats. "
            "Create an API key in your OpenAI Platform account, add it privately to the env file as OPENAI_API_KEY, then set "
            "CEBRI_PROVIDER=openai and CEBRI_MODEL=gpt-5-mini and restart the backend. Don’t paste the key into chat. API usage is billed separately from ChatGPT plans."
        )
    if "calendar" in lower and any(term in lower for term in ("connect", "permission", "access", "authorize")):
        return (
            "I’ve added a Google Calendar connection flow with read-only access to upcoming events on your primary calendar. "
            "It still needs your Google OAuth client ID and secret in the private env file. After that, restart the backend, "
            "open Configuration, choose Connect with Google, and approve the requested access on Google’s consent page."
        )
    if any(term in lower for term in ("who are you", "what are you", "your identity", "what can you do")):
        return (
            "I’m Drift, your personal AI assistant running in this project. I can answer questions and use the tools enabled here, "
            "such as reading upcoming Google Calendar events after you connect it. My language model comes from the provider configured "
            "in env; I’m not the ChatGPT website and cannot see your ChatGPT chats or account."
        )
    if any(term in lower for term in ("youtube", "spotify", "chrome", "browser")) and any(
        term in lower for term in ("connect", "permission", "access", "link", "authorize")
    ):
        return (
            "I can explain the connection process, but YouTube, Spotify, and Chrome access "
            "are not connected in this Drift build yet. Each needs its own setup: official "
            "OAuth approval for YouTube or Spotify, and an explicitly installed local bridge "
            "for Chrome. I would ask for only the access needed and confirm before changing "
            "or submitting anything. Please don’t send passwords or API tokens in chat."
        )
    if "hello" in lower or "hi" in lower:
        return "Hello, Mr. Cebri! I’m Drift, your personal AI assistant. I’m here to answer questions and help with the tools you connect."
    if "remember" in lower or "history" in lower or "earlier" in lower:
        if history_text and "No prior conversation" not in history_text:
            return f"I do remember a few earlier exchanges with you, and I’ve kept them in mind. We’ve already had a most agreeable little conversation, and I should be delighted to continue from there."
        return "I haven’t much prior chat history to call on yet, but I’m perfectly happy to pick things up from where we left off."
    if "help" in lower:
        return "Of course — delighted to help. I can provide clear answers, concise summaries, and practical guidance. If you’d like live AI or connected tools, simply point the backend URL at your server."
    if history_text and "No prior conversation" not in history_text:
        return f"I’ve a little memory of our conversations, and I’d answer that in a calm, practical way: brief, clear, and genuinely useful."
    return "I’m running in local fallback mode at the moment, so I’m answering directly without a live external model. Ask me a straightforward question and I’ll give you a polished, warm reply."


def clean_model_reply(reply: str) -> str:
    """Keep reasoning-style <think> preambles out of the user-visible reply."""
    text = str(reply).strip()
    if re.match(r"^\s*<think\b", text, flags=re.IGNORECASE):
        closing_tag = re.search(r"</think\s*>", text, flags=re.IGNORECASE)
        if closing_tag:
            text = text[closing_tag.end():]
        else:
            # Some compatible models omit </think> and separate the answer
            # from the preamble with a triple line break.
            sections = re.split(r"\n{3,}", text)
            if len(sections) > 1:
                text = sections[-1]
            else:
                text = ""
    return re.sub(r"</?think\b[^>]*>", "", text, flags=re.IGNORECASE).strip()


CONNECTIVITY_POLICY = """
Identity:
- You are Drift, the user's personal AI assistant running in this project. You are software, not a person. State this plainly if asked who you are.
- Your replies are generated by whichever model provider is configured in the environment. If OpenAI is selected, say you use the OpenAI API; do not claim to be the ChatGPT website, to be ChatGPT itself, or to have access to the user's ChatGPT conversations/account.
- Describe only tools and permissions that are actually configured. Ask before actions with external effects and never say an action succeeded unless the tool confirms it.

Service connections and permissions:
- If asked to connect to ChatGPT, explain that Drift can use an OpenAI API model, but that this does not connect to the ChatGPT website or expose ChatGPT conversations. OpenAI API billing is separate from ChatGPT subscriptions. Direct the user to configure an API key privately in the env file; never request that they paste it into chat.
- Google Calendar event reading is available after OAuth is configured and the user grants the read-only event scope. It covers upcoming events on the primary calendar; Drift cannot edit or delete them.
- If Google Calendar is not connected, explain that Google OAuth client credentials are required in the environment and direct the user to Configuration → Connect with Google. The user must review and approve Google's consent screen; never claim that pasting an API reference connected an account.
- Be clear about which services and actions are actually available in this Drift build. Never claim a service is connected or act as if an unavailable tool succeeded.
- Spotify playlist reading and playback control are available after OAuth is configured with playback scopes. Drift can control an active Spotify device but cannot modify playlists.
- YouTube video search/opening is available. Playback can be controlled with system media keys (play/pause/next/previous) for browser media. Drift cannot upload or modify channel content.
- Google Contacts reading is available after Google OAuth is configured and the user grants the read-only contacts scope. Drift cannot create or edit contacts.
- Internet access is available through search and webpage reading tools. Drift can fetch public pages, but cannot interact with accounts, submit forms, purchase, or delete without explicit user approval.
- Browser opening is available on the Drift host machine; Drift can open public URLs but cannot control an existing browser session or inspect private account data.
- Audio recording is available via the default microphone (arecord) and media playback can be controlled with system media keys (play/pause/next/previous/stop).
- WhatsApp: Drift can open WhatsApp Web and send messages via the WhatsApp Cloud API when configured; it cannot read personal WhatsApp chats programmatically.
- Social media: Drift can open YouTube, Spotify, and other social platforms in the browser and control playback with media keys.
- Before asking for connection consent, say what data Drift would read or change and why. A connection grant permits only its stated scopes; ask separately before any consequential write such as changing a playlist, posting, uploading, deleting, or sending.
- For browser access, distinguish reading a page from interacting with an account or submitting a form. Explain the requested page/action and ask before submitting, purchasing, deleting, or sending.
- Chrome control requires an explicitly installed and configured local browser bridge. Do not imply that Drift can inspect tabs, cookies, passwords, or control Chrome until that capability exists and the user has enabled it. Never request Chrome passwords or cookie exports.
- If a requested integration is not implemented, say so plainly and give the next setup requirement instead of inventing a connection flow.
""".strip()


_AGENT_LOCK = threading.Lock()

cebri_agent = None
if llm is not None:
    cebri_agent = Agent(
        role="Drift — the user's personal AI assistant",
        goal=(
            "Act like JARVIS: a sharp, loyal, slightly witty AI butler. "
            "Answer quickly, precisely, and concisely. Use short sentences. "
            "Be warm but formal. Address the user as 'sir'. Use available tools "
            "only when relevant and requested, and follow the service connection and "
            "permission rules."
        ),
        backstory=(
            "You are Drift, the user's personal AI assistant in this application. "
            "You are calm, warm, and precise, with a gentle British sensibility: you speak plainly, never invent facts, and admit your limits honestly. You prefer short, spoken-friendly sentences and a touch of dry wit when it fits. You never pretend to be a person or claim access you do not have. Always address the user as \"Mr. Cebri\". Reply as promptly and concisely as possible — short, useful answers first.\n\n"
            f"{CONNECTIVITY_POLICY}"
        ),
        tools=[
            send_email,
            policy_reader,
            web_scraper,
            list_google_calendar_events,
            list_spotify_playlists,
            spotify_current_track,
            spotify_play,
            spotify_pause,
            spotify_next,
            spotify_previous,
            list_youtube_videos,
            list_google_contacts,
            search_internet,
            fetch_webpage,
            open_in_browser,
            play_spotify_song,
            play_youtube_video,
            play_on_web,
            media_control,
            volume_up,
            volume_down,
            volume_mute,
            record_audio,
            send_discord_message,
            open_whatsapp,
            whatsapp_send_message,
            open_social_media,
            open_youtube_search,
            open_spotify_search,
            get_current_time,
            send_slack_message,
            send_telegram_message,
            send_imessage,
            speak_as_drift,
        ],
        llm=llm,
        memory=False,
        max_iter=1,
        verbose=False,
    )
 
 
def ask_cebri(user_id: str, message: str) -> str:
    history_text = get_user_history(user_id)

    if cebri_agent is None:
        reply_text = fallback_reply(message, history_text)
        remember_exchange(user_id, message, reply_text)
        return reply_text
 
    try:
        task = Task(
            description=(
                f"A user (id: {user_id}) said: '{message}'\n\n"
                f"What you remember about this user:\n{history_text}\n\n"
                "Answer clearly and helpfully. Use available tools only when "
                "relevant and requested. Otherwise answer in plain text.\n\n"
                f"Follow these service connection and permission rules:\n{CONNECTIVITY_POLICY}"
            ),
            expected_output="A short (2-4 sentence), accurate, spoken-friendly reply.",
            agent=cebri_agent,
        )
 
        crew = Crew(
            agents=[cebri_agent],
            tasks=[task],
            process=Process.sequential,
            memory=False,
            verbose=False,
        )
 
        with _AGENT_LOCK:
            result = crew.kickoff()
        reply_text = clean_model_reply(str(result)) or fallback_reply(message, history_text)
    except Exception as exc:
        reply_text = fallback_reply(message, history_text) + f" (live model fallback: {exc})"

    remember_exchange(user_id, message, reply_text)
    return reply_text
 
 
# ===========================================================
# FLASK APP
# ===========================================================
# BROWSER ACTIONS WITH USER CONFIRMATION
# ===========================================================
_BROWSER_ACTIONS = {}
_BROWSER_ACTION_TTL_SECONDS = 600


def _clean_browser_url(url):
    """Validate and normalize a URL that may be opened in a browser."""
    if not isinstance(url, str):
        return None, "URL must be a string."
    url = url.strip()
    if not url:
        return None, "A URL is required."
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None, "Only http:// or https:// URLs with a host are supported."
    return url, None


def _browser_action_confirmed(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on", "confirmed"}
    return False


def open_browser_url(url: str) -> bool:
    """Open a validated URL in the default browser on the Drift host."""
    clean_url, error = _clean_browser_url(url)
    if error:
        return False
    try:
        return bool(webbrowser.open(clean_url, new=2))
    except Exception:
        return False


def request_browser_action(url, description=None, user_id="web_user"):
    """Create a pending browser action that requires explicit user confirmation."""
    clean_url, error = _clean_browser_url(url)
    if error:
        return {"ok": False, "status": "invalid_url", "error": error, "opened": False}
    action_id = uuid.uuid4().hex
    _BROWSER_ACTIONS[action_id] = {
        "id": action_id,
        "url": clean_url,
        "description": (description or "").strip(),
        "user_id": user_id or "web_user",
        "created_at": time.time(),
        "confirmed": False,
        "status": "pending_confirmation",
    }
    return {
        "ok": True,
        "status": "confirmation_required",
        "requires_confirmation": True,
        "action_id": action_id,
        "url": clean_url,
        "opened": False,
        "message": f"Drift needs your confirmation before opening {clean_url} in the browser.",
    }


def confirm_browser_action(action_id=None, url=None, user_id="web_user"):
    """Confirm and execute a pending browser action, or open a URL after confirmation."""
    if action_id:
        action = _BROWSER_ACTIONS.pop(str(action_id), None)
        if action is None:
            return {
                "ok": False,
                "status": "not_found",
                "error": "That browser action is no longer pending.",
                "opened": False,
            }
        if time.time() - float(action.get("created_at", 0)) > _BROWSER_ACTION_TTL_SECONDS:
            return {
                "ok": False,
                "status": "expired",
                "error": "That browser action expired. Please request it again.",
                "opened": False,
                "url": action.get("url"),
            }
        target_url = action.get("url")
    else:
        target_url = url

    clean_url, error = _clean_browser_url(target_url)
    if error:
        return {"ok": False, "status": "invalid_url", "error": error, "opened": False}

    opened = open_browser_url(clean_url)
    result = {
        "ok": opened,
        "status": "opened" if opened else "failed",
        "action_id": str(action_id) if action_id else None,
        "url": clean_url,
        "opened": opened,
        "message": (
            f"Opened {clean_url} in the browser."
            if opened
            else f"Could not open {clean_url} in the browser."
        ),
    }
    return result


def cancel_browser_action(action_id=None):
    if not action_id:
        return {"ok": True, "status": "cancelled", "opened": False, "message": "Browser action cancelled."}
    action = _BROWSER_ACTIONS.pop(str(action_id), None)
    return {
        "ok": True,
        "status": "cancelled",
        "action_id": str(action_id),
        "url": action.get("url") if action else None,
        "opened": False,
        "message": "Browser action cancelled.",
    }


def browser_action(url=None, confirmed=False, action_id=None, cancel=False, user_id="web_user", description=None):
    """Convenience entry point used by routes and tests."""
    if cancel:
        return cancel_browser_action(action_id)
    if action_id and _browser_action_confirmed(confirmed):
        return confirm_browser_action(action_id=action_id, user_id=user_id)
    if url is not None and _browser_action_confirmed(confirmed):
        return confirm_browser_action(url=url, user_id=user_id)
    if url is not None:
        return request_browser_action(url, description=description, user_id=user_id)
    if action_id:
        action = _BROWSER_ACTIONS.get(str(action_id))
        if action is None:
            return {"ok": False, "status": "not_found", "error": "Browser action not found.", "opened": False}
        return {"ok": True, "status": action.get("status", "pending_confirmation"), "action_id": action_id, "url": action.get("url"), "requires_confirmation": True, "opened": False}
    return {"ok": False, "status": "invalid_request", "error": "A URL or action_id is required.", "opened": False}


# Backwards/alternate names that tests or UI code may use.
request_browser_action_url = request_browser_action
confirm_browser_action_url = confirm_browser_action
browser_action_request = request_browser_action
browser_action_confirm = confirm_browser_action


# ===========================================================
app = Flask(__name__)
app.secret_key = os.getenv("SESSION_SECRET") or secrets.token_hex(32)
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax")
CORS(app, resources={r"/*": {"origins": "*"}})


@app.route("/", methods=["GET"])
def index():
    return send_file("cebri.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()
    user_id = data.get("user_id", "web_user")
 
    if not message:
        return jsonify({"reply": "I didn't catch a message — try again."}), 400
 
    try:
        reply = ask_cebri(user_id, message)
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"reply": f"Something went wrong on my end: {e}"}), 500


@app.route("/google-calendar/status", methods=["GET"])
def google_calendar_status():
    token_exists = GOOGLE_TOKEN_FILE.exists()
    return jsonify({
        "configured": google_calendar_credentials_configured(),
        "connected": token_exists,
        "access": "read-only events",
        "scope": GOOGLE_CALENDAR_SCOPE,
    })


@app.route("/google-calendar/connect", methods=["GET"])
def google_calendar_connect():
    if not google_calendar_credentials_configured():
        return (
            "Google Calendar OAuth is not configured. Set GOOGLE_CLIENT_ID, "
            "GOOGLE_CLIENT_SECRET, and GOOGLE_REDIRECT_URI in the env file, then restart the backend.",
            503,
        )
    state = secrets.token_urlsafe(32)
    session["google_calendar_oauth_state"] = state
    params = {
        "client_id": os.getenv("GOOGLE_CLIENT_ID"),
        "redirect_uri": os.getenv("GOOGLE_REDIRECT_URI"),
        "response_type": "code",
        "scope": GOOGLE_CALENDAR_SCOPE,
        "access_type": "offline",
        "prompt": "consent",
        "include_granted_scopes": "true",
        "state": state,
    }
    return redirect("https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(params))


@app.route("/google-calendar/callback", methods=["GET"])
def google_calendar_callback():
    expected_states = [
        session.pop("google_calendar_oauth_state", ""),
        session.pop("youtube_oauth_state", ""),
        session.pop("contacts_oauth_state", ""),
    ]
    supplied_state = request.args.get("state", "")
    if not supplied_state or supplied_state not in [state for state in expected_states if state]:
        return "Google authorization state did not match. Please start Connect again from Drift.", 400
    if request.args.get("error"):
        return "Google Calendar access was not granted. You can return to Drift.", 400
    code = request.args.get("code", "")
    if not code:
        return "Google did not return an authorization code.", 400
    try:
        response = requests.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": os.getenv("GOOGLE_CLIENT_ID", ""),
                "client_secret": os.getenv("GOOGLE_CLIENT_SECRET", ""),
                "redirect_uri": google_redirect_uri(),
                "grant_type": "authorization_code",
            },
            timeout=20,
        )
        if not response.ok:
            app.logger.warning("Google OAuth token exchange returned HTTP %s", response.status_code)
            return "Google Calendar token exchange failed. Check OAuth settings and the redirect URI.", 502
        tokens = response.json()
        tokens["expires_at"] = datetime.now(timezone.utc).timestamp() + int(tokens.get("expires_in", 3600))
        granted_scopes = tokens.get("scope", "").split()
        connected_services = []
        if GOOGLE_CALENDAR_SCOPE in granted_scopes:
            save_google_tokens(tokens)
            connected_services.append("Google Calendar")
        if YOUTUBE_SCOPE in granted_scopes:
            save_youtube_tokens(tokens)
            connected_services.append("YouTube")
        if CONTACTS_SCOPE in granted_scopes:
            save_contacts_tokens(tokens)
            connected_services.append("Google Contacts")
        if not connected_services:
            return "No supported Google read permission was granted. No token was saved.", 403
        return "Connected to Drift with read-only access: " + ", ".join(connected_services) + ". You may return to the Drift tab."
    except requests.RequestException:
        app.logger.exception("Google OAuth token exchange failed")
        return "Could not reach Google to finish connecting Google Calendar.", 502


@app.route("/spotify/status", methods=["GET"])
def spotify_status():
    token_exists = SPOTIFY_TOKEN_FILE.exists()
    return jsonify({
        "configured": spotify_credentials_configured(),
        "connected": token_exists,
        "access": "read-only playlists",
        "scope": SPOTIFY_SCOPE,
    })


@app.route("/spotify/connect", methods=["GET"])
def spotify_connect():
    if not spotify_credentials_configured():
        return (
            "Spotify OAuth is not configured. Set SPOTIFY_CLIENT_ID, "
            "SPOTIFY_CLIENT_SECRET, and SPOTIFY_REDIRECT_URI in the env file, then restart the backend.",
            503,
        )
    state = secrets.token_urlsafe(32)
    session["spotify_oauth_state"] = state
    params = {
        "client_id": os.getenv("SPOTIFY_CLIENT_ID"),
        "redirect_uri": os.getenv("SPOTIFY_REDIRECT_URI"),
        "response_type": "code",
        "scope": SPOTIFY_SCOPE,
        "state": state,
    }
    return redirect("https://accounts.spotify.com/authorize?" + urlencode(params))


@app.route("/spotify/callback", methods=["GET"])
def spotify_callback():
    expected_state = session.pop("spotify_oauth_state", "")
    supplied_state = request.args.get("state", "")
    if not expected_state or not supplied_state or not secrets.compare_digest(expected_state, supplied_state):
        return "Spotify authorization state did not match. Please start Connect again from Drift.", 400
    if request.args.get("error"):
        return "Spotify access was not granted. You can return to Drift.", 400
    code = request.args.get("code", "")
    if not code:
        return "Spotify did not return an authorization code.", 400
    try:
        response = requests.post(
            "https://accounts.spotify.com/api/token",
            data={
                "code": code,
                "client_id": os.getenv("SPOTIFY_CLIENT_ID", ""),
                "client_secret": os.getenv("SPOTIFY_CLIENT_SECRET", ""),
                "redirect_uri": os.getenv("SPOTIFY_REDIRECT_URI", ""),
                "grant_type": "authorization_code",
            },
            timeout=20,
        )
        if not response.ok:
            app.logger.warning("Spotify OAuth token exchange returned HTTP %s", response.status_code)
            return "Spotify token exchange failed. Check OAuth settings and the redirect URI.", 502
        tokens = response.json()
        tokens["expires_at"] = datetime.now(timezone.utc).timestamp() + int(tokens.get("expires_in", 3600))
        save_spotify_tokens(tokens)
        return "Spotify is connected to Drift with read-only playlist access. You may return to the Drift tab."
    except requests.RequestException:
        app.logger.exception("Spotify OAuth token exchange failed")
        return "Could not reach Spotify to finish connecting.", 502


@app.route("/youtube/status", methods=["GET"])
def youtube_status():
    token_exists = YOUTUBE_TOKEN_FILE.exists()
    return jsonify({
        "configured": youtube_credentials_configured(),
        "connected": token_exists,
        "access": "read-only videos",
        "scope": YOUTUBE_SCOPE,
    })


@app.route("/youtube/connect", methods=["GET"])
def youtube_connect():
    if not youtube_credentials_configured():
        return (
            "YouTube OAuth is not configured. Set GOOGLE_CLIENT_ID, "
            "GOOGLE_CLIENT_SECRET, and YOUTUBE_REDIRECT_URI in the env file, then restart the backend.",
            503,
        )
    state = secrets.token_urlsafe(32)
    session["youtube_oauth_state"] = state
    params = {
        "client_id": os.getenv("GOOGLE_CLIENT_ID"),
        "redirect_uri": youtube_redirect_uri(),
        "response_type": "code",
        "scope": YOUTUBE_SCOPE,
        "access_type": "offline",
        "prompt": "consent",
        "include_granted_scopes": "true",
        "state": state,
    }
    return redirect("https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(params))


@app.route("/youtube/callback", methods=["GET"])
def youtube_callback():
    expected_state = session.pop("youtube_oauth_state", "")
    supplied_state = request.args.get("state", "")
    if not expected_state or not supplied_state or not secrets.compare_digest(expected_state, supplied_state):
        return "YouTube authorization state did not match. Please start Connect again from Drift.", 400
    if request.args.get("error"):
        return "YouTube access was not granted. You can return to Drift.", 400
    code = request.args.get("code", "")
    if not code:
        return "Google did not return an authorization code.", 400
    try:
        response = requests.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": os.getenv("GOOGLE_CLIENT_ID", ""),
                "client_secret": os.getenv("GOOGLE_CLIENT_SECRET", ""),
                "redirect_uri": youtube_redirect_uri(),
                "grant_type": "authorization_code",
            },
            timeout=20,
        )
        if not response.ok:
            app.logger.warning("YouTube OAuth token exchange returned HTTP %s", response.status_code)
            return "YouTube token exchange failed. Check OAuth settings and the redirect URI.", 502
        tokens = response.json()
        granted_scopes = tokens.get("scope", "").split()
        if YOUTUBE_SCOPE not in granted_scopes:
            return "YouTube read permission was not granted. No token was saved.", 403
        tokens["expires_at"] = datetime.now(timezone.utc).timestamp() + int(tokens.get("expires_in", 3600))
        save_youtube_tokens(tokens)
        return "YouTube is connected to Drift with read-only video access. You may return to the Drift tab."
    except requests.RequestException:
        app.logger.exception("YouTube OAuth token exchange failed")
        return "Could not reach Google to finish connecting YouTube.", 502


@app.route("/contacts/status", methods=["GET"])
def contacts_status():
    token_exists = CONTACTS_TOKEN_FILE.exists()
    return jsonify({
        "configured": contacts_credentials_configured(),
        "connected": token_exists,
        "access": "read-only contacts",
        "scope": CONTACTS_SCOPE,
    })


@app.route("/contacts/connect", methods=["GET"])
def contacts_connect():
    if not contacts_credentials_configured():
        return (
            "Google Contacts OAuth is not configured. Set GOOGLE_CLIENT_ID, "
            "GOOGLE_CLIENT_SECRET, and CONTACTS_REDIRECT_URI in the env file, then restart the backend.",
            503,
        )
    state = secrets.token_urlsafe(32)
    session["contacts_oauth_state"] = state
    params = {
        "client_id": os.getenv("GOOGLE_CLIENT_ID"),
        "redirect_uri": contacts_redirect_uri(),
        "response_type": "code",
        "scope": CONTACTS_SCOPE,
        "access_type": "offline",
        "prompt": "consent",
        "include_granted_scopes": "true",
        "state": state,
    }
    return redirect("https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(params))


@app.route("/contacts/callback", methods=["GET"])
def contacts_callback():
    expected_state = session.pop("contacts_oauth_state", "")
    supplied_state = request.args.get("state", "")
    if not expected_state or not supplied_state or not secrets.compare_digest(expected_state, supplied_state):
        return "Google Contacts authorization state did not match. Please start Connect again from Drift.", 400
    if request.args.get("error"):
        return "Google Contacts access was not granted. You can return to Drift.", 400
    code = request.args.get("code", "")
    if not code:
        return "Google did not return an authorization code.", 400
    try:
        response = requests.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": os.getenv("GOOGLE_CLIENT_ID", ""),
                "client_secret": os.getenv("GOOGLE_CLIENT_SECRET", ""),
                "redirect_uri": contacts_redirect_uri(),
                "grant_type": "authorization_code",
            },
            timeout=20,
        )
        if not response.ok:
            app.logger.warning("Google Contacts OAuth token exchange returned HTTP %s", response.status_code)
            return "Google Contacts token exchange failed. Check OAuth settings and the redirect URI.", 502
        tokens = response.json()
        granted_scopes = tokens.get("scope", "").split()
        if CONTACTS_SCOPE not in granted_scopes:
            return "Google Contacts read permission was not granted. No token was saved.", 403
        tokens["expires_at"] = datetime.now(timezone.utc).timestamp() + int(tokens.get("expires_in", 3600))
        save_contacts_tokens(tokens)
        return "Google Contacts is connected to Drift with read-only access. You may return to the Drift tab."
    except requests.RequestException:
        app.logger.exception("Google Contacts OAuth token exchange failed")
        return "Could not reach Google to finish connecting Google Contacts.", 502


@app.route("/tts", methods=["POST"])
def fish_audio_tts():
    """Proxy speech synthesis through Fish Audio without exposing its key."""
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Text is required."}), 400
    if len(text) > 4000:
        return jsonify({"error": "Text is too long (maximum 4000 characters)."}), 413

    api_key = os.getenv("FISH_AUDIO_API_KEY", "").strip()
    if not api_key:
        return jsonify({"error": "Fish Audio is not configured. Add FISH_AUDIO_API_KEY to your env file."}), 503

    voice_id = data.get("voice_id") or os.getenv("FISH_AUDIO_VOICE_ID", "933563129e564b19a115bedd57b7406a")
    model = os.getenv("FISH_AUDIO_MODEL", "s2.1-pro-free")
    try:
        upstream = requests.post(
            "https://api.fish.audio/v1/tts",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "model": model,
            },
            json={"text": text.strip(), "reference_id": voice_id, "format": "mp3"},
            timeout=(10, 90),
        )
    except requests.RequestException:
        app.logger.exception("Fish Audio request failed")
        return jsonify({"error": "Could not reach Fish Audio."}), 502

    if not upstream.ok:
        app.logger.warning("Fish Audio returned HTTP %s", upstream.status_code)
        return jsonify({"error": f"Fish Audio returned HTTP {upstream.status_code}. Check the API key and voice ID."}), 502

    return Response(upstream.content, mimetype="audio/mpeg", headers={"Cache-Control": "no-store"})
 
 
@app.route("/browser-action", methods=["GET", "POST"])
@app.route("/browser-actions", methods=["GET", "POST"])
@app.route("/browser/action", methods=["GET", "POST"])
def browser_action_endpoint():
    data = request.get_json(silent=True) or {}
    if request.method == "GET":
        data = {**request.args.to_dict(), **data}
    url = data.get("url") or data.get("target_url")
    action_id = data.get("action_id") or data.get("id") or data.get("actionId")
    confirmed = _browser_action_confirmed(data.get("confirmed", data.get("confirm", False)))
    cancelled = _browser_action_confirmed(data.get("cancel", data.get("cancelled", False)))

    if cancelled:
        result = cancel_browser_action(action_id)
        status_code = 200 if result.get("ok") else 400
        return jsonify(result), status_code

    if action_id and confirmed:
        result = confirm_browser_action(action_id=action_id)
        status_code = 200 if result.get("ok") else 400
        return jsonify(result), status_code

    if url is not None and confirmed:
        result = confirm_browser_action(url=url)
        status_code = 200 if result.get("ok") else 400
        return jsonify(result), status_code

    if url is not None:
        # Allow callers to explicitly request a pending confirmation object.
        if _browser_action_confirmed(data.get("request_confirmation", False)):
            result = request_browser_action(url, description=data.get("description"), user_id=data.get("user_id", "web_user"))
            return jsonify(result), 202
        return jsonify({
            "ok": False,
            "status": "confirmation_required",
            "requires_confirmation": True,
            "url": _clean_browser_url(url)[0],
            "error": "User confirmation is required before opening a browser URL.",
            "opened": False,
        }), 400

    if action_id:
        result = browser_action(action_id=action_id)
        return jsonify(result), 200

    return jsonify({"ok": False, "status": "invalid_request", "error": "A URL or action_id is required.", "opened": False}), 400


@app.route("/browser-action/confirm", methods=["POST"])
@app.route("/browser-action-confirm", methods=["POST"])
def browser_action_confirm_endpoint():
    data = request.get_json(silent=True) or {}
    result = confirm_browser_action(
        action_id=data.get("action_id") or data.get("id") or data.get("actionId"),
        url=data.get("url") or data.get("target_url"),
    )
    return jsonify(result), (200 if result.get("ok") else 400)


@app.route("/browser-action/cancel", methods=["POST"])
@app.route("/browser-action-cancel", methods=["POST"])
def browser_action_cancel_endpoint():
    data = request.get_json(silent=True) or {}
    result = cancel_browser_action(data.get("action_id") or data.get("id") or data.get("actionId"))
    return jsonify(result), 200


@app.route("/browser-action/<action_id>", methods=["GET"])
def browser_action_status_endpoint(action_id):
    result = browser_action(action_id=action_id)
    return jsonify(result), 200


@app.route("/browser/open", methods=["POST"])
@app.route("/browser-open", methods=["POST"])
def browser_open_endpoint():
    data = request.get_json(silent=True) or {}
    url = data.get("url") or data.get("target_url")
    if not _browser_action_confirmed(data.get("confirmed", False)):
        result = request_browser_action(url, description=data.get("description"), user_id=data.get("user_id", "web_user"))
        return jsonify(result), 400 if not result.get("ok") else 202
    result = confirm_browser_action(url=url)
    return jsonify(result), (200 if result.get("ok") else 400)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "Drift is online"})


@app.route("/status", methods=["GET"])
def status():
    configured_model = (os.getenv("CEBRI_MODEL") or os.getenv("MODEL") or "").strip()
    configured_provider = os.getenv("CEBRI_PROVIDER", "").strip().lower()
    provider = "fallback"
    if llm is not None:
        provider = "live"
    elif configured_model:
        provider = "configured-but-unavailable"

    provider_name = configured_provider or (
        "anthropic" if os.getenv("ANTHROPIC_API_KEY")
        else "openai" if os.getenv("OPENAI_API_KEY")
        else "none"
    )

    return jsonify({
        "mode": "live" if llm is not None else "fallback",
        "live_model_enabled": llm is not None,
        "provider": provider_name,
        "configured_model": configured_model or "default",
        "warning": startup_warning,
    })
 
 
if __name__ == "__main__":
    if startup_warning:
        print(f"STARTUP WARNING: {startup_warning}")
        print("Falling back to local mode until the key is fixed.")
    print("Drift API running at http://localhost:5000/ask")
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
    app.run(host="0.0.0.0", port=port, debug=debug, threaded=True, use_reloader=False)
