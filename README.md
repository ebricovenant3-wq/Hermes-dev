# Drift (Hermes)

Lightweight Flask + CrewAI backend and browser UI for the Drift personal assistant.

## Run

```bash
pip install -r requirements.txt 2>/dev/null || pip install flask flask-cors crewai crewai-tools python-dotenv requests
python app.py
```

Open `cebri.html` (or `hermes.html`, which redirects to it) in a browser.

## Voice and browser actions

- Microphone input uses the browser Web Speech API (`SpeechRecognition`).
- Spoken replies use browser `speechSynthesis` or the optional Fish Audio proxy at `/tts`.
- Browser actions are gated: the UI asks for confirmation, and the backend only opens a URL after `confirmed: true` is supplied to `/browser-action`.
- Pending browser actions can be confirmed through `/browser-action/confirm` or cancelled through `/browser-action/cancel`.

## Tests

```bash
pytest tests/test_app.py
```
