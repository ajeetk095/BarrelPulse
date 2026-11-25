# Contributing
Academic project; fixes and clarity improvements are welcome.
1. `pip install -r requirements-dev.txt`
2. `python -m pytest tests -q` must pass; add a test with each change.
3. Never commit secrets (`.env`, `.streamlit/secrets.toml` are git-ignored).
4. Keep recorded (submitted-notebook) values and computed values clearly labelled; never hard-code a computed result.
