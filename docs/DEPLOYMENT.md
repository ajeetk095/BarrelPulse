# Deployment: GitHub + Streamlit Community Cloud (guide — not yet performed)
1. Create an empty GitHub repo (suggested `BarrelPulse`), then:
   ```bash
   cd BarrelPulse && git init && git add . && git commit -m "BarrelPulse GWP3"
   git branch -M main && git remote add origin https://github.com/<you>/BarrelPulse.git && git push -u origin main
   ```
   Check `git status` shows no `.env` / `secrets.toml` (both are git-ignored). The CI workflow runs tests, bandit and pip-audit.
2. share.streamlit.io → *Create app* → repo, branch `main`, main file `app.py`; Advanced settings → Python 3.12.
3. App settings → **Secrets** (optional, TOML): `FRED_API_KEY = "…"`, `EIA_API_KEY = "…"`. Without keys the app uses FRED's public CSV and Yahoo Finance.
4. Open **Live Market**: if a provider is rate-limited the page says which one failed and keeps working on cached/bundled data.
5. Post-deploy checklist: open each of the 14 pages, press ↺ Reset all, switch Data source to Live, press Refresh, run the Improvements sweep.
Streamlit Cloud restarts clear runtime caches; first load fits models (≈ 5–20 s).

## Contact form (Developer page)
The form sends mail from inside the app through Gmail SMTP (smtp.gmail.com:465, SSL). It needs `GMAIL_USER` and `GMAIL_APP_PASSWORD` in Streamlit Secrets: enable 2-Step Verification on the sending Google account, create an **App Password**, paste it as a secret. Without them the page shows a notice and sends nothing (no false "sent" message). Messages arrive in the inbox of `CONTACT_TO` (default: the developer address) with Reply-To set to the visitor. Never commit these secrets. After deploying, send yourself a test message — the build sandbox could not reach Gmail, so real delivery is untested.
