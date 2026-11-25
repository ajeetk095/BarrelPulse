"""Contact form back-end: sends mail through Gmail SMTP (SSL) with credentials from secrets only.

Needs two secrets (Streamlit secrets or environment variables):
  GMAIL_USER          the Gmail address that sends the mail (e.g. the developer's own address)
  GMAIL_APP_PASSWORD  a 16-character Google *App Password* (requires 2-Step Verification) — NOT the account password
Optional: CONTACT_TO (defaults to the developer address).
Nothing is sent, and no success is reported, unless the SMTP server accepts the message.
"""
from __future__ import annotations

import re
import smtplib
import ssl
from email.message import EmailMessage
from email.utils import formataddr
from typing import Callable, Optional

from . import gwp3_config as G
from .secrets import get_secret

EMAIL_RE = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,255}\.[A-Za-z]{2,}$")
MAX = {"name": 80, "email": 254, "subject": 120, "message": 3000}


class ContactError(Exception):
    """Validation or delivery failure with a user-presentable message."""


def clean_line(s: str) -> str:
    """Single-line field: strip control chars (blocks header injection)."""
    return re.sub(r"[\x00-\x1f\x7f]+", " ", s or "").strip()


def validate(name: str, email: str, subject: str, message: str) -> dict:
    d = {
        "name": clean_line(name),
        "email": clean_line(email),
        "subject": clean_line(subject),
        "message": (message or "").replace("\x00", "").strip(),
    }
    if not d["name"]:
        raise ContactError("Please enter your name.")
    if not EMAIL_RE.match(d["email"]):
        raise ContactError("Please enter a valid email address so the developer can reply.")
    if not d["subject"]:
        raise ContactError("Please enter a subject.")
    if len(d["message"]) < 10:
        raise ContactError("Please write a message of at least 10 characters.")
    for k, lim in MAX.items():
        if len(d[k]) > lim:
            raise ContactError(f"{k.title()} is too long (maximum {lim} characters).")
    return d


def _credentials() -> tuple[str, str]:
    user = (get_secret("GMAIL_USER") or "").strip()
    pwd = (get_secret("GMAIL_APP_PASSWORD") or "").replace(" ", "").strip()
    return user, pwd


def configured() -> bool:
    user, pwd = _credentials()
    return bool(user and pwd)


def build_message(d: dict, sender: str, to: str) -> EmailMessage:
    m = EmailMessage()
    m["From"] = formataddr(("BarrelPulse contact form", sender))
    m["To"] = to
    m["Reply-To"] = formataddr((d["name"], d["email"]))
    m["Subject"] = f"[BarrelPulse] {d['subject']}"
    m.set_content(
        f"From: {d['name']} <{d['email']}>\n\n{d['message']}\n\n— sent via the BarrelPulse app contact form"
    )
    return m


def send_message(
    name: str,
    email: str,
    subject: str,
    message: str,
    smtp_factory: Optional[Callable] = None,
) -> None:
    """Validate and send. Raises :class:`ContactError` on any problem (never returns success silently)."""
    d = validate(name, email, subject, message)
    user, pwd = _credentials()
    if not (user and pwd):
        raise ContactError(
            "The mail service is not configured on this deployment yet "
            "(GMAIL_USER / GMAIL_APP_PASSWORD secrets missing). Please use the email or LinkedIn buttons instead."
        )
    to = (get_secret("CONTACT_TO") or G.DEVELOPER["email"]).strip()
    msg = build_message(d, user, to)
    try:
        factory = smtp_factory or (
            lambda: smtplib.SMTP_SSL(
                "smtp.gmail.com", 465, timeout=20, context=ssl.create_default_context()
            )
        )
        with factory() as s:
            s.login(user, pwd)
            s.send_message(msg)
    except smtplib.SMTPAuthenticationError as exc:
        raise ContactError("Gmail rejected the sign-in (check the app password).") from exc
    except (smtplib.SMTPException, OSError) as exc:
        raise ContactError(
            "The message could not be delivered right now. Please try again later or use the email button."
        ) from exc
