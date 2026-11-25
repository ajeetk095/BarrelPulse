import os, sys
import pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from core import contact as C


class FakeSMTP:
    sent, logged = [], []
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def login(self, u, p): FakeSMTP.logged.append((u, p))
    def send_message(self, m): FakeSMTP.sent.append(m)


@pytest.fixture(autouse=True)
def env(monkeypatch):
    FakeSMTP.sent, FakeSMTP.logged = [], []
    monkeypatch.setenv("GMAIL_USER", "sender@example.com"); monkeypatch.setenv("GMAIL_APP_PASSWORD", "x" * 16)
    monkeypatch.delenv("CONTACT_TO", raising=False)


def test_sends_to_developer_with_reply_to():
    C.send_message("Ann", "ann@x.org", "Hi", "Hello there, nice app!", FakeSMTP)
    m = FakeSMTP.sent[0]
    assert m["To"] == "ajeetk095@gmail.com" and "ann@x.org" in m["Reply-To"] and m["Subject"].startswith("[BarrelPulse]")


@pytest.mark.parametrize("args", [("", "a@b.co", "s", "long enough message"), ("A", "bad", "s", "long enough message"),
                                  ("A", "a@b.co", "", "long enough message"), ("A", "a@b.co", "s", "short")])
def test_validation_blocks_send(args):
    with pytest.raises(C.ContactError):
        C.send_message(*args, smtp_factory=FakeSMTP)
    assert not FakeSMTP.sent


def test_header_injection_neutralised():
    C.send_message("Ann\r\nBcc: evil@x.com", "ann@x.org", "Hi\nBcc: evil@x.com", "Hello there, nice app!", FakeSMTP)
    m = FakeSMTP.sent[0]
    assert "evil" not in (m["Bcc"] or "") and "\n" not in m["Subject"]


def test_not_configured_reports_failure(monkeypatch):
    monkeypatch.delenv("GMAIL_USER"); monkeypatch.delenv("GMAIL_APP_PASSWORD")
    with pytest.raises(C.ContactError):
        C.send_message("Ann", "ann@x.org", "Hi", "Hello there, nice app!", FakeSMTP)


def test_smtp_failure_is_not_success():
    import smtplib
    class Bad(FakeSMTP):
        def send_message(self, m): raise smtplib.SMTPException("down")
    with pytest.raises(C.ContactError):
        C.send_message("Ann", "ann@x.org", "Hi", "Hello there, nice app!", Bad)


def test_form_flow_and_popup(monkeypatch):
    from streamlit.testing.v1 import AppTest
    calls = []
    monkeypatch.setattr(C, "send_message", lambda *a, **k: calls.append(a))
    at = AppTest.from_file(os.path.join(ROOT, "views", "developer.py"), default_timeout=60).run()
    assert not at.exception
    at.text_input(key="contact.name").set_value("Ann"); at.text_input(key="contact.email").set_value("ann@x.org")
    at.text_input(key="contact.subject").set_value("Hi"); at.text_area(key="contact.message").set_value("Hello there, nice app!")
    at.button[0].click().run()
    assert calls and not at.exception
    assert at.session_state["contact.name"] == ""          # form cleared after success
