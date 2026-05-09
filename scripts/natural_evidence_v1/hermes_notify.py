from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import json
import os
import shutil
import smtplib
import socket
import ssl
import subprocess
import urllib.parse
import urllib.request
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Mapping

from scripts.natural_evidence_v1.common import write_json


SCHEMA_NAME = "natural_evidence_v1_hermes_notification_v1"
DEFAULT_ENV_FILE = Path.home() / ".hermes" / ".env"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Send Hermes supervision notifications through Telegram and email. "
            "This is for progress supervision only; it does not run experiments."
        )
    )
    parser.add_argument("--subject", required=True)
    parser.add_argument("--body", default="")
    parser.add_argument("--body-file", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--channels", default="telegram,email")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE))
    parser.add_argument("--no-env-file", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def _parse_env_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return None
    if stripped.startswith("export "):
        stripped = stripped[len("export ") :].strip()
    key, value = stripped.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        return None
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return key, value


def load_env_file(path: Path, *, override: bool = False) -> dict[str, Any]:
    loaded: list[str] = []
    skipped_existing: list[str] = []
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "loaded_keys": loaded,
            "skipped_existing_keys": skipped_existing,
        }
    for line in path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_env_line(line)
        if parsed is None:
            continue
        key, value = parsed
        if key in os.environ and not override:
            skipped_existing.append(key)
            continue
        os.environ[key] = value
        loaded.append(key)
    return {
        "path": str(path),
        "exists": True,
        "loaded_keys": sorted(loaded),
        "skipped_existing_keys": sorted(skipped_existing),
    }


def _env(*names: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return ""


def _read_body(body: str, body_file: str) -> str:
    if body_file:
        return Path(body_file).read_text(encoding="utf-8")
    return body


def _redact(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 6:
        return "***"
    return value[:3] + "***" + value[-3:]


def send_telegram(*, subject: str, body: str, dry_run: bool) -> dict[str, Any]:
    token = _env("HERMES_TG_BOT_TOKEN", "TELEGRAM_BOT_TOKEN", "TG_BOT_TOKEN")
    chat_id = _env(
        "HERMES_TG_CHAT_ID",
        "TELEGRAM_CHAT_ID",
        "TG_CHAT_ID",
        "TELEGRAM_HOME_CHANNEL",
    )
    if not token or not chat_id:
        return {
            "channel": "telegram",
            "status": "NOT_CONFIGURED",
            "sent": False,
            "missing": [
                name
                for name, value in {
                    "HERMES_TG_BOT_TOKEN": token,
                    "HERMES_TG_CHAT_ID": chat_id,
                }.items()
                if not value
            ],
        }
    text = f"{subject}\n\n{body}".strip()
    if len(text) > 4096:
        text = text[:4000] + "\n\n[truncated]"
    if dry_run:
        return {
            "channel": "telegram",
            "status": "DRY_RUN_NOT_SENT",
            "sent": False,
            "bot_token": _redact(token),
            "chat_id": _redact(chat_id),
            "text_chars": len(text),
        }
    payload = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode("utf-8")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        request = urllib.request.Request(url, data=payload, method="POST")
        with urllib.request.urlopen(request, timeout=20) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
        return {
            "channel": "telegram",
            "status": "SENT" if response_payload.get("ok") else "FAILED",
            "sent": bool(response_payload.get("ok")),
            "response_ok": bool(response_payload.get("ok")),
        }
    except Exception as exc:  # pragma: no cover - network failure path
        return {
            "channel": "telegram",
            "status": "FAILED",
            "sent": False,
            "error": str(exc),
        }


def _default_from() -> str:
    configured = _env("HERMES_EMAIL_FROM", "EMAIL_FROM", "EMAIL_ADDRESS")
    if configured:
        return configured
    user = _env("USER") or "codex"
    host = socket.getfqdn() or socket.gethostname() or "localhost"
    return f"{user}@{host}"


def _send_email_smtp(*, to_addr: str, from_addr: str, subject: str, body: str, dry_run: bool) -> dict[str, Any]:
    host = _env("HERMES_SMTP_HOST", "SMTP_HOST", "EMAIL_SMTP_HOST")
    port = int(_env("HERMES_SMTP_PORT", "SMTP_PORT", "EMAIL_SMTP_PORT") or "587")
    username = _env("HERMES_SMTP_USER", "SMTP_USER", "EMAIL_ADDRESS")
    password = _env("HERMES_SMTP_PASSWORD", "SMTP_PASSWORD", "EMAIL_PASSWORD")
    use_ssl = (_env("HERMES_SMTP_SSL", "SMTP_SSL", "EMAIL_SMTP_SSL") or "").lower() in {
        "1",
        "true",
        "yes",
    } or port == 465
    use_tls = (_env("HERMES_SMTP_TLS", "SMTP_TLS", "EMAIL_SMTP_TLS") or "1").lower() not in {
        "0",
        "false",
        "no",
    }
    if not host:
        return {"status": "NO_SMTP_HOST", "sent": False}
    if dry_run:
        return {
            "status": "DRY_RUN_NOT_SENT",
            "sent": False,
            "method": "smtp_ssl" if use_ssl else "smtp",
            "smtp_host": host,
            "smtp_port": port,
            "smtp_user": _redact(username),
            "to": to_addr,
            "from": from_addr,
        }
    message = EmailMessage()
    message["To"] = to_addr
    message["From"] = from_addr
    message["Subject"] = subject
    message.set_content(body)
    try:
        smtp_cls = smtplib.SMTP_SSL if use_ssl else smtplib.SMTP
        with smtp_cls(host, port, timeout=30) as smtp:
            if use_tls and not use_ssl:
                smtp.starttls(context=ssl.create_default_context())
            if username or password:
                smtp.login(username, password)
            smtp.send_message(message)
        return {
            "status": "SENT",
            "sent": True,
            "method": "smtp_ssl" if use_ssl else "smtp",
            "smtp_host": host,
        }
    except Exception as exc:  # pragma: no cover - network failure path
        return {"status": "FAILED", "sent": False, "method": "smtp", "error": str(exc)}


def _send_email_sendmail(*, to_addr: str, from_addr: str, subject: str, body: str, dry_run: bool) -> dict[str, Any]:
    sendmail = _env("HERMES_SENDMAIL", "SENDMAIL") or shutil.which("sendmail") or ""
    if not sendmail:
        for candidate in ("/usr/sbin/sendmail", "/usr/bin/sendmail"):
            if Path(candidate).exists():
                sendmail = candidate
                break
    if not sendmail:
        return {"status": "NO_SENDMAIL", "sent": False}
    if dry_run:
        return {
            "status": "DRY_RUN_NOT_SENT",
            "sent": False,
            "method": "sendmail",
            "sendmail": sendmail,
            "to": to_addr,
            "from": from_addr,
        }
    message = EmailMessage()
    message["To"] = to_addr
    message["From"] = from_addr
    message["Subject"] = subject
    message.set_content(body)
    try:
        subprocess.run(
            [sendmail, "-t", "-oi"],
            input=message.as_bytes(),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return {"status": "SENT", "sent": True, "method": "sendmail", "sendmail": sendmail}
    except Exception as exc:  # pragma: no cover - host-specific failure path
        return {"status": "FAILED", "sent": False, "method": "sendmail", "error": str(exc)}


def send_email(*, subject: str, body: str, dry_run: bool) -> dict[str, Any]:
    to_addr = _env(
        "HERMES_EMAIL_TO",
        "HERMES_NOTIFY_EMAIL_TO",
        "EMAIL_TO",
        "EMAIL_HOME_ADDRESS",
        "EMAIL_ADDRESS",
    )
    from_addr = _env("HERMES_EMAIL_FROM", "EMAIL_FROM", "EMAIL_ADDRESS") or _default_from()
    if not to_addr:
        return {
            "channel": "email",
            "status": "NOT_CONFIGURED",
            "sent": False,
            "missing": ["HERMES_EMAIL_TO"],
        }
    smtp_result = _send_email_smtp(
        to_addr=to_addr,
        from_addr=from_addr,
        subject=subject,
        body=body,
        dry_run=dry_run,
    )
    if smtp_result["status"] not in {"NO_SMTP_HOST"}:
        return {"channel": "email", **smtp_result}
    sendmail_result = _send_email_sendmail(
        to_addr=to_addr,
        from_addr=from_addr,
        subject=subject,
        body=body,
        dry_run=dry_run,
    )
    return {"channel": "email", **sendmail_result}


def notify(
    *,
    subject: str,
    body: str,
    channels: list[str],
    dry_run: bool,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for channel in channels:
        if channel == "telegram":
            results[channel] = send_telegram(subject=subject, body=body, dry_run=dry_run)
        elif channel == "email":
            results[channel] = send_email(subject=subject, body=body, dry_run=dry_run)
        else:
            results[channel] = {"channel": channel, "status": "UNKNOWN_CHANNEL", "sent": False}
    all_sent = all(result.get("sent") for result in results.values()) if not dry_run else False
    all_configured = all(result.get("status") != "NOT_CONFIGURED" for result in results.values())
    if dry_run:
        status = "DRY_RUN_COMPLETE"
    elif all_sent:
        status = "SENT_ALL_REQUIRED_CHANNELS"
    elif not all_configured:
        status = "NOT_CONFIGURED"
    else:
        status = "FAILED"
    return {
        "schema_name": SCHEMA_NAME,
        "status": status,
        "dry_run": dry_run,
        "subject": subject,
        "body_chars": len(body),
        "channels": results,
        "sent_all_required_channels": all_sent,
        "configured_all_required_channels": all_configured,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    env_file_result: dict[str, Any] = {"path": args.env_file, "exists": False, "loaded_keys": []}
    if not args.no_env_file and args.env_file:
        env_file_result = load_env_file(Path(args.env_file), override=False)
    body = _read_body(args.body, args.body_file)
    channels = [channel.strip() for channel in args.channels.split(",") if channel.strip()]
    payload = notify(subject=args.subject, body=body, channels=channels, dry_run=bool(args.dry_run))
    payload["env_file"] = env_file_result
    if args.output_json:
        write_json(Path(args.output_json), payload)
    print(json.dumps(payload, sort_keys=True))
    if args.strict and not args.dry_run and not payload["sent_all_required_channels"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
