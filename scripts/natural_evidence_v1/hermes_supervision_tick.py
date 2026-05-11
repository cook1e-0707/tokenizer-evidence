from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.natural_evidence_v1.common import write_json


REPO_ROOT = Path(__file__).resolve().parents[2]
STATUS_DIR = REPO_ROOT / "results" / "natural_evidence_v1" / "status"
HERMES_REPORT_DIR = STATUS_DIR / "hermes_reports"
GATE_STATUS_JSON = STATUS_DIR / "gate_status.json"
PROJECT_EVENT_DIR = REPO_ROOT / ".agent_events"
PAUSE_FILE = PROJECT_EVENT_DIR / "auto.pause"
LOCK_DIR = STATUS_DIR / ".hermes_natural_evidence_tick.lock"
WORKER_LOG_DIR = STATUS_DIR / "hermes_worker_logs"
NOTIFY_SCRIPT = REPO_ROOT / "scripts" / "natural_evidence_v1" / "hermes_notify.py"
COMPACT_STATE_MD = REPO_ROOT / "docs" / "natural_evidence_v2" / "CURRENT_STATE.md"
CODEX_TIMEOUT_SECONDS = int(os.environ.get("HERMES_NAT_EV_CODEX_TIMEOUT_SECONDS", "900"))
STALE_LOCK_SECONDS = int(
    os.environ.get("HERMES_NAT_EV_STALE_LOCK_SECONDS", str(CODEX_TIMEOUT_SECONDS + 300))
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Hermes 15-minute supervision tick for natural_evidence_v1. "
            "Launch mode sends TG/email and starts a background Codex worker."
        )
    )
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--timestamp", default="")
    parser.add_argument("--report-md", default="")
    parser.add_argument("--start-notification-json", default="")
    parser.add_argument("--run-codex", default=os.environ.get("HERMES_NAT_EV_RUN_CODEX", "1"))
    return parser.parse_args(argv)


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def stamp_from_now() -> str:
    return utc_now().strftime("%Y%m%d_%H%M")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_gate_state() -> dict[str, Any]:
    payload = read_json(GATE_STATUS_JSON)
    return {
        "current_phase": payload.get("current_phase", "UNKNOWN"),
        "next_allowed_action": payload.get("next_allowed_action", "UNKNOWN"),
        "last_state_changing_action": payload.get("last_state_changing_action", "UNKNOWN"),
        "hermes_status": payload.get("hermes_15min_coordination", {}).get("status", "UNKNOWN"),
        "training_allowed": bool(payload.get("training_allowed", False)),
        "llama_allowed": bool(payload.get("llama_allowed", False)),
        "same_family_null_allowed": bool(payload.get("same_family_null_allowed", False)),
        "sanitizer_allowed": bool(payload.get("sanitizer_allowed", False)),
        "far_aggregation_allowed": bool(payload.get("far_aggregation_allowed", False)),
        "paper_claim_allowed": bool(payload.get("paper_claim_allowed", False)),
    }


def blocked_gate_controlled_actions(gate: dict[str, Any]) -> list[str]:
    action_flags = [
        ("training_allowed", "training"),
        ("llama_allowed", "Llama"),
        ("same_family_null_allowed", "same-family null"),
        ("sanitizer_allowed", "sanitizer benchmark"),
        ("far_aggregation_allowed", "FAR aggregation"),
        ("paper_claim_allowed", "paper-facing positive claims"),
    ]
    return [label for key, label in action_flags if not bool(gate.get(key, False))]


def blocked_gate_controlled_actions_text(gate: dict[str, Any]) -> str:
    blocked = blocked_gate_controlled_actions(gate)
    if not blocked:
        return "No gate-controlled action is locked by its boolean gate, but allowlist, notification, one-job, artifact, and claim-provenance constraints still apply."
    return "Gate-controlled and not yet unlocked: " + "; ".join(blocked) + "."


def r3_qwen_locked_scale_context(gate: dict[str, Any]) -> bool:
    phase = str(gate.get("current_phase", ""))
    next_action = str(gate.get("next_allowed_action", ""))
    return (
        phase == "V2_R3_QWEN_LOCKED_SCALE_ROUTE_APPROVED"
        and "R3.2 Qwen locked-scale" in next_action
    )


def forbidden_actions_text(gate: dict[str, Any]) -> str:
    blocked = blocked_gate_controlled_actions_text(gate)
    if r3_qwen_locked_scale_context(gate):
        return (
            f"{blocked} Hard-blocked in this phase: unreviewed or non-allowlisted generation; "
            "Qwen E2E outside the reviewed R3.2 locked-scale wrapper."
        )
    return (
        f"{blocked} Hard-blocked unless explicitly allowed by the current "
        "next_allowed_action: generation and Qwen E2E reruns."
    )


def red_flags_text(gate: dict[str, Any]) -> str:
    blocked = blocked_gate_controlled_actions_text(gate)
    if r3_qwen_locked_scale_context(gate):
        return (
            "R3.2 Qwen locked-scale generation/eval is permitted only after the "
            "full wrapper is reviewed, the allowlist entry is enabled for one job, "
            f"and TG/email notification succeeds. Gate-blocked actions: {blocked} "
            "Do not submit a plan-only wrapper as a full eval."
        )
    return (
        f"Gate-blocked actions: {blocked} Generation and Qwen E2E reruns also "
        "require explicit next_allowed_action support."
    )


def hard_constraints_text(gate: dict[str, Any]) -> str:
    gate_blocked = "\n".join(
        f"- gate-locked until its prerequisite gate passes: {action};"
        for action in blocked_gate_controlled_actions(gate)
    )
    if not gate_blocked:
        gate_blocked = "- no gate-controlled action is locked by its boolean gate, but all route-specific gates and allowlists still apply;"
    if r3_qwen_locked_scale_context(gate):
        return f"""{gate_blocked}
- R3.2 Qwen locked-scale generation/eval is allowed only through a reviewed full wrapper, one enabled allowlist entry, successful TG/email notification, and exactly one Chimera Slurm job;
- do not submit the current plan-only wrapper as a full eval;
- no Qwen E2E outside the reviewed R3.2 locked-scale route;
- any Chimera CPU/GPU work must use Slurm;
- do not run CPU work directly on the Chimera login node;
- do not overwrite existing artifacts."""
    return f"""{gate_blocked}
- no generation;
- no Qwen E2E rerun;
- any Chimera CPU/GPU work must use Slurm;
- do not run CPU work directly on the Chimera login node;
- do not overwrite existing artifacts."""


def run_monitor_command(command: list[str], *, timeout: int = 20) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return {
            "command": command,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip()[:4000],
            "stderr": proc.stderr.strip()[:2000],
        }
    except Exception as exc:
        return {"command": command, "returncode": None, "error": str(exc)}


def gather_monitor_snapshot() -> dict[str, Any]:
    return {
        "git_status_short": run_monitor_command(["git", "status", "--short"], timeout=15),
        "git_branch": run_monitor_command(["git", "branch", "--show-current"], timeout=15),
        "git_log": run_monitor_command(["git", "log", "-1", "--oneline"], timeout=15),
        "chimera_squeue": run_monitor_command(
            ["ssh", "chimera", "squeue -u $USER"],
            timeout=30,
        ),
        "chimera_sacct": run_monitor_command(
            [
                "ssh",
                "chimera",
                "sacct -u $USER --format=JobID,JobName,State,Elapsed,ExitCode -S now-2days",
            ],
            timeout=30,
        ),
    }


def write_report(
    *,
    report_md: Path,
    timestamp: str,
    gate: dict[str, Any],
    snapshot: dict[str, Any],
    state_changing_action: str,
    notification_json: Path,
) -> None:
    HERMES_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    active_jobs = snapshot.get("chimera_squeue", {}).get("stdout") or "No squeue rows captured."
    body = f"""# Hermes natural_evidence_v1 supervision tick

phase:
{gate["current_phase"]}

active_jobs:
```text
{active_jobs}
```

completed_jobs:
Latest known completed branch-aware score diagnostic remains job 848414 unless
Chimera monitoring above shows newer completions.

artifact_changes:
This report was generated by the Hermes scheduler tick before Codex execution.

gate_changes:
No gate changes before Codex execution.

red_flags:
{red_flags_text(gate)}

telegram_notification:
Required and attempted by `scripts/natural_evidence_v1/hermes_notify.py`.

email_notification:
Required and attempted by `scripts/natural_evidence_v1/hermes_notify.py`.

notification_json:
`{notification_json.relative_to(REPO_ROOT)}`

state_changing_action:
{state_changing_action}

next_allowed_action:
{gate["next_allowed_action"]}

gate_controlled_actions_not_yet_unlocked:
{forbidden_actions_text(gate)}

timestamp_utc:
{timestamp}
"""
    report_md.write_text(body, encoding="utf-8")


def send_notification(*, subject: str, report_md: Path, output_json: Path) -> int:
    cmd = [
        sys.executable,
        str(NOTIFY_SCRIPT),
        "--subject",
        subject,
        "--body-file",
        str(report_md),
        "--channels",
        "telegram,email",
        "--strict",
        "--output-json",
        str(output_json),
    ]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        failure = {
            "schema_name": "natural_evidence_v1_hermes_tick_notification_failure_v1",
            "status": "NOTIFICATION_FAILED",
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "command": cmd,
        }
        write_json(output_json.with_suffix(".failure.json"), failure)
    return proc.returncode


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _lock_age_seconds() -> float:
    try:
        return utc_now().timestamp() - LOCK_DIR.stat().st_mtime
    except FileNotFoundError:
        return 0.0


def _read_lock_pid() -> int | None:
    pid_path = LOCK_DIR / "worker_pid.txt"
    if not pid_path.exists():
        return None
    try:
        return int(pid_path.read_text(encoding="utf-8").strip())
    except ValueError:
        return None


def _lock_is_stale() -> tuple[bool, str]:
    if not LOCK_DIR.exists():
        return False, "no_lock"
    pid = _read_lock_pid()
    age_seconds = _lock_age_seconds()
    if pid is not None and not _pid_is_alive(pid):
        return True, f"worker_pid_not_alive:{pid}"
    if age_seconds > STALE_LOCK_SECONDS:
        return True, f"lock_age_seconds>{STALE_LOCK_SECONDS}:{age_seconds:.0f}"
    return False, f"active_pid={pid};age_seconds={age_seconds:.0f}"


def acquire_lock(timestamp: str) -> tuple[bool, str]:
    stale, reason = _lock_is_stale()
    if stale:
        release_lock()
    try:
        LOCK_DIR.mkdir(parents=True)
    except FileExistsError:
        return False, reason
    (LOCK_DIR / "tick_stamp.txt").write_text(timestamp + "\n", encoding="utf-8")
    (LOCK_DIR / "started_at_utc.txt").write_text(
        utc_now().isoformat().replace("+00:00", "Z") + "\n",
        encoding="utf-8",
    )
    (LOCK_DIR / "started_at.txt").write_text(timestamp + "\n", encoding="utf-8")
    return True, "acquired_after_stale_cleanup" if stale else "acquired"


def release_lock() -> None:
    if LOCK_DIR.exists():
        for child in LOCK_DIR.iterdir():
            child.unlink()
        LOCK_DIR.rmdir()


def launch_worker(*, timestamp: str, report_md: Path, notification_json: Path, run_codex: bool) -> Path:
    WORKER_LOG_DIR.mkdir(parents=True, exist_ok=True)
    worker_log = WORKER_LOG_DIR / f"{timestamp}_worker.log"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--timestamp",
        timestamp,
        "--report-md",
        str(report_md),
        "--start-notification-json",
        str(notification_json),
        "--run-codex",
        "1" if run_codex else "0",
    ]
    with worker_log.open("ab") as handle:
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    if LOCK_DIR.exists():
        (LOCK_DIR / "worker_pid.txt").write_text(str(proc.pid) + "\n", encoding="utf-8")
        write_json(
            LOCK_DIR / "worker_command.json",
            {
                "schema_name": "natural_evidence_v1_hermes_worker_command_v1",
                "pid": proc.pid,
                "timestamp": timestamp,
                "worker_log": str(worker_log.relative_to(REPO_ROOT)),
                "run_codex": run_codex,
                "cmd": cmd,
            },
        )
    return worker_log


def update_gate_after_launch(*, timestamp: str, report_md: Path, notification_json: Path, worker_log: Path) -> None:
    data = read_json(GATE_STATUS_JSON)
    coord = data.setdefault("hermes_15min_coordination", {})
    coord.update(
        {
            "last_scheduler_tick": {
                "time": utc_now().isoformat().replace("+00:00", "Z"),
                "report_md": str(report_md.relative_to(REPO_ROOT)),
                "notification_json": str(notification_json.relative_to(REPO_ROOT)),
                "worker_log": str(worker_log.relative_to(REPO_ROOT)),
                "background_worker_launched": True,
            },
            "status": "ACTIVE_SUPERVISION_PROTOCOL_SCHEDULED_TICK_LAUNCHED",
        }
    )
    data["last_checked_time"] = utc_now().isoformat().replace("+00:00", "Z")
    data["last_state_changing_action"] = (
        "Hermes scheduled tick sent Telegram/email and launched a background Codex worker. "
        "Training, generation, E2E rerun, Llama, same-family null, sanitizer, FAR, and paper claim gates remained locked."
    )
    write_json(GATE_STATUS_JSON, data)


def launch_mode(args: argparse.Namespace) -> int:
    timestamp = args.timestamp or stamp_from_now()
    report_md = HERMES_REPORT_DIR / f"{timestamp}_scheduled_tick.md"
    notification_json = HERMES_REPORT_DIR / f"{timestamp}_scheduled_tick_notification.json"
    gate = read_gate_state()
    snapshot = gather_monitor_snapshot()
    run_codex = str(args.run_codex).lower() not in {"0", "false", "no"}

    if PAUSE_FILE.exists():
        paused_report = HERMES_REPORT_DIR / f"{timestamp}_scheduled_tick_paused.md"
        paused_at = PAUSE_FILE.read_text(encoding="utf-8", errors="replace").strip()
        write_report(
            report_md=paused_report,
            timestamp=timestamp,
            gate=gate,
            snapshot=snapshot,
            state_changing_action=(
                "No Codex worker launched because project auto-progress is paused. "
                f"Pause file: {PAUSE_FILE}; paused_at={paused_at or 'unknown'}."
            ),
            notification_json=notification_json,
        )
        return send_notification(
            subject="Hermes natural_evidence_v1 tick paused",
            report_md=paused_report,
            output_json=notification_json,
        )

    acquired, lock_reason = acquire_lock(timestamp)
    if not acquired:
        blocked_report = HERMES_REPORT_DIR / f"{timestamp}_scheduled_tick_blocked.md"
        write_report(
            report_md=blocked_report,
            timestamp=timestamp,
            gate=gate,
            snapshot=snapshot,
            state_changing_action=(
                "No new Codex worker launched; previous Hermes worker lock is still active. "
                f"Lock state: {lock_reason}."
            ),
            notification_json=notification_json,
        )
        return send_notification(
            subject="Hermes natural_evidence_v1 tick blocked: previous worker still active",
            report_md=blocked_report,
            output_json=notification_json,
        )

    write_report(
        report_md=report_md,
        timestamp=timestamp,
        gate=gate,
        snapshot=snapshot,
        state_changing_action=(
            "Telegram/email notification sent before launching one background Codex worker."
        ),
        notification_json=notification_json,
    )
    notify_status = send_notification(
        subject="Hermes natural_evidence_v1 progress tick",
        report_md=report_md,
        output_json=notification_json,
    )
    if notify_status != 0:
        release_lock()
        return notify_status

    worker_log = launch_worker(
        timestamp=timestamp,
        report_md=report_md,
        notification_json=notification_json,
        run_codex=run_codex,
    )
    update_gate_after_launch(
        timestamp=timestamp,
        report_md=report_md,
        notification_json=notification_json,
        worker_log=worker_log,
    )
    print(f"launched natural_evidence_v1 Hermes worker: {worker_log}")
    return 0


def codex_prompt(gate: dict[str, Any], report_md: Path) -> str:
    return f"""Follow AGENTS.md and the natural_evidence_v1 project guardrails strictly.

You are being invoked by the Hermes 15-minute supervisor for tokenizer_alignment.
Hermes already sent the required Telegram and email progress notification before
this Codex worker started.

Current phase:
{gate["current_phase"]}

Next allowed action:
{gate["next_allowed_action"]}

Current Hermes report:
{report_md}

Do exactly one small allowed project-advancing action. Treat the "Next allowed
action" above as the controlling action for this tick. Do not continue stale v1
repaired target-mass probes. Start by reading:
- docs/natural_evidence_v2/CURRENT_STATE.md
- results/natural_evidence_v1/status/gate_status.json
- results/natural_evidence_v2/status/gate_status.json
Consult the long historical files only if the compact state is ambiguous:
- docs/natural_evidence_v1/AUTOMATION_STATE.md
- docs/natural_evidence_v1/next_step_codex_plan.md
- docs/natural_evidence_v2/PROTOCOL_CONTRACT.md
- docs/natural_evidence_v2/CLAIM_GUARDRAILS.md

Hard constraints:
{hard_constraints_text(gate)}

If the next action is not safe or not unambiguous, write a blocker report and
stop. If you make a state change, update the relevant docs/status artifacts and
validate with the smallest relevant checks.
"""


def find_codex_binary() -> str:
    for env_name in ("HERMES_CODEX_BIN", "CODEX_BIN"):
        configured = os.environ.get(env_name, "").strip()
        if configured:
            candidate = Path(configured).expanduser()
            if candidate.exists() and os.access(candidate, os.X_OK):
                return str(candidate)
    discovered = shutil.which("codex")
    if discovered:
        return discovered
    candidates = [
        Path.home() / ".local" / "bin" / "codex",
        Path("/opt/homebrew/bin/codex"),
        Path("/usr/local/bin/codex"),
    ]
    extension_root = Path.home() / ".vscode" / "extensions"
    if extension_root.exists():
        candidates.extend(
            sorted(
                extension_root.glob("openai.chatgpt-*/bin/macos-aarch64/codex"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        )
    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return ""


def send_completion_notification(*, timestamp: str, codex_report: Path, success: bool) -> None:
    completion_report = HERMES_REPORT_DIR / f"{timestamp}_scheduled_tick_completion.md"
    notification_json = HERMES_REPORT_DIR / f"{timestamp}_scheduled_tick_completion_notification.json"
    summary = codex_report.read_text(encoding="utf-8", errors="replace")[:6000] if codex_report.exists() else ""
    status = "completed" if success else "failed"
    completion_report.write_text(
        f"""# Hermes natural_evidence_v1 Codex worker {status}

phase:
{read_gate_state()["current_phase"]}

codex_report:
`{codex_report.relative_to(REPO_ROOT) if codex_report.exists() else codex_report}`

summary:
```text
{summary}
```

next_allowed_action:
{read_gate_state()["next_allowed_action"]}

gate_controlled_actions_not_yet_unlocked:
{forbidden_actions_text(read_gate_state())}
""",
        encoding="utf-8",
    )
    send_notification(
        subject=f"Hermes natural_evidence_v1 Codex worker {status}",
        report_md=completion_report,
        output_json=notification_json,
    )


def worker_mode(args: argparse.Namespace) -> int:
    timestamp = args.timestamp or stamp_from_now()
    report_md = Path(args.report_md) if args.report_md else HERMES_REPORT_DIR / f"{timestamp}_scheduled_tick.md"
    codex_report = HERMES_REPORT_DIR / f"{timestamp}_scheduled_tick_codex_report.md"
    gate = read_gate_state()
    run_codex = str(args.run_codex).lower() not in {"0", "false", "no"}
    try:
        if not run_codex:
            codex_report.write_text(
                "Codex worker launch was tested with HERMES_NAT_EV_RUN_CODEX=0; no project action executed.\n",
                encoding="utf-8",
            )
            send_completion_notification(timestamp=timestamp, codex_report=codex_report, success=True)
            return 0
        codex_bin = find_codex_binary()
        if not codex_bin:
            codex_report.write_text("codex CLI not found on PATH.\n", encoding="utf-8")
            send_completion_notification(timestamp=timestamp, codex_report=codex_report, success=False)
            return 1
        cmd = [
            codex_bin,
            "exec",
            "--dangerously-bypass-approvals-and-sandbox",
            "--cd",
            str(REPO_ROOT),
            "--output-last-message",
            str(codex_report),
            "-",
        ]
        proc = subprocess.run(
            cmd,
            input=codex_prompt(gate, report_md),
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=CODEX_TIMEOUT_SECONDS,
            check=False,
        )
        transcript = HERMES_REPORT_DIR / f"{timestamp}_scheduled_tick_codex_transcript.log"
        transcript.write_text(proc.stdout, encoding="utf-8", errors="replace")
        success = proc.returncode == 0
        if not codex_report.exists():
            codex_report.write_text(proc.stdout[-6000:], encoding="utf-8", errors="replace")
        send_completion_notification(timestamp=timestamp, codex_report=codex_report, success=success)
        return proc.returncode
    except Exception as exc:
        codex_report.write_text(f"Codex worker exception: {exc}\n", encoding="utf-8")
        send_completion_notification(timestamp=timestamp, codex_report=codex_report, success=False)
        return 1
    finally:
        release_lock()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker:
        return worker_mode(args)
    return launch_mode(args)


if __name__ == "__main__":
    raise SystemExit(main())
