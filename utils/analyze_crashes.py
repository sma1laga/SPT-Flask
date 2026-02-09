#!/usr/bin/env python3
"""Analyze crashes.log JSONL files and print actionable summaries."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class CrashEntry:
    timestamp: datetime | None
    error_signature: str
    endpoint: str
    path: str
    method: str
    user_id: str
    client_ip: str


def _parse_timestamp(value: Any) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    normalized = value.replace('Z', '+00:00')
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def load_entries(log_path: Path) -> tuple[list[CrashEntry], int]:
    entries: list[CrashEntry] = []
    skipped = 0

    with log_path.open('r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            error_type = str(data.get('error_type') or '').strip()
            error_message = str(data.get('error_message') or '').strip()
            error_fallback = str(data.get('error') or '').strip()
            if error_type or error_message:
                signature = f'{error_type}: {error_message}'.strip(': ')
            else:
                signature = error_fallback or 'UnknownError'

            entries.append(
                CrashEntry(
                    timestamp=_parse_timestamp(data.get('timestamp')),
                    error_signature=signature,
                    endpoint=str(data.get('endpoint') or 'unknown'),
                    path=str(data.get('path') or 'unknown'),
                    method=str(data.get('method') or 'unknown'),
                    user_id=str(data.get('anonymous_user_id') or 'unknown'),
                    client_ip=str(data.get('client_ip') or 'unknown'),
                )
            )

    return entries, skipped


def _fmt_ts(ts: datetime | None) -> str:
    return ts.isoformat(timespec='seconds') if ts else 'unknown'


def build_report(entries: list[CrashEntry], skipped: int) -> str:
    if not entries:
        return 'No crash entries found.\n'

    entries_with_ts = [item for item in entries if item.timestamp is not None]
    first_seen = min((item.timestamp for item in entries_with_ts), default=None)
    last_seen = max((item.timestamp for item in entries_with_ts), default=None)

    by_error = Counter(item.error_signature for item in entries)
    by_endpoint = Counter(f'{item.method} {item.path} ({item.endpoint})' for item in entries)

    users_by_error: dict[str, set[str]] = defaultdict(set)
    ips_by_error: dict[str, set[str]] = defaultdict(set)
    for item in entries:
        users_by_error[item.error_signature].add(item.user_id)
        ips_by_error[item.error_signature].add(item.client_ip)

    repeated_errors = {err: count for err, count in by_error.items() if count > 1}

    lines = [
        'Crash Log Analysis',
        '==================',
        f'Total crash entries: {len(entries)}',
        f'Invalid/skipped lines: {skipped}',
        f'First seen crash: {_fmt_ts(first_seen)}',
        f'Last seen crash: {_fmt_ts(last_seen)}',
        f'Unique error signatures: {len(by_error)}',
        '',
        'Top errors:',
    ]

    for error, count in by_error.most_common(10):
        user_count = len(users_by_error[error])
        ip_count = len(ips_by_error[error])
        scope = 'single-user pattern' if user_count == 1 else 'multi-user pattern'
        lines.append(
            f'- {count}x | users={user_count}, ips={ip_count} ({scope}) | {error}'
        )

    lines.append('')
    lines.append('Top affected endpoints:')
    for endpoint, count in by_endpoint.most_common(10):
        lines.append(f'- {count}x | {endpoint}')

    lines.append('')
    lines.append('Repeated errors (occurred more than once):')
    if repeated_errors:
        for error, count in sorted(repeated_errors.items(), key=lambda item: item[1], reverse=True):
            lines.append(f'- {count}x | {error}')
    else:
        lines.append('- None (all recorded errors occurred once)')

    lines.append('')
    lines.append('Per-error user spread:')
    for error, count in by_error.most_common(10):
        lines.append(
            f'- {error} -> {len(users_by_error[error])} anonymous users, '
            f'{len(ips_by_error[error])} client IPs'
        )

    return '\n'.join(lines) + '\n'


def main() -> int:
    parser = argparse.ArgumentParser(description='Analyze crashes.log JSONL entries.')
    parser.add_argument(
        'logfile',
        nargs='?',
        default='../crashes.log',
        help='Path to JSONL crash log (default: crashes.log)',
    )
    args = parser.parse_args()

    log_path = Path(args.logfile)
    if not log_path.exists():
        print(f'Crash log not found: {log_path}')
        return 1

    entries, skipped = load_entries(log_path)
    print(build_report(entries, skipped), end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())