"""Read-only release audit. Reports redact all matched private values.

python scripts/privacy_audit.py --report ../release-audit.json
python scripts/privacy_audit.py --git-index --git-history --report ../git-audit.json

An optional private denylist JSON has ``literals`` and ``patterns`` arrays.
Keep names and experiment map seeds in that private file, outside this repo.
This is a pattern/structure audit, not a guarantee against encoded disclosure.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import struct
import subprocess
import sys
import zipfile
import zlib

EXCLUDED = {'.git', '__pycache__', '.venv', '.pytest_cache', 'node_modules'}
PRIVATE_PARTS = {'private_reference_do_not_publish', 'private_delivery', 'private_handoff',
                 'private_truth', 'private_course', 'source_text', 'experiment_truth'}
PRIVATE_KEYS = {'map_seed', 'map_seeds', 'seeds', 'mines', 'mine_map', 'hidden_mines',
                'private_snapshot', 'future_results', 'future_outcomes'}
FONT_SUFFIXES = {'.ttf', '.ttc', '.otf', '.woff', '.woff2'}
MEDIA_SUFFIXES = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.mp4', '.webm'}
TEXT_SUFFIXES = {'.py', '.js', '.json', '.jsonl', '.md', '.txt', '.html', '.css', '.svg',
                 '.yml', '.yaml', '.toml', '.ini', '.bat', '.ps1', '.csv', '.xml', '.sh'}
PATTERNS = {
    'secret_token': r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,}|sk-[A-Za-z0-9_-]{24,}|hf_[A-Za-z0-9]{24,}|AKIA[A-Z0-9]{16})',
    'private_key': r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    'phone_number': r'(?<![\w.])1[3-9][0-9]{9}(?![\w.])',
    'student_id': r'(?:学号|student[_ -]?id)[\s"\x27]*[:：=][\s"\x27]*[A-Za-z]?[0-9]{6,18}',
    'personal_absolute_path': r'(?i)(?:\b[A-Z]:[\\/](?:Users|Downloads|Codex)[\\/]|/(?:Users|home)/[^/\s]+/)',
}
EMAIL = re.compile(r'[A-Za-z0-9_.+%-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')
LIMIT = 128 * 1024 * 1024


class Audit:
    def __init__(self, root, denylist=None, map_manifest=None):
        self.root = Path(root).resolve()
        self.findings, self.files, self.manual_media, self.array_metadata = [], [], [], []
        self.scopes = ['worktree']
        self.git_counts = {'index_entries': 0, 'history_blobs': 0, 'commits': 0}
        self.parsed_records = 0
        self.patterns = [(name, re.compile(pattern)) for name, pattern in PATTERNS.items()]
        self.literals = []
        self.private_map_seeds = set()
        self.private_map_seed_text = set()
        self.map_seed_pattern = None
        if denylist:
            source = json.loads(Path(denylist).read_text(encoding='utf-8-sig'))
            self.literals = source.get('literals', [])
            self.patterns.extend(('private_deny_pattern', re.compile(p)) for p in source.get('patterns', []))
        if map_manifest:
            manifests = map_manifest if isinstance(map_manifest, (list, tuple)) else [map_manifest]
            def collect_seeds(node):
                if isinstance(node, dict):
                    for child in node.values():
                        collect_seeds(child)
                elif isinstance(node, list):
                    for seed in node:
                        if type(seed) is not int:
                            raise ValueError('Private map split must contain integer seeds')
                        self.private_map_seeds.add(seed)
                else:
                    raise ValueError('Unsupported private map split structure')
            for manifest in manifests:
                source = json.loads(Path(manifest).read_text(encoding='utf-8-sig'))
                collect_seeds(source['splits'])
            self.private_map_seed_text = {str(seed) for seed in self.private_map_seeds}
            lengths = [len(seed) for seed in self.private_map_seed_text]
            if lengths:
                self.map_seed_pattern = re.compile(r'(?<![0-9])[0-9]{' + str(min(lengths)) + ',' + str(max(lengths)) + r'}(?![0-9])')

    def finding(self, location, category):
        value = {'location': location, 'category': category}
        if value not in self.findings:
            self.findings.append(value)

    def text(self, label, value, structured=False):
        if self.map_seed_pattern and any(m.group() in self.private_map_seed_text
                                         for m in self.map_seed_pattern.finditer(value)):
            self.finding(label, 'private_map_seed_value')
        if structured:
            # Exact model-count integers can coincidentally resemble a phone.
            # Only this documented numeric solver field is exempted; a string
            # in the same field or any phone/contact field remains inspected.
            value = re.sub(r'("models"\s*:\s*)[0-9]+', r'\g<1>0', value)
        for literal in self.literals:
            if literal and literal in value:
                self.finding(label, 'private_deny_literal')
        for category, pattern in self.patterns:
            match = pattern.search(value)
            if match:
                line = value.count('\n', 0, match.start()) + 1
                self.finding(f'{label}:line{line}', category)
        for match in EMAIL.finditer(value):
            if not match.group().lower().endswith('@users.noreply.github.com'):
                self.finding(label, 'email_address')
                break

    def filename(self, label):
        self.text(label, label)
        parts = {p.lower() for p in label.replace('!', '/').split('/')}
        if PRIVATE_PARTS & parts:
            self.finding(label, 'private_directory')
        if Path(label).suffix.lower() in FONT_SUFFIXES:
            self.finding(label, 'bundled_font')
        if Path(label).name.lower() in {'map_manifest.json', '.env', 'id_rsa', 'id_ed25519'}:
            self.finding(label, 'private_file_name')
        if Path(label).suffix.lower() in {'.doc', '.docx', '.ppt', '.pptx', '.pdf'}:
            self.finding(label, 'document_requires_explicit_release_review')

    def structure(self, label, value, trail='$', optimizer_seed=False):
        if isinstance(value, dict):
            for key, item in value.items():
                normalized = str(key).lower()
                if normalized in PRIVATE_KEYS or (normalized == 'seed' and not optimizer_seed):
                    self.finding(f'{label}:{trail}.{key}', 'private_metadata_key')
                self.structure(label, item, f'{trail}.{key}', optimizer_seed)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                self.structure(label, item, f'{trail}[{i}]', optimizer_seed)

    def json_record(self, label, text):
        try:
            value = json.loads(text)
        except (json.JSONDecodeError, UnicodeDecodeError):
            self.finding(label, 'invalid_json')
            return
        optimizer_seed = (label.replace('\\', '/').startswith('models/') and
                          label.endswith('_training.json') and isinstance(value, dict) and
                          value.get('method') == 'supervised behavior cloning')
        self.structure(label, value, optimizer_seed=optimizer_seed)
        self.parsed_records += 1

    def content(self, label, raw, depth=0):
        self.filename(label)
        if depth > 3:
            self.finding(label, 'archive_recursion_limit')
            return
        suffix = Path(label).suffix.lower()
        if suffix in {'.zip', '.npz', '.docx', '.pptx', '.xlsx'}:
            try:
                with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                    for info in archive.infolist():
                        if info.is_dir():
                            continue
                        member = label + '!' + info.filename
                        self.filename(member)
                        if info.flag_bits & 1 or info.file_size > LIMIT:
                            self.finding(member, 'archive_member_not_inspected')
                        else:
                            self.content(member, archive.read(info), depth + 1)
            except (ValueError, zipfile.BadZipFile, RuntimeError):
                self.finding(label, 'unreadable_archive')
            return
        if suffix == '.gz':
            try:
                with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
                    for number, line in enumerate(stream, 1):
                        if len(line) > LIMIT:
                            self.finding(label, 'oversized_gzip_record')
                            continue
                        record_label = f'{label}!record{number}'
                        text = line.decode('utf-8-sig')
                        self.text(record_label, text, structured=True)
                        self.json_record(record_label, text)
            except (OSError, EOFError, UnicodeDecodeError):
                self.finding(label, 'invalid_gzip_or_text')
            return
        if suffix == '.npy':
            try:
                import numpy as np
                array = np.load(io.BytesIO(raw), allow_pickle=False)
                key = Path(label.split('!')[-1]).stem.lower()
                if key in PRIVATE_KEYS or key == 'seed':
                    self.finding(label, 'private_numpy_array')
                self.array_metadata.append({'location': label, 'shape': list(array.shape), 'dtype': str(array.dtype)})
                if array.dtype.kind in {'i', 'u'} and array.size and self.private_map_seeds:
                    candidates = [seed for seed in self.private_map_seeds if int(array.min()) <= seed <= int(array.max())]
                    if candidates and np.isin(array, candidates).any():
                        self.finding(label, 'private_map_seed_in_numeric_array')
                if array.dtype.names:
                    for field in array.dtype.names:
                        if field.lower() in PRIVATE_KEYS or field.lower() == 'seed':
                            self.finding(label, 'private_numpy_field')
                if array.dtype.kind in {'U', 'S'}:
                    for item in array.flat:
                        self.text(label, str(item))
            except (ValueError, ImportError, OSError):
                self.finding(label, 'numpy_array_not_safely_inspected')
            return
        if suffix in MEDIA_SUFFIXES:
            self.manual_media.append(label)
            if suffix == '.png' and raw.startswith(b'\x89PNG\r\n\x1a\n'):
                offset = 8
                while offset + 12 <= len(raw):
                    size = struct.unpack('>I', raw[offset:offset + 4])[0]
                    kind = raw[offset + 4:offset + 8]
                    payload = raw[offset + 8:offset + 8 + size]
                    if kind in {b'tEXt', b'iTXt', b'eXIf'}:
                        self.text(label + '!metadata', payload.decode('utf-8', errors='replace'))
                    elif kind == b'zTXt':
                        try:
                            compressed = payload.split(b'\0', 1)[1][1:]
                            self.text(label + '!metadata', zlib.decompress(compressed).decode('utf-8', errors='replace'))
                        except (IndexError, zlib.error):
                            self.finding(label, 'png_metadata_not_inspected')
                    offset += size + 12
            return
        if suffix in TEXT_SUFFIXES or suffix == '' or Path(label).name == '.gitignore':
            try:
                text = raw.decode('utf-8-sig')
            except UnicodeDecodeError:
                self.finding(label, 'unexpected_non_utf8_text')
                return
            self.text(label, text, structured=suffix in {'.json', '.jsonl'})
            if suffix == '.json':
                self.json_record(label, text)
            elif suffix == '.jsonl':
                for i, line in enumerate(text.splitlines(), 1):
                    if line.strip():
                        self.json_record(f'{label}!record{i}', line)
            elif suffix == '.csv':
                for row in csv.DictReader(io.StringIO(text)):
                    self.structure(label, row)
                    self.parsed_records += 1
        else:
            self.finding(label, 'unrecognized_binary_requires_review')

    def blob(self, label, raw, origin):
        self.files.append({'path': label, 'origin': origin, 'bytes': len(raw),
                           'sha256': hashlib.sha256(raw).hexdigest()})
        self.content(label, raw)

    def worktree(self):
        for path in sorted(self.root.rglob('*')):
            relative = path.relative_to(self.root)
            if any(part in EXCLUDED for part in relative.parts) or path.suffix in {'.pyc', '.pyo'}:
                continue
            if path.is_symlink():
                self.finding(relative.as_posix(), 'symlink_requires_review')
            elif path.is_file():
                before = path.stat()
                self.blob(relative.as_posix(), path.read_bytes(), 'worktree')
                if path.stat().st_mtime_ns != before.st_mtime_ns:
                    self.finding(relative.as_posix(), 'changed_during_audit')

    def git(self, index=False, history=False):
        def run(*args):
            return subprocess.check_output(['git', '-C', str(self.root), *args], stderr=subprocess.PIPE)
        try:
            if index:
                self.scopes.append('git-index')
                for row in run('ls-files', '--stage', '-z').split(b'\0'):
                    if not row:
                        continue
                    metadata, filename = row.split(b'\t', 1)
                    self.git_counts['index_entries'] += 1
                    mode, sha, stage = metadata.decode().split()
                    label = filename.decode('utf-8')
                    if mode == '160000':
                        self.finding(label, 'submodule_requires_review')
                    else:
                        self.blob(label, run('cat-file', 'blob', sha), 'git-index')
            if history:
                self.scopes.append('git-history')
                self.git_counts['commits'] = len(run('rev-list', '--all').splitlines())
                for row in run('rev-list', '--objects', '--all').decode('utf-8').splitlines():
                    sha, _, label = row.partition(' ')
                    if run('cat-file', '-t', sha).strip() == b'blob':
                        self.git_counts['history_blobs'] += 1
                        self.blob(label, run('cat-file', 'blob', sha), 'git-history')
                self.text('git-commit-identities', run('log', '--all', '--format=%an <%ae>%n%cn <%ce>%n%B').decode('utf-8'))
        except (subprocess.CalledProcessError, OSError, UnicodeDecodeError):
            self.finding('git', 'requested_git_scope_not_inspected')

    def report(self):
        return {'schema': 'arena-release-audit-v1', 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                'passed_automated_checks': not self.findings,
                'scope_note': 'Read-only pattern/structure audit, not a guarantee against all disclosure. Images need visual review.',
                'matched_values_redacted': True, 'files_scanned': len(self.files),
                'scopes': self.scopes, 'git_counts': self.git_counts,
                'private_map_seed_values_checked': len(self.private_map_seeds),
                'structured_records': self.parsed_records, 'findings': self.findings,
                'media_for_visual_review': sorted(set(self.manual_media)),
                'numpy_metadata': self.array_metadata, 'files': self.files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--denylist', type=Path)
    parser.add_argument('--map-manifest', type=Path, action='append', help='Private frozen manifest; repeat for legacy and challenge maps')
    parser.add_argument('--git-index', action='store_true')
    parser.add_argument('--git-history', action='store_true')
    args = parser.parse_args()
    if args.report.resolve().is_relative_to(args.root.resolve()):
        parser.error('Audit report must be outside the public repository')
    if args.denylist and args.denylist.resolve().is_relative_to(args.root.resolve()):
        parser.error('Private denylist must be outside the public repository')
    if args.map_manifest and any(path.resolve().is_relative_to(args.root.resolve()) for path in args.map_manifest):
        parser.error('Private map manifest must be outside the public repository')
    audit = Audit(args.root, args.denylist, args.map_manifest)
    audit.worktree()
    audit.git(args.git_index, args.git_history)
    result = audit.report()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('passed_automated_checks', 'files_scanned',
                                                 'structured_records')} |
                     {'finding_count': len(result['findings']),
                      'first_findings': result['findings'][:10]}, ensure_ascii=False))
    return 0 if result['passed_automated_checks'] else 1


if __name__ == '__main__':
    sys.exit(main())
