"""Retrieve only the pinned final-v2 files and model tensors, using HTTP ranges.

External source/data/model/tokenizer files are ignored by Git. A local full TAR
can be used instead of HTTP with --local-archive. No training code is imported.
"""
import argparse
import hashlib
import html
import io
import json
import pickletools
import re
import struct
import time
import zipfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARCHIVE_SIZE = 63775365120
DRIVE_ID = '1j6cuNaGKZmvNsbB1A489K0PG6pLj8svY'

class ArtifactReader:
    def __init__(self, local=None, url=None):
        self.local = Path(local) if local else None
        self.url = url
        if self.local:
            if self.local.stat().st_size != ARCHIVE_SIZE:
                raise ValueError('Expected the original uncompressed final-v2 TAR (63,775,365,120 bytes)')
        else:
            import requests
            self.session = requests.Session()
            if not self.url:
                base = 'https://drive.usercontent.google.com/download'
                params = {'id': DRIVE_ID, 'export': 'download'}
                with self.session.get(base, params=params, headers={'Range': 'bytes=0-1023'},
                                      timeout=(10, 30), stream=True) as r:
                    r.raise_for_status()
                    if r.status_code == 206:
                        self.url = r.url
                    else:
                        body = bytearray()
                        for block in r.iter_content(8192):
                            body.extend(block)
                            if len(body) > 262144:
                                raise ValueError('Unexpected Google Drive confirmation response')
                        fields = dict(re.findall(r'name="([^"]+)" value="([^"]*)"', body.decode('utf-8')))
                        params.update({k: html.unescape(v) for k, v in fields.items()})
                        req = requests.Request('GET', base, params=params).prepare()
                        self.url = req.url
            # Stop before downloading anything if the source no longer serves this TAR.
            self.read(0, 512)

    def read(self, start, size):
        if size < 0 or start < 0 or start+size > ARCHIVE_SIZE:
            raise ValueError('Range outside the pinned TAR')
        if self.local:
            with self.local.open('rb') as f:
                f.seek(start)
                data = f.read(size)
            if len(data) != size:
                raise ValueError('Short local read')
            return data
        for attempt in range(6):
            try:
                with self.session.get(self.url, headers={'Range': f'bytes={start}-{start+size-1}'},
                                      timeout=(10, 30), stream=True) as r:
                    r.raise_for_status()
                    expected = f'bytes {start}-{start+size-1}/{ARCHIVE_SIZE}'
                    if r.status_code != 206 or r.headers.get('Content-Range') != expected:
                        raise ValueError('Source does not serve the pinned byte range; no full download attempted')
                    data = r.content
                if len(data) != size:
                    raise ValueError('Short range response')
                return data
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(2)

class RemoteZip(io.RawIOBase):
    def __init__(self, reader, offset, size):
        self.reader, self.offset, self.size, self.pos = reader, offset, size, 0
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self, pos, whence=0):
        self.pos = pos if whence == 0 else self.pos+pos if whence == 1 else self.size+pos
        return self.pos
    def read(self, size=-1):
        size = self.size-self.pos if size < 0 else min(size, self.size-self.pos)
        if not size: return b''
        data = self.reader.read(self.offset+self.pos, size)
        self.pos += size
        return data

def sha_file(path):
    with path.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()

def copy_range(reader, offset, size, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(destination.name+'.part')
    done = part.stat().st_size if part.exists() else 0
    if done > size: raise ValueError('Partial file exceeds expected size')
    with part.open('ab') as f:
        while done < size:
            block = reader.read(offset+done, min(4*1024*1024, size-done))
            f.write(block)
            done += len(block)
    part.replace(destination)

def prepare_files(reader):
    files = json.loads((ROOT / 'provenance/artifact_files.json').read_text())
    for name, entry in files.items():
        destination = ROOT / 'author_v2' / name if name.startswith('src/') else ROOT / 'artifact_v2' / name
        if not destination.exists() or sha_file(destination) != entry['sha256']:
            copy_range(reader, entry['offset'], entry['size'], destination)
        if sha_file(destination) != entry['sha256']:
            raise ValueError('Pinned artifact SHA-256 mismatch: '+name)
        print('Verified', name, flush=True)
    (ROOT / 'artifact_v2_manifest.json').write_text(json.dumps(files, indent=2)+'\n', encoding='utf-8')

def prepare_tokenizer():
    import requests
    entries = json.loads((ROOT / 'provenance/tokenizer_manifest.json').read_text())
    for entry in entries:
        path = ROOT / 'tokenizer' / entry['file']
        path.parent.mkdir(exist_ok=True)
        if not path.exists() or sha_file(path) != entry['sha256']:
            r = requests.get(entry['source'], timeout=(10, 60))
            r.raise_for_status()
            if hashlib.sha256(r.content).hexdigest() != entry['sha256']:
                raise ValueError('Tokenizer source changed: '+entry['file'])
            path.write_bytes(r.content)
        print('Verified tokenizer', entry['file'], flush=True)

def prepare_checkpoint(reader):
    meta = json.loads((ROOT / 'provenance/checkpoint_index.json').read_text())
    baseline = json.loads((ROOT / 'provenance/checkpoint_manifest.json').read_text())
    path = ROOT / baseline['inference_checkpoint']
    manifest_path = ROOT / 'models/v2_suite_checkpoint_manifest.json'
    if path.exists() and manifest_path.exists():
        local = json.loads(manifest_path.read_text())
        if sha_file(path) == local['sha256'] and local['globals'] == baseline['globals']:
            print('Verified existing checkpoint', flush=True)
            return
    entry = meta['entry']
    with zipfile.ZipFile(RemoteZip(reader, entry['offset'], entry['size'])) as archive:
        actual = [{'name':v.filename,'size':v.file_size,'offset':v.header_offset,
                   'compression':v.compress_type,'crc':v.CRC} for v in archive.infolist()]
        if actual != meta['records']:
            raise ValueError('Checkpoint ZIP index changed')
        data = archive.read(meta['pickle_name'])
    cuts = [pos for op, arg, pos in pickletools.genops(data) if op.name == 'BINUNICODE' and arg == 'optimizer']
    if len(cuts) != 1 or data[:2] != b'\x80\x02':
        raise ValueError('Unexpected checkpoint dictionary layout')
    model_pickle = data[:cuts[0]]+b'u.'
    if hashlib.sha256(model_pickle).hexdigest() != meta['model_pickle_sha256']:
        raise ValueError('Model pickle differs from inspected final-v2 metadata')
    keys = set(meta['storage_keys'])
    records = [v for v in meta['records'] if '/data/' in v['name'] and v['name'].split('/')[-1] in keys]
    small = [v for v in meta['records'] if '/data/' not in v['name'] and not v['name'].endswith('/data.pkl')]
    if len(records) != baseline['model_tensor_records']:
        raise ValueError('Unexpected model tensor count')
    path.parent.mkdir(exist_ok=True)
    part = path.with_suffix('.part')
    with zipfile.ZipFile(part, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as out:
        out.writestr(zipfile.ZipInfo(meta['pickle_name'], (2024, 1, 1, 0, 0, 0)), model_pickle)
        for i, record in enumerate(records+small):
            local = ROOT / 'work/checkpoint_parts' / record['name'].replace('/', '_')
            if not local.exists() or local.stat().st_size != record['size']:
                head = reader.read(entry['offset']+record['offset'], 30)
                values = struct.unpack('<IHHHHHIIIHH', head)
                if values[0] != 0x04034b50 or record['compression'] != 0:
                    raise ValueError('Unsupported checkpoint ZIP record')
                offset = entry['offset']+record['offset']+30+values[-2]+values[-1]
                copy_range(reader, offset, record['size'], local)
            data = local.read_bytes()
            if zlib.crc32(data) != record['crc']:
                raise ValueError('Original tensor CRC mismatch: '+record['name'])
            out.writestr(zipfile.ZipInfo(record['name'], (2024, 1, 1, 0, 0, 0)), data)
            if i % 20 == 0: print('Verified checkpoint records', i+1, '/', len(records+small), flush=True)
    part.replace(path)
    manifest = dict(baseline, sha256=sha_file(path),
                    verification='All pinned original tensor ZIP CRC32 values and model-pickle SHA256 verified; optimizer omitted; deterministic ZIP metadata')
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print('Prepared model-only checkpoint', manifest['sha256'], flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local-archive', help='Path to the original final-v2 TAR; avoids Google Drive')
    parser.add_argument('--artifact-url', help='Temporary direct URL for the same pinned final-v2 TAR')
    parser.add_argument('--assets-only', action='store_true', help='Download raw data, preprocessing source, references and tokenizer; skip weights')
    args = parser.parse_args()
    reader = ArtifactReader(args.local_archive, args.artifact_url)
    prepare_files(reader)
    prepare_tokenizer()
    if not args.assets_only: prepare_checkpoint(reader)
