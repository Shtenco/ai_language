#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
from pathlib import Path


def clean_text(s: str) -> str:
    s = s.replace('\r', '\n').replace('\x00', ' ')
    s = re.sub(r'[ \t]+', ' ', s)
    s = re.sub(r'\n{4,}', '\n\n', s)
    return s.strip()


def cap_utf8_text(s: str, max_bytes: int) -> str:
    b = clean_text(s).encode('utf-8')
    if len(b) <= max_bytes:
        return b.decode('utf-8')
    b = b[:max_bytes]
    while b:
        try:
            return b.decode('utf-8')
        except UnicodeDecodeError:
            b = b[:-1]
    return ''


def sha_bucket(key: str, n: int = 100) -> int:
    return int.from_bytes(hashlib.sha256(key.encode('utf-8', 'ignore')).digest()[:8], 'big') % n


def conllu_docs(path: str, sentences_per_doc: int = 32):
    batch = []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if line.startswith('# text = '):
            t = clean_text(line[9:])
            if t:
                batch.append(t)
                if len(batch) >= sentences_per_doc:
                    yield '\n'.join(batch)
                    batch = []
    if batch:
        yield '\n'.join(batch)


def add_doc(dst, source, doc_id, text, limit_bytes, max_doc_bytes=24000):
    if sum(x['bytes'] for x in dst) >= limit_bytes:
        return False
    text = cap_utf8_text(text, max_doc_bytes)
    if len(text) < 200:
        return False
    raw = (text + '\n').encode('utf-8')
    left = limit_bytes - sum(x['bytes'] for x in dst)
    if left <= 0:
        return False
    if len(raw) > left:
        text = cap_utf8_text(text, max(0, left - 1))
        raw = (text + '\n').encode('utf-8')
    if len(raw) < 200:
        return False
    dst.append({'source': source, 'id': str(doc_id), 'text': text, 'bytes': len(raw)})
    return True


def write_jsonl(path: Path, docs):
    with path.open('w', encoding='utf-8') as f:
        for d in docs:
            f.write(json.dumps({k: d[k] for k in ('source', 'id', 'text')}, ensure_ascii=False) + '\n')


def write_text(path: Path, docs):
    path.write_text('\n\n'.join(d['text'] for d in docs) + '\n', encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--synt-train', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--wiki-train-mb', type=int, default=20)
    ap.add_argument('--wiki-memory-mb', type=int, default=5)
    ap.add_argument('--wiki-test-mb', type=int, default=3)
    ap.add_argument('--heritage-train-mb', type=int, default=6)
    ap.add_argument('--heritage-test-mb', type=int, default=2)
    ap.add_argument('--synt-train-mb', type=int, default=4)
    ap.add_argument('--max-doc-kb', type=int, default=24)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    MB = 1024 * 1024
    max_doc = a.max_doc_kb * 1024

    train, memory, wiki_test, heritage_test = [], [], [], []
    meta = {
        'format': 'nexus-r57-russian-corpus/1',
        'utf8_safe_document_shuffle': True,
        'max_doc_bytes': max_doc,
        'sources': {},
        'errors': [],
    }

    # Wikimedia Wikipedia: openly licensed, diverse article-level source.
    # Buckets are article-level and disjoint: 0-9 test, 10-24 graph memory, 25-99 LM train.
    try:
        from datasets import load_dataset
        ds = load_dataset('wikimedia/wikipedia', '20231101.ru', split='train', streaming=True)
        try:
            ds = ds.shuffle(seed=20260810, buffer_size=5000)
        except Exception:
            pass
        lim_tr = a.wiki_train_mb * MB
        lim_mem = a.wiki_memory_mb * MB
        lim_te = a.wiki_test_mb * MB
        seen = 0
        for r in ds:
            text = r.get('text') or ''
            title = str(r.get('title') or '')
            rid = str(r.get('id') or title)
            if len(text) < 500:
                continue
            b = sha_bucket(rid + '\x1f' + title)
            if b < 10:
                add_doc(wiki_test, 'wikimedia/wikipedia:20231101.ru', rid, text, lim_te, max_doc)
            elif b < 25:
                add_doc(memory, 'wikimedia/wikipedia:20231101.ru', rid, text, lim_mem, max_doc)
            else:
                add_doc(train, 'wikimedia/wikipedia:20231101.ru', rid, text, lim_tr, max_doc)
            seen += 1
            if (sum(x['bytes'] for x in train) >= lim_tr and
                sum(x['bytes'] for x in memory) >= lim_mem and
                sum(x['bytes'] for x in wiki_test) >= lim_te):
                break
        del ds
        meta['sources']['Wikipedia_ru'] = {
            'records_seen': seen,
            'train_docs': sum(d['source'].startswith('wikimedia/') for d in train),
            'train_bytes': sum(d['bytes'] for d in train if d['source'].startswith('wikimedia/')),
            'memory_docs': len(memory),
            'memory_bytes': sum(d['bytes'] for d in memory),
            'test_docs': len(wiki_test),
            'test_bytes': sum(d['bytes'] for d in wiki_test),
        }
    except Exception as e:
        meta['errors'].append('Wikipedia: ' + repr(e))

    # RuHeritage: literary style, capped per document so a few giant works cannot dominate.
    try:
        from datasets import load_dataset
        ds = load_dataset('maxzt/RuHeritage-Corpus', split='train', streaming=True)
        try:
            ds = ds.shuffle(seed=20260810, buffer_size=2000)
        except Exception:
            pass
        lim_tr = a.heritage_train_mb * MB
        lim_te = a.heritage_test_mb * MB
        start_train_bytes = sum(d['bytes'] for d in train)
        seen = 0
        for r in ds:
            text = r.get('text') or ''
            if len(text) < 500:
                continue
            key = str(r.get('author', '')) + '\x1f' + str(r.get('title', ''))
            b = sha_bucket(key)
            if b < 20:
                add_doc(heritage_test, 'maxzt/RuHeritage-Corpus', key, text, lim_te, max_doc)
            else:
                # limit applies to this source, not total train corpus
                source_bytes = sum(d['bytes'] for d in train) - start_train_bytes
                if source_bytes < lim_tr:
                    add_doc(train, 'maxzt/RuHeritage-Corpus', key, text,
                            start_train_bytes + lim_tr, max_doc)
            seen += 1
            source_bytes = sum(d['bytes'] for d in train) - start_train_bytes
            if source_bytes >= lim_tr and sum(d['bytes'] for d in heritage_test) >= lim_te:
                break
        del ds
        meta['sources']['RuHeritage'] = {
            'records_seen': seen,
            'train_docs': sum(d['source'].startswith('maxzt/') for d in train),
            'train_bytes': sum(d['bytes'] for d in train if d['source'].startswith('maxzt/')),
            'test_docs': len(heritage_test),
            'test_bytes': sum(d['bytes'] for d in heritage_test),
        }
    except Exception as e:
        meta['errors'].append('RuHeritage: ' + repr(e))

    # SynTagRus adds contemporary, syntactically curated sentences.
    synt_docs = []
    synt_lim = a.synt_train_mb * MB
    for i, text in enumerate(conllu_docs(a.synt_train)):
        add_doc(synt_docs, 'UD_Russian-SynTagRus', i, text, synt_lim, max_doc)
        if sum(d['bytes'] for d in synt_docs) >= synt_lim:
            break
    train.extend(synt_docs)
    meta['sources']['SynTagRus_train'] = {
        'train_docs': len(synt_docs),
        'train_bytes': sum(d['bytes'] for d in synt_docs),
    }

    # Deterministic DOCUMENT shuffle. Never cut or permute raw byte blocks.
    train.sort(key=lambda d: hashlib.sha256((d['source'] + '\x1f' + d['id']).encode('utf-8')).digest())
    memory.sort(key=lambda d: hashlib.sha256((d['source'] + '\x1f' + d['id']).encode('utf-8')).digest())
    wiki_test.sort(key=lambda d: d['id'])
    heritage_test.sort(key=lambda d: d['id'])

    write_jsonl(out / 'ru_train_docs.jsonl', train)
    write_jsonl(out / 'graph_memory_docs.jsonl', memory)
    write_jsonl(out / 'ru_wiki_test_docs.jsonl', wiki_test)
    write_jsonl(out / 'ru_heritage_test_docs.jsonl', heritage_test)
    write_text(out / 'ru_train.txt', train)
    write_text(out / 'graph_memory.txt', memory)
    write_text(out / 'ru_wiki_test.txt', wiki_test)
    write_text(out / 'ru_heritage_test.txt', heritage_test)

    meta['totals'] = {
        'train_docs': len(train),
        'train_bytes': sum(d['bytes'] for d in train),
        'memory_docs': len(memory),
        'memory_bytes': sum(d['bytes'] for d in memory),
        'wiki_test_docs': len(wiki_test),
        'wiki_test_bytes': sum(d['bytes'] for d in wiki_test),
        'heritage_test_docs': len(heritage_test),
        'heritage_test_bytes': sum(d['bytes'] for d in heritage_test),
    }
    meta['sha256'] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(out.glob('*')) if p.is_file()
    }
    (out / 'CORPUS_META.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)

    if meta['totals']['train_bytes'] < 12 * MB:
        raise RuntimeError(f"R5.7 Russian train corpus too small: {meta['totals']['train_bytes']} bytes")
    if meta['totals']['memory_bytes'] < 1 * MB:
        raise RuntimeError(f"R5.7 graph memory too small: {meta['totals']['memory_bytes']} bytes")


if __name__ == '__main__':
    main()
