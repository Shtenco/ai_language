#!/usr/bin/env python3
import argparse
from pathlib import Path

def conllu_text(p):
    out=[]
    for line in Path(p).read_text(encoding='utf-8').splitlines():
        if line.startswith('# text = '):out.append(line[9:])
    return ('\n'.join(out)+'\n').encode('utf-8')

def cap_utf8(b,n):
    if len(b)<=n:return b
    x=b[:n]
    while True:
        try:x.decode('utf-8');return x
        except UnicodeDecodeError:x=x[:-1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--wiki-train',required=True);p.add_argument('--synt-train',required=True);p.add_argument('--synt-test',required=True);p.add_argument('--gsd-test',required=True);p.add_argument('--out',required=True);p.add_argument('--per-lang-bytes',type=int,default=5000000);a=p.parse_args()
    o=Path(a.out);o.mkdir(parents=True,exist_ok=True)
    en=cap_utf8(Path(a.wiki_train).read_bytes(),a.per_lang_bytes)
    ru=cap_utf8(conllu_text(a.synt_train),a.per_lang_bytes)
    # Keep each UTF-8 stream intact. Tokenizer training is order-insensitive enough for this preselection,
    # and this avoids creating invalid multibyte boundaries by slicing raw bytes into blocks.
    mixed=en+b'\n'+ru
    mixed.decode('utf-8')  # hard gate
    (o/'mixed_train.txt').write_bytes(mixed)
    (o/'ru_synt_test.txt').write_bytes(conllu_text(a.synt_test))
    (o/'ru_gsd_shift.txt').write_bytes(conllu_text(a.gsd_test))
    print('EN train bytes',len(en));print('RU train bytes',len(ru));print('mixed',len(mixed))
if __name__=='__main__':main()
