#!/usr/bin/env python3
import argparse, json, random, sqlite3, statistics, time
from pathlib import Path

SEED=20260810
SCHEMA='''
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS node(id INTEGER PRIMARY KEY, kind TEXT NOT NULL, label TEXT NOT NULL UNIQUE, content TEXT DEFAULT '');
CREATE TABLE IF NOT EXISTS claim(id INTEGER PRIMARY KEY, entity_id INTEGER NOT NULL, relation TEXT NOT NULL, value TEXT NOT NULL, polarity INTEGER NOT NULL DEFAULT 1, valid_from INTEGER NOT NULL, valid_to INTEGER NOT NULL, confidence REAL NOT NULL, source TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'verified', FOREIGN KEY(entity_id) REFERENCES node(id));
CREATE INDEX IF NOT EXISTS claim_lookup ON claim(entity_id,relation,valid_from,valid_to);
CREATE TABLE IF NOT EXISTS edge(id INTEGER PRIMARY KEY, src INTEGER NOT NULL, relation TEXT NOT NULL, dst INTEGER NOT NULL, confidence REAL NOT NULL DEFAULT 1.0, source TEXT NOT NULL DEFAULT '', FOREIGN KEY(src) REFERENCES node(id), FOREIGN KEY(dst) REFERENCES node(id));
CREATE INDEX IF NOT EXISTS edge_src ON edge(src,relation);
CREATE TABLE IF NOT EXISTS rule(id INTEGER PRIMARY KEY, premise_src INTEGER NOT NULL, premise_relation TEXT NOT NULL, conclusion_dst INTEGER NOT NULL, conclusion_relation TEXT NOT NULL, source TEXT NOT NULL DEFAULT '');
CREATE TABLE IF NOT EXISTS goal(id INTEGER PRIMARY KEY, session TEXT NOT NULL, text TEXT NOT NULL, criterion TEXT NOT NULL, expected TEXT NOT NULL, observed TEXT DEFAULT '', error TEXT DEFAULT '', mode TEXT NOT NULL DEFAULT 'observe');
CREATE TABLE IF NOT EXISTS episode(id INTEGER PRIMARY KEY, session TEXT NOT NULL, ts INTEGER NOT NULL, observation TEXT NOT NULL, action TEXT NOT NULL, result TEXT NOT NULL, success INTEGER NOT NULL, provenance TEXT NOT NULL);
CREATE VIRTUAL TABLE IF NOT EXISTS node_fts USING fts5(label,content,content='node',content_rowid='id');
CREATE TRIGGER IF NOT EXISTS node_ai AFTER INSERT ON node BEGIN INSERT INTO node_fts(rowid,label,content) VALUES(new.id,new.label,new.content); END;
CREATE TRIGGER IF NOT EXISTS node_ad AFTER DELETE ON node BEGIN INSERT INTO node_fts(node_fts,rowid,label,content) VALUES('delete',old.id,old.label,old.content); END;
CREATE TRIGGER IF NOT EXISTS node_au AFTER UPDATE ON node BEGIN INSERT INTO node_fts(node_fts,rowid,label,content) VALUES('delete',old.id,old.label,old.content); INSERT INTO node_fts(rowid,label,content) VALUES(new.id,new.label,new.content); END;
'''

class Memory:
    def __init__(self,path):
        self.db=sqlite3.connect(path);self.db.execute('PRAGMA foreign_keys=ON');self.db.executescript(SCHEMA)
    def close(self):self.db.commit();self.db.close()
    def node(self,label,kind='ENTITY',content=''):
        self.db.execute('INSERT OR IGNORE INTO node(kind,label,content) VALUES(?,?,?)',(kind,label,content));r=self.db.execute('SELECT id FROM node WHERE label=?',(label,)).fetchone();return r[0]
    def claim(self,e,rel,val,vf,vt,source,conf=1.0,pol=True,status='verified'):
        eid=self.node(e);self.db.execute('INSERT INTO claim(entity_id,relation,value,polarity,valid_from,valid_to,confidence,source,status) VALUES(?,?,?,?,?,?,?,?,?)',(eid,rel,val,int(pol),vf,vt,conf,source,status))
    def edge(self,a,rel,b,source='',conf=1.0):
        ai=self.node(a);bi=self.node(b);self.db.execute('INSERT INTO edge(src,relation,dst,confidence,source) VALUES(?,?,?,?,?)',(ai,rel,bi,conf,source))
    def current(self,e,rel,t):
        q='''SELECT c.value,c.polarity,c.valid_from,c.valid_to,c.confidence,c.source,c.status FROM claim c JOIN node n ON n.id=c.entity_id WHERE n.label=? AND c.relation=? AND c.valid_from<=? AND c.valid_to>=? ORDER BY c.valid_from DESC,c.confidence DESC,c.id DESC'''
        rows=self.db.execute(q,(e,rel,t,t)).fetchall()
        if not rows:return {'status':'missing','claim':None,'conflict':False}
        top=rows[0];conflict=any(r[2]==top[2] and abs(r[4]-top[4])<1e-12 and (r[0]!=top[0] or r[1]!=top[1]) for r in rows[1:])
        return {'status':'conflict' if conflict else 'resolved','claim':{'value':top[0],'polarity':bool(top[1]),'valid_from':top[2],'valid_to':top[3],'confidence':top[4],'source':top[5],'claim_status':top[6]},'conflict':conflict}
    def search(self,text,limit=8):
        # FTS5 query tokens joined with OR to form GraphRAG entry points.
        toks=[x.replace('"','') for x in text.lower().split() if len(x)>2][:8]
        if not toks:return []
        q=' OR '.join(f'"{x}"' for x in toks)
        try:rows=self.db.execute('SELECT rowid,label,content,bm25(node_fts) FROM node_fts WHERE node_fts MATCH ? ORDER BY bm25(node_fts) LIMIT ?',(q,limit)).fetchall()
        except sqlite3.OperationalError:return []
        return [{'id':r[0],'label':r[1],'content':r[2],'score':r[3]} for r in rows]
    def expand(self,node_ids,hops=2,limit=64):
        seen=set(node_ids);front=set(node_ids);edges=[]
        for _ in range(hops):
            if not front:break
            marks=','.join('?'*len(front));rows=self.db.execute(f'SELECT e.src,e.relation,e.dst,e.confidence,e.source,a.label,b.label FROM edge e JOIN node a ON a.id=e.src JOIN node b ON b.id=e.dst WHERE e.src IN ({marks}) OR e.dst IN ({marks}) LIMIT ?',[*front,*front,limit]).fetchall();nxt=set()
            for r in rows:
                edges.append({'src':r[5],'relation':r[1],'dst':r[6],'confidence':r[3],'source':r[4]})
                for x in (r[0],r[2]):
                    if x not in seen:seen.add(x);nxt.add(x)
            front=nxt
        return edges
    def retrieve(self,text,hops=2):
        roots=self.search(text);return {'roots':roots,'edges':self.expand([r['id'] for r in roots],hops=hops),'claims':self._claims_for([r['id'] for r in roots])}
    def _claims_for(self,ids):
        if not ids:return []
        marks=','.join('?'*len(ids));rows=self.db.execute(f'SELECT n.label,c.relation,c.value,c.valid_from,c.valid_to,c.confidence,c.source,c.polarity,c.status FROM claim c JOIN node n ON n.id=c.entity_id WHERE c.entity_id IN ({marks}) ORDER BY c.valid_from DESC',ids).fetchall();return [{'entity':r[0],'relation':r[1],'value':r[2],'valid_from':r[3],'valid_to':r[4],'confidence':r[5],'source':r[6],'polarity':bool(r[7]),'status':r[8]} for r in rows]
    def set_goal(self,session,text,criterion,expected):
        self.db.execute('INSERT INTO goal(session,text,criterion,expected,mode) VALUES(?,?,?,?,?)',(session,text,criterion,expected,'observe'));return self.db.execute('SELECT last_insert_rowid()').fetchone()[0]
    def feedback(self,gid,observed):
        row=self.db.execute('SELECT expected FROM goal WHERE id=?',(gid,)).fetchone();err='' if row and row[0]==observed else f'expected={row[0] if row else None}; observed={observed}';mode='stable' if not err else 'recover';self.db.execute('UPDATE goal SET observed=?,error=?,mode=? WHERE id=?',(observed,err,mode,gid));return {'mode':mode,'error':err}


def build(path,n_entities=10000):
    rng=random.Random(SEED);m=Memory(path)
    for i in range(n_entities):
        e=f'объект-{i:05d}';m.node(e,'ENTITY',f'Тестовый объект номер {i} для графовой памяти.')
        cut=100+(i%700);m.claim(e,'состояние',f'старое-{i%17}',0,cut-1,'archive',.9);m.claim(e,'состояние',f'новое-{i%23}',cut,9999,'sensor',.98)
        c=f'понятие-{i%250:03d}';m.node(c,'CONCEPT',f'Понятие класса {i%250}');m.edge(e,'INSTANCE_OF',c,'generator')
        if i>0:m.edge(f'объект-{i-1:05d}','CAUSES',e,'causal-generator')
    # Equal-priority contradiction set.
    m.claim('конфликтный-объект','режим','А',50,100,'source-A',.95);m.claim('конфликтный-объект','режим','Б',50,100,'source-B',.95)
    m.db.commit();return m


def benchmark(path,queries=5000):
    rng=random.Random(SEED+1);m=Memory(path);hist_ok=0;cur_ok=0;lat=[]
    for _ in range(queries):
        i=rng.randrange(10000);e=f'объект-{i:05d}';cut=100+(i%700)
        t0=time.perf_counter_ns();a=m.current(e,'состояние',cut-1);lat.append((time.perf_counter_ns()-t0)/1e6);hist_ok+=a['claim']['value']==f'старое-{i%17}'
        b=m.current(e,'состояние',cut);cur_ok+=b['claim']['value']==f'новое-{i%23}'
    c=m.current('конфликтный-объект','режим',75)
    # Search and 2-hop expansion smoke on persisted DB.
    r=m.retrieve('объект-05000 понятие',hops=2)
    gid=m.set_goal('s1','получить HTTP 200','HTTP status','200');bad=m.feedback(gid,'500');good_gid=m.set_goal('s2','получить HTTP 200','HTTP status','200');good=m.feedback(good_gid,'200');m.db.commit();m.close()
    # Reopen persistence invariant.
    m2=Memory(path);persist=m2.current('объект-05000','состояние',999)['status']=='resolved';counts={t:m2.db.execute(f'SELECT count(*) FROM {t}').fetchone()[0] for t in ('node','claim','edge','goal')};m2.close()
    return {'historical_accuracy':hist_ok/queries,'boundary_current_accuracy':cur_ok/queries,'conflict_detected':c['status']=='conflict','search_roots':len(r['roots']),'expanded_edges':len(r['edges']),'retrieved_claims':len(r['claims']),'cyber_bad_mode':bad['mode'],'cyber_good_mode':good['mode'],'persist_after_reopen':persist,'latency_ms':{'median':statistics.median(lat),'p95':sorted(lat)[int(.95*len(lat))-1],'max':max(lat)},'counts':counts}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='nexus_r525_results');a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);db=out/'nexus_graph_memory.sqlite';m=build(db);m.close();res=benchmark(db);result={'format':'nexus-r525-persistent-graphrag/1','schema':['NODE(kind,label,content)','CLAIM(entity,relation,value,polarity,valid_time,confidence,source,status)','EDGE(src,relation,dst,confidence,source)','RULE','GOAL','EPISODE','FTS5 lexical entry'],'benchmark':res,'note':'Synthetic persistence/temporal benchmark; not a claim of 100% retrieval on arbitrary natural-language knowledge.'};(out/'00_R525_RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');(out/'README_RU.md').write_text('# NEXUS R5.25 Persistent GraphRAG Memory\n\nТипизированная долговременная память с временными фактами, provenance, конфликтами, FTS entry points, graph expansion и кибернетическим goal/feedback state.\n',encoding='utf-8');print(json.dumps(result,ensure_ascii=False));assert res['historical_accuracy']==1 and res['boundary_current_accuracy']==1 and res['conflict_detected'] and res['persist_after_reopen'] and res['cyber_bad_mode']=='recover' and res['cyber_good_mode']=='stable'
if __name__=='__main__':main()
