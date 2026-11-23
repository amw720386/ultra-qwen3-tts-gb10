"""Pooled WER per KV format over five seeds, with a bootstrap over texts of the differences.

    python bench/gb10/kv_wer_stats.py   # from the repo root
"""
import json,random,statistics as st
F=['bf16','int8','fp8']; E={};N={}
for f in F:
    for s in range(1,6):
        for r in json.load(open(f'results/gb10/kvseed/{f}-s{s}.json'))['rows']:
            E.setdefault(r['id'],{}).setdefault(f,[]).append((r['errors'],r['n_ref']))
ids=list(E); assert all(len(E[i][f])==5 for i in ids for f in F)
def wer(f,ii,excl=False):
    e=n=0
    for i in ii:
        for a,b in E[i][f]:
            if excl and a/max(b,1)>.5: continue
            e+=a;n+=b
    return 100*e/n
for f in F:
    cat=sum(a/max(b,1)>.5 for i in ids for a,b in E[i][f])
    per=[100*sum(E[i][f][s][0] for i in ids)/sum(E[i][f][s][1] for i in ids) for s in range(5)]
    print(f,'WER %.3f'%wer(f,ids),'excl %.3f'%wer(f,ids,True),'catastrophic',cat,'/1500 per-seed',[round(x,2) for x in per],'sd %.2f'%st.stdev(per))
random.seed(0)
for a,b in (('int8','bf16'),('fp8','bf16'),('int8','fp8')):
    for ex in (False,True):
        d=[]
        for _ in range(4000):
            ii=random.choices(ids,k=len(ids)); d.append(wer(a,ii,ex)-wer(b,ii,ex))
        d.sort(); print(a,'-',b,'excl' if ex else 'all','%.3f'%(wer(a,ids,ex)-wer(b,ids,ex)),'95%% CI [%.3f, %.3f]'%(d[100],d[3899]))
