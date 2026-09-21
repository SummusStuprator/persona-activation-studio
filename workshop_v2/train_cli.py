"""One-time native training of the broad emotion bank on an existing checkpoint."""
import argparse,json,time,traceback
from pathlib import Path
from .library import train_go,train_extensions,GO_LABELS,EXTENSIONS
ROOT=Path(__file__).resolve().parent.parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',default='llama3.2:1b');p.add_argument('--resume',action='store_true');p.add_argument('--cpu',action='store_true');args=p.parse_args()
    from isolated_engine import Engine
    from model_store import grouped_inventory
    from science import load_bank
    model=next(m for m in grouped_inventory()[0] if args.model==m['name'] or args.model in m['aliases'])
    engine=Engine();report={'model':args.model,'started':time.time(),'results':[]}
    folder=ROOT/'reports';folder.mkdir(exist_ok=True)
    def progress(i,n,msg):
        if 'held-out' in msg:print(msg,flush=True)
    try:
        engine.open(model,context=1024,mode='CPU' if args.cpu else 'Auto')
        print('LOADED',model['name'],engine.load_plan['gpu_layers'],flush=True)
        for target in GO_LABELS+list(EXTENSIONS):
            name=('emotion_' if target in GO_LABELS else 'state_')+target
            try:
                old=load_bank(engine)
                if not args.resume or name not in old:
                    if target in GO_LABELS:train_go(engine,[target],progress)
                    else:train_extensions(engine,[target],progress)
                bank=load_bank(engine);meta=bank[name]['meta']
                report['results'].append({'name':name,'status':'passed','auc':meta['heldout_auc'],'n':meta['test_n'],'layer':meta['layer']})
                print('TRAINED',name,meta['heldout_auc'],meta['test_n'],flush=True)
            except Exception as exc:
                report['results'].append({'name':name,'status':'failed','error':str(exc)});print('FAILED',name,traceback.format_exc(),flush=True)
            (folder/'workshop-training.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        report['status']='passed' if all(r['status']=='passed' for r in report['results']) else 'partial';report['finished']=time.time()
    finally:
        engine.close();(folder/'workshop-training.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('TRAINING_COMPLETE',report.get('status'),flush=True)
if __name__=='__main__':main()
