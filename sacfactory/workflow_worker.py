"""One bounded background workflow worker per local data space."""
import threading
from pathlib import Path

_guard=threading.Lock();_workers={}


def workspace_key(service):return str(Path(service.store.root).resolve())+':'+service.store.origin


def worker_status(service):
    with _guard:
        value=_workers.get(workspace_key(service))
        if not value:return {'active':False,'run_id':None,'error':None}
        return {'active':value['thread'].is_alive(),'run_id':value['run_id'],'error':value.get('error')}


def launch(service,run_id,max_steps=48):
    key=workspace_key(service)
    with _guard:
        current=_workers.get(key)
        if current and current['thread'].is_alive():raise ValueError('A background worker is already active for '+current['run_id'])
        run=service.db.get('runs',run_id)
        if run is None or run['state'] not in {'READY','RUNNING','QUEUED'}:raise ValueError('Choose a ready or queued run.')
        value={'run_id':run_id,'error':None}
        def work():
            try:service.db.run_ready(run_id,service.execute,max_steps=max_steps)
            except Exception as exc:
                with _guard:value['error']=f'{type(exc).__name__}: {exc}'
        thread=threading.Thread(target=work,name='SACWorkflow-'+run_id,daemon=True);value['thread']=thread
        _workers[key]=value;thread.start()
    return {'run_id':run_id,'started':True}
