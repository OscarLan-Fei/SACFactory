import json
from pathlib import Path


def load_demo_seed(root):
    """Load a portable fixture and restore its declared simulator inputs."""
    from .demo_preparation import restore_content
    path = Path(root) / 'examples/demo_seed.json'
    if path.is_file():
        content = json.loads(path.read_text(encoding='utf-8'))
    else:
        import gzip
        with gzip.open(str(path) + '.gz', 'rt', encoding='utf-8') as stream:
            content = json.load(stream)
    return restore_content(content)


def seed(store, root):
    if store.get("meta", "seed-v1"):
        from .demo_preparation import migrate_store
        migrate_store(store)
        return
    if store.origin == "REAL":
        store.put_many("meta", [{"id": "seed-v1", "origin": store.origin}], "id")
        return
    content = load_demo_seed(root)
    from .domain import validate_sample
    from .storage import now
    fixture=[validate_sample(s) for s in content['samples']]
    if any(s.get('origin')!='DEMO' or not s.get('record_uid') for s in fixture):
        raise ValueError('Trusted DEMO fixture requires DEMO origin and stable record identity.')
    if len({s['sample_id'] for s in fixture})!=len(fixture) or len({s['record_uid'] for s in fixture})!=len(fixture):
        raise ValueError('DEMO fixture identities must be unique.')
    # Trusted fixture insertion avoids quadratic sample-identity scans. Normal
    # experimental/manual imports retain the Store identity/revision safeguards.
    with store.connect() as c:
        c.execute('BEGIN IMMEDIATE')
        existing={row[0] for row in c.execute("SELECT id FROM items WHERE kind='sample'")}
        incoming=[s for s in fixture if s['sample_id'] not in existing]
        encoded=[(s['sample_id'],json.dumps(s,allow_nan=False)) for s in incoming]
        c.executemany("INSERT INTO items VALUES('sample',?,?)",encoded)
        c.executemany("INSERT INTO audit(created,action,kind,id,before_json,after_json) VALUES(?,'seed','sample',?,NULL,?)",[(now(),ident,payload) for ident,payload in encoded])
    for kind, key, field in (("method", "methods", "method_id"),):
        existing = {s[field] for s in store.all(kind)}
        store.put_many(kind, [s for s in content[key] if s[field] not in existing], field)
    store.put_many("meta", [{"id": "seed-v1", "origin": store.origin,
        'demo_version':content['demo_version'],'loss_model_version':content.get('loss_model_version'),
        'observation_settings_sha256':content.get('observation_settings',{}).get('settings_sha256')}], "id")
    for key, field in [('plan_context','planning_context'),('space','planning_space')]:
        if field in content:
            store.put_many('meta',[{'id':key,'origin':'DEMO','context' if key=='plan_context' else 'space':content[field]}],'id')
    store.put_many('meta',[{'id':'element_campaign','origin':'DEMO','enabled':True,'strategy':'SINGLE_ELEMENT_FIRST'}],'id')


def replace_demo(store, root):
    """Explicitly reload disposable DEMO fixtures; REAL is strictly rejected.

    Normal startup never resets a populated namespace. This helper is used for
    the authorized demo test campaign and does not access filesystem backups.
    """
    from uuid import uuid4
    from .domain import validate_sample
    from .storage import now
    if store.origin != 'DEMO':
        raise ValueError('Demo replacement cannot modify REAL data.')
    content = load_demo_seed(root)
    samples = [validate_sample(s) for s in content['samples']]
    if any(s['origin'] != 'DEMO' for s in samples) or any(m['origin'] != 'DEMO' for m in content['methods']):
        raise ValueError('The demo seed must contain only DEMO records.')
    with store.connect() as c:
        c.execute('BEGIN IMMEDIATE')
        c.execute('DELETE FROM items')
        c.execute('DELETE FROM messages')
        c.execute('DELETE FROM audit')
        for kind, objects, field in (('sample', samples, 'sample_id'), ('method', content['methods'], 'method_id')):
            for obj in objects:
                c.execute('INSERT INTO items VALUES(?,?,?)', (kind, obj[field], json.dumps(obj, allow_nan=False)))
        for meta in ({'id': 'seed-v1', 'origin': 'DEMO', 'demo_version': content['demo_version']}, {'id': 'data-epoch', 'origin': 'DEMO', 'value': uuid4().hex}, {'id': 'icp-mass-percent-v1', 'origin': 'DEMO', 'unit': 'wt%'}, {'id': 'sample-numbering', 'origin': 'DEMO', 'high_water': 0}):
            c.execute("INSERT INTO items VALUES('meta',?,?)", (meta['id'], json.dumps(meta)))
        c.execute("INSERT INTO audit(created,action,kind,id,before_json,after_json) VALUES(?,'replace_demo','meta','seed-v1',NULL,?)", (now(), json.dumps({'demo_version': content['demo_version'], 'samples': len(samples)})))
        for key,field in [('plan_context','planning_context'),('space','planning_space')]:
            if field in content:
                c.execute("INSERT INTO items VALUES('meta',?,?)",(key,json.dumps({'id':key,'origin':'DEMO','context' if key=='plan_context' else 'space':content[field]})))
        c.execute("INSERT INTO items VALUES('meta','element_campaign',?)",(json.dumps({'id':'element_campaign','origin':'DEMO','enabled':True,'strategy':'SINGLE_ELEMENT_FIRST'}),))
    return len(samples)
