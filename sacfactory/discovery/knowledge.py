"""Source-linked local retrieval; retrieved text is evidence, never an instruction."""
import json
import re
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from .common import digest,query,outcomes


def public_documents():
    root=Path(__file__).resolve().parents[2]/'resources'
    return [row for name in ['discovery_library.json','characterization_library.json']
            for row in json.loads((root/name).read_text(encoding='utf-8'))]


def document(title,text,source,kind='literature'):
    if not title.strip() or not text.strip():raise ValueError('A title and source text are required.')
    return {'document_id':digest([title,text,source])[:20],'title':title,'text':text,'source':source or 'User-supplied; bibliographic source not provided',
            'kind':kind,'status':'REFERENCE_TEXT_NOT_EXPERIMENTAL_LABEL'}


def read_upload(name,content,source=''):
    if len(content)>20_000_000:raise ValueError('Use documents below 20 MB.')
    if name.lower().endswith('.pdf'):
        import pymupdf
        with pymupdf.open(stream=content,filetype='pdf') as pdf:
            text='\n'.join(page.get_text() for page in pdf)
    else:text=content.decode('utf-8-sig')
    if not text.strip():raise ValueError('No extractable text; scanned documents need OCR first.')
    return document(Path(name).name,text[:1_000_000],source)


def history_documents(samples):
    result=[]
    for sample in samples:
        try:
            conditions=query(sample);actual,label=outcomes(sample)
        except (ValueError,TypeError):continue
        text=json.dumps({'conditions':conditions,'ICP_W_wt_percent':actual,'structural_PASS':label},sort_keys=True)
        result.append(document('Experiment '+str(sample['sample_id']),text,'record:'+str(sample['sample_id']),'experiment'))
    return result


def search(documents,question,limit=5):
    if not isinstance(question,str) or not question.strip() or len(question)>4000:raise ValueError('Enter a search question below 4000 characters.')
    if not 1<=int(limit)<=12:raise ValueError('Select 1-12 evidence passages.')
    chunks=[]
    for entry in documents:
        if not all(k in entry for k in ['document_id','title','text','source']):continue
        text=re.sub(r'\s+',' ',entry['text']).strip()
        for start in range(0,len(text),900):
            passage=text[start:start+1200]
            if passage:chunks.append({**{k:v for k,v in entry.items() if k!='text'},'passage':passage,'offset':start})
    if not chunks:return []
    vectorizer=TfidfVectorizer(analyzer='char_wb',ngram_range=(2,5),lowercase=True)
    matrix=vectorizer.fit_transform([r['title']+' '+r['passage'] for r in chunks])
    scores=(matrix@vectorizer.transform([question]).T).toarray().ravel()
    return [{**chunks[i],'retrieval_score':float(scores[i]),'retrieval_method':'Local lexical TF-IDF embeddings; not a dense semantic model',
             'evidence_role':'Retrieved context; not a fitted parameter or a new measurement'}
            for i in np.argsort(-scores)[:int(limit)] if scores[i]>.02]
