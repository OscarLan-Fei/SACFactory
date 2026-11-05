"""Categorized equipment library and editable instrument registry."""
import json
import pandas as pd
import streamlit as st

from .device_catalog import CATALOG,CATEGORIES,CAPABILITIES,PROFILES,category_of,modalities_of,profile_device


def render_devices(service,prefix):
    st.write('**Laboratory equipment directory**')
    registered=service.db.all('devices')
    legacy=[d for d in registered if d['id']=='demo_characterizer']
    if legacy and not st.checkbox('Show legacy combined fixture',key=prefix+'_show_legacy'):
        registered=[d for d in registered if d['id']!='demo_characterizer']
    cols=st.columns(4)
    for col,category in zip(cols,CATEGORIES):
        count=sum(category_of(d)==category for d in registered)
        col.metric(category,str(count),help=f'{sum(p["category"]==category for p in CATALOG)} equipment types in the library')
    st.caption('Equipment types are reusable profiles. Registered entries are planning inventory or explicitly configured instruments; a catalog entry does not imply ownership or a working connection.')
    view=st.radio('Directory view',['Equipment catalog','Registered devices'],horizontal=True,key=prefix+'_directory')
    categories=st.multiselect('Equipment categories',CATEGORIES,default=CATEGORIES,key=prefix+'_categories')
    search=st.text_input('Find equipment',placeholder='ICP, furnace, electrochemistry, balance…',key=prefix+'_search').strip().casefold()
    source=CATALOG if view=='Equipment catalog' else registered
    shown=[d for d in source if category_of(d) in categories and (not search or search in ' '.join([d['name'],d['id'],*d.get('modalities',[]),*d.get('capabilities',[])]).casefold())]
    rows=[{'Category':category_of(d),'Equipment':d['name'],'Type':d.get('equipment_type',d['id']),
           'Capabilities':', '.join(d['capabilities']),'Measurements':', '.join(modalities_of(d)),
           **({'Device ID':d['id'],'Mode':d['mode'],'Status':d.get('connection_state','NOT_CONNECTED')} if view=='Registered devices' else
              {'Registered instances':sum(r.get('equipment_type')==d['id'] for r in registered)})} for d in shown]
    if rows:st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch',height=min(600,80+35*len(rows)))
    else:st.info('No matching equipment. Change the filters or register a custom type below.')
    st.caption(f'{len(shown)} displayed / {len(source)} available. General equipment is grouped as a collection on the Live board.')
    if view=='Equipment catalog':
        if st.button('Add displayed types to planning inventory',disabled=not shown,key=prefix+'_import_catalog'):
            added=0
            for profile in shown:
                ident='inventory_'+profile['id']
                if not service.db.get('devices',ident):service.devices.register(profile_device(profile['id']));added+=1
            st.success(f'Added {added} planning entries. These entries have no connected execution adapter.');st.rerun()
    else:
        st.download_button('Export equipment registry',json.dumps(registered,indent=2).encode(),'equipment_registry.json','application/json',key=prefix+'_registry_export')
    if st.button('Register independent acquisition simulators',key=prefix+'_defaults'):
        service.devices.defaults();st.success('Registered the pulsed heater plus independent ICP, XRD, STEM and SEM fixtures.');st.rerun()
    requested=st.session_state.get('workflow_device_profile')
    with st.expander('Register or edit an instrument',expanded=not registered or bool(requested and requested['origin']==service.store.origin)):
        lookup={d['id']:d for d in registered}
        if requested and requested['origin']==service.store.origin:
            st.session_state[prefix+'_edit_entry']='New instrument'
            st.session_state[prefix+'_profile_New instrument']=requested['profile_id']
            del st.session_state['workflow_device_profile']
        edit=st.selectbox('Instrument entry',['New instrument']+list(lookup),format_func=lambda key:lookup[key]['name']+' · '+key if key in lookup else key,key=prefix+'_edit_entry')
        existing=lookup.get(edit,{})
        default_type=existing.get('equipment_type','custom')
        types=['custom']+list(PROFILES)
        profile_id=st.selectbox('Equipment type',types,index=types.index(default_type) if default_type in types else 0,
             format_func=lambda key:'Custom equipment' if key=='custom' else PROFILES[key]['name'],key=prefix+'_profile_'+edit)
        defaults=profile_device(profile_id) if profile_id!='custom' else {'id':'custom_device','name':'Custom instrument','category':CATEGORIES[3],
            'capabilities':['prepare'],'modalities':[],'mode':'INVENTORY_ONLY','limits':{}}
        base={**defaults,**existing} if existing.get('equipment_type','custom')==profile_id else {**defaults,'id':existing.get('id',defaults['id'])}
        key=prefix+'_entry_'+edit+'_'+profile_id
        ident=st.text_input('Device ID',value=base['id'],disabled=bool(existing),key=key+'_id')
        name=st.text_input('Device name',value=base['name'],key=key+'_name')
        category=st.selectbox('Equipment category',CATEGORIES,index=CATEGORIES.index(category_of(base)),key=key+'_category')
        capabilities=st.multiselect('Capabilities',CAPABILITIES,default=base['capabilities'],key=key+'_caps')
        modalities=st.text_input('Measurement types (comma-separated)',value=', '.join(modalities_of(base)),key=key+'_modalities')
        modes=['INVENTORY_ONLY','MCP_NOT_CONNECTED','SIMULATOR']
        mode=st.selectbox('Connection mode',modes,index=modes.index(base['mode']),key=key+'_mode')
        limits={}
        if 'synthesize' in capabilities:
            bounds=base.get('limits',{})
            cols=st.columns(4)
            low=cols[0].number_input('Minimum cutoff / C',1,3900,int(bounds.get('cutoff_C',[100,2600])[0]),key=key+'_low')
            high=cols[1].number_input('Maximum cutoff / C',1,3900,int(bounds.get('cutoff_C',[100,2600])[1]),key=key+'_high')
            pmin=cols[2].number_input('Minimum pulses',1,1000,int(bounds.get('pulse_count',[1,16])[0]),key=key+'_pmin')
            pmax=cols[3].number_input('Maximum pulses',1,1000,int(bounds.get('pulse_count',[1,16])[1]),key=key+'_pmax')
            limits={'cutoff_C':[low,high],'pulse_count':[pmin,pmax]}
        endpoint=st.text_input('MCP HTTPS endpoint (no credentials)',value=base.get('mcp_endpoint',''),key=key+'_endpoint') if mode=='MCP_NOT_CONNECTED' else ''
        notes=st.text_area('Location, sample transfer and operating notes',value=base.get('description',''),key=key+'_notes')
        fail=st.checkbox('Inject one pre-execution failure for simulator recovery testing',value=base.get('fail_once',False),key=key+'_fail') if mode=='SIMULATOR' else False
        if st.button('Save device contract',key=key+'_save'):
            service.devices.register(dict(id=ident,name=name,category=category,equipment_type=profile_id,capabilities=capabilities,
                modalities=[m.strip().upper() for m in modalities.split(',') if m.strip()],limits=limits,mode=mode,mcp_endpoint=endpoint,description=notes,fail_once=fail))
            st.success('Device contract saved.');st.rerun()
        st.caption('Acquisition fixtures currently cover pulsed synthesis, ICP, XRD, STEM and SEM. Other types remain planning inventory or disconnected MCP contracts until an adapter is implemented.')
    jobs=service.db.all('jobs')
    if jobs:
        with st.expander('Instrument jobs'):
            st.dataframe(pd.DataFrame([{k:j.get(k) for k in ['id','device_id','capability','state','attempts','simulated']} for j in jobs]),hide_index=True,width='stretch')
