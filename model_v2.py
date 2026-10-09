"""Rebuild the serialized v2 encoder using its actual tensor shapes."""
import hashlib,json,pickle,sys,types,math
from pathlib import Path
import core as base
import torch
from transformers import RobertaConfig
ROOT=Path(__file__).resolve().parent
def load_v2(kind,device):
    manifest=json.loads((ROOT/'models'/f'v2_{kind}_checkpoint_manifest.json').read_text())
    path=ROOT/manifest['inference_checkpoint']
    with path.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    if sha!=manifest['sha256']:raise RuntimeError('v2 checkpoint SHA256 mismatch')
    approved=set(manifest['globals'])
    known=set(json.loads((ROOT/'provenance/checkpoint_manifest.json').read_text())['globals'])
    if approved-known:raise RuntimeError('Unreviewed checkpoint globals: '+str(approved-known))
    class RestoredAuthorModule(torch.nn.Module):
        """Restore the author's nn.Module state without importing training code."""
        pass
    class ApprovedUnpickler(pickle.Unpickler):
        def find_class(self,module,name):
            if module+'.'+name not in approved:raise pickle.UnpicklingError(module+'.'+name)
            if module+'.'+name=='models.trans.PretainedTrans.PretrainedTrans':return RestoredAuthorModule
            return super().find_class(module,name)
    pm=types.ModuleType('v2_approved_checkpoint_pickle');pm.Unpickler=ApprovedUnpickler
    pm.load=lambda f,**kw:ApprovedUnpickler(f,**kw).load()
    saved=torch.load(path,map_location='cpu',weights_only=False,pickle_module=pm)
    old=saved['model'].eval()
    config=RobertaConfig.from_dict(old.trans.config.to_dict())
    declared_positions=config.max_position_embeddings
    actual_positions=old.trans.embeddings.position_embeddings.weight.shape[0]
    # Author extends the position weight by concatenating it. Config and Embedding
    # num_embeddings metadata may still differ from the actual serialized weight.
    config.max_position_embeddings=actual_positions
    config._attn_implementation='sdpa'
    fresh=base.MutationBERT(config)
    mismatch=fresh.load_state_dict(old.state_dict(),strict=False)
    allowed={'trans.embeddings.position_ids','trans.embeddings.token_type_ids'}
    if mismatch.missing_keys or set(mismatch.unexpected_keys)-allowed:raise RuntimeError(str(mismatch))
    info={'checkpoint_kind':kind,'source_record':'10654933','sha256':sha,'epoch':int(saved['epoch']),
          'declared_max_position_embeddings':declared_positions,'actual_position_weight_rows':actual_positions,
          'legacy_position_ids_shape':list(old.trans.embeddings.position_ids.shape),
          'ignored_legacy_buffers':mismatch.unexpected_keys,'input_context':1024}
    return fresh.to(device).eval(),old,info

def historical_reference(old,ids,mask):
    """Independent HF 4.23.1 encoder path using the original serialized modules."""
    e=old.trans.embeddings;nonpad=ids.ne(e.padding_idx).int()
    position_ids=(torch.cumsum(nonpad,dim=1).type_as(nonpad)*nonpad).long()+e.padding_idx
    x=e.word_embeddings(ids)+e.token_type_embeddings(torch.zeros_like(ids))+e.position_embeddings(position_ids)
    x=e.dropout(e.LayerNorm(x))
    additive=(1.0-mask[:,None,None,:].to(x.dtype))*torch.finfo(x.dtype).min
    for layer in old.trans.encoder.layer:
        a=layer.attention.self
        def reshape(t):return t.view(*t.shape[:-1],a.num_attention_heads,a.attention_head_size).permute(0,2,1,3)
        q,k,v=reshape(a.query(x)),reshape(a.key(x)),reshape(a.value(x))
        probs=a.dropout(torch.softmax((q@k.transpose(-1,-2))/math.sqrt(a.attention_head_size)+additive,dim=-1))
        context=(probs@v).permute(0,2,1,3).contiguous().view(*x.shape[:-1],a.all_head_size)
        output=layer.attention.output;x=output.LayerNorm(output.dropout(output.dense(context))+x)
        h=layer.intermediate.intermediate_act_fn(layer.intermediate.dense(x))
        output=layer.output;x=output.LayerNorm(output.dropout(output.dense(h))+x)
    return torch.softmax(old.linear(x[:,0]),dim=-1)[:,1]
