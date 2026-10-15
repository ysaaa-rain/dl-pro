"""Fixed geometric image summaries; never consume annotated boundaries or lengths."""
import numpy as np


def summarize_shapes(x):
    n=x.shape[-1];distance=abs(np.arange(n)[:,None]-np.arange(n)[None,:]);center=(n-1)/2;position=np.arange(n)-center;features=[]
    def stats(v):
        return [v.mean(1),v.std(1),np.quantile(v,.9,axis=1)]
    for channel in range(x.shape[1]):
        image=x[:,channel]
        for start,width in [(0,n),(n//4,n//2)]:
            region=image[:,start:start+width,start:start+width]
            tiles=region.reshape(len(x),4,width//4,4,width//4).transpose(0,1,3,2,4).reshape(len(x),16,-1)
            features.extend([tiles.mean(2),tiles.std(2)])
        for radius in [8,16,32,48]:
            roi=(abs(position[:,None])<=radius)&(abs(position[None,:])<=radius)&(distance>=2)
            features.extend([v[:,None] for v in stats(image[:,roi])])
        core=(abs(position[:,None])<32)&(abs(position[None,:])<32)
        for lo,hi in [(2,8),(8,16),(16,32),(32,64)]:
            roi=(distance>=lo)&(distance<hi)&core
            features.extend([v[:,None] for v in stats(image[:,roi])])
    result=np.concatenate(features,1)
    if not np.isfinite(result).all():raise ValueError('Nonfinite shape features')
    return result


def forest_probabilities(arrays,features):
    """Predict from portable tree arrays, independently of sklearn's tree engine."""
    x=np.asarray(features,dtype=np.float32);tree_ids=sorted({int(k.split('_')[0]) for k in arrays if k.endswith('_left')});probability=np.zeros((len(x),3),dtype=float)
    for tree in tree_ids:
        left=arrays[f'{tree}_left'];right=arrays[f'{tree}_right'];feature=arrays[f'{tree}_feature'];threshold=arrays[f'{tree}_threshold'];values=arrays[f'{tree}_value'];node=np.zeros(len(x),int)
        active=left[node]>=0
        while active.any():
            ids=np.flatnonzero(active);current=node[ids];go_left=x[ids,feature[current]]<=threshold[current];node[ids]=np.where(go_left,left[current],right[current]);active=left[node]>=0
        mass=values[node,0];probability+=mass/np.maximum(mass.sum(1,keepdims=True),1e-12)
    return probability/len(tree_ids)
