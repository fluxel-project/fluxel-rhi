"""Summarize visible n-body coverage; simulation iteration rates differ across programs."""
import json
from pathlib import Path
from PIL import Image
import numpy as np
root = Path('临时/当前验证/dx12')
results = {}
for name in ('0.17-official-nbody','0.17-rhi-nbody'):
    frames=[]
    for index,p in enumerate(sorted((root/name).glob('dynamic-*.png' if name == '0.17-rhi-nbody' else 'hello-*.png'))):
        a=np.asarray(Image.open(p).convert('RGB'),dtype=np.int16)
        red=a[:,:,0]; active=red>20; bright=(red>200)&(a[:,:,1]>100)
        y,x=np.nonzero(active)
        frames.append({'time_seconds':(3,5,7)[index], 'lit_pixels':int(active.sum()),'bright_pixels':int(bright.sum()),'centroid':[round(float(x.mean()),1),round(float(y.mean()),1)],'mean_red':round(float(red.mean()),2),'bbox':[int(x.min()),int(y.min()),int(x.max()),int(y.max())]})
    assert len(frames)==3 and all(f['lit_pixels']>10000 and f['bright_pixels']>1000 for f in frames)
    assert len({f['lit_pixels'] for f in frames}) == 3
    results[name]=frames
print(json.dumps(results,indent=2))

