"""Compare controlled official and RHI nBody iteration-one GPU output."""
import json
import struct
from pathlib import Path
from PIL import Image, ImageChops
p=Path('临时/当前验证/dx12/0.17-rhi-nbody')
a=(p/'official-step1.bin').read_bytes(); b=(p/'step1.bin').read_bytes()
assert len(a)==len(b)==320000
x=struct.unpack('<80000f',a); y=struct.unpack('<80000f',b)
d=[abs(a-b) for a,b in zip(x,y)]
image=ImageChops.difference(Image.open(p/'official-step1.ppm').convert('RGB'),Image.open(p/'step1.ppm').convert('RGB'))
assert image.size == (1280,720)
pixels=image.load(); different=0; total=0; maximum=0
for yy in range(720):
    for xx in range(1280):
        pixel=pixels[xx,yy]; different+=any(pixel); total+=sum(pixel); maximum=max(maximum,*pixel)
print(json.dumps({'particle_max_abs_diff':max(d),'particle_mean_abs_diff':sum(d)/len(d),'image_different_pixels':different,'image_max_channel_diff':maximum,'image_mean_channel_diff':total/(1280*720*3)},indent=2))
