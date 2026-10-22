import os,time,subprocess,pathlib,ctypes
from PIL import ImageGrab,ImageChops,ImageStat
root=pathlib.Path.cwd();ref=root/'临时/实验/nvidia-donut-samples';exe=ref/'bin/rt_particles.exe';u=ctypes.windll.user32;u.SetProcessDPIAware()
class RECT(ctypes.Structure):_fields_=[('left',ctypes.c_long),('top',ctypes.c_long),('right',ctypes.c_long),('bottom',ctypes.c_long)]
class POINT(ctypes.Structure):_fields_=[('x',ctypes.c_long),('y',ctypes.c_long)]
for backend in ['reference']:
 out=root/'documents/evidence/rhi-advanced-0.17/demo19-reference-trajectory';out.mkdir(parents=True,exist_ok=True);e=os.environ.copy();e['FLUXEL_REFERENCE_FRAME']='360';e['FLUXEL_REFERENCE_STATE']=str(out/'state.txt')
 if backend=='vulkan':e['VK_INSTANCE_LAYERS']='VK_LAYER_KHRONOS_validation'
 with (out/'live.log').open('w') as log:
  p=subprocess.Popen([str(exe),'-dx12'],cwd=ref/'bin',env=e,stdout=log,stderr=subprocess.STDOUT)
  try:
   time.sleep(9);handles=[]
   @ctypes.WINFUNCTYPE(ctypes.c_bool,ctypes.c_void_p,ctypes.c_void_p)
   def cb(h,_):
    pid=ctypes.c_ulong();u.GetWindowThreadProcessId(ctypes.c_void_p(h),ctypes.byref(pid))
    if pid.value==p.pid and u.IsWindowVisible(ctypes.c_void_p(h)):handles.append(h)
    return True
   u.EnumWindows(cb,0);assert handles;h=ctypes.c_void_p(handles[0]);r=RECT();c=RECT();u.GetWindowRect(h,ctypes.byref(r));u.GetClientRect(h,ctypes.byref(c));u.SetWindowPos(h,ctypes.c_void_p(-1),0,0,1000+r.right-r.left-c.right,709+r.bottom-r.top-c.bottom,0x40);time.sleep(2)
   c=RECT();u.GetClientRect(h,ctypes.byref(c));pt=POINT(0,0);u.ClientToScreen(h,ctypes.byref(pt));print(backend,c.right,c.bottom)
   def snap(name):ImageGrab.grab(bbox=(pt.x,pt.y,pt.x+c.right,pt.y+c.bottom)).save(out/name)
   def click(x,y):
    pos=x|(y<<16);u.PostMessageW(h,0x0200,0,pos);u.PostMessageW(h,0x0201,1,pos);u.PostMessageW(h,0x0202,0,pos);time.sleep(1)
   snap('initial.png')
   def drag(button, x1, y1, x2, y2):
    pos1=x1|(y1<<16);pos2=x2|(y2<<16)
    down, up, flag = (0x0201,0x0202,1) if button=='left' else (0x0207,0x0208,16)
    u.PostMessageW(h,0x0200,0,pos1);u.PostMessageW(h,down,flag,pos1);u.PostMessageW(h,0x0200,flag,pos2);time.sleep(1);u.PostMessageW(h,up,0,pos2);time.sleep(1)
   drag('left',650,350,730,390);snap('orbit.png')
   drag('middle',650,350,700,380);snap('pan.png')
   u.PostMessageW(h,0x0104,0x12,0x00380001)
   drag('middle',650,350,650,410)
   u.PostMessageW(h,0x0105,0x12,0xC0380001);time.sleep(1);snap('alt-pan.png')
   ptwheel=POINT(pt.x+650,pt.y+350)
   u.PostMessageW(h,0x020A,120<<16,ptwheel.x|(ptwheel.y<<16));time.sleep(1);snap('zoom.png')

   print('process',p.poll())
  finally:p.terminate();p.wait()

