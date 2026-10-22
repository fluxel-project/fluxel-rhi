import os,time,subprocess,pathlib,ctypes
from PIL import ImageGrab,ImageChops,ImageStat
root=pathlib.Path.cwd();exe=root/'target/debug/examples/19_ray_query_particles.exe';u=ctypes.windll.user32;u.SetProcessDPIAware()
class RECT(ctypes.Structure):_fields_=[('left',ctypes.c_long),('top',ctypes.c_long),('right',ctypes.c_long),('bottom',ctypes.c_long)]
class POINT(ctypes.Structure):_fields_=[('x',ctypes.c_long),('y',ctypes.c_long)]
for backend in ['dx12','vulkan']:
 out=root/'target/demo19-ui'/backend;e=os.environ.copy();e['FLUXEL_RAY_QUERY_PRESENT']='1';e['FLUXEL_RAY_QUERY_TRACE']='1';e['FLUXEL_DX12_DEBUG']='1'
 if backend=='vulkan':e['VK_INSTANCE_LAYERS']='VK_LAYER_KHRONOS_validation'
 with (out/'live.log').open('w') as log:
  p=subprocess.Popen([str(exe),'--backend',backend],cwd=out,env=e,stdout=log,stderr=subprocess.STDOUT)
  try:
   time.sleep(3);handles=[]
   @ctypes.WINFUNCTYPE(ctypes.c_bool,ctypes.c_void_p,ctypes.c_void_p)
   def cb(h,_):
    pid=ctypes.c_ulong();u.GetWindowThreadProcessId(ctypes.c_void_p(h),ctypes.byref(pid))
    if pid.value==p.pid and u.IsWindowVisible(ctypes.c_void_p(h)):handles.append(h)
    return True
   u.EnumWindows(cb,0);assert handles;h=ctypes.c_void_p(handles[0]);u.SetWindowPos(h,ctypes.c_void_p(-1),0,0,0,0,0x43);time.sleep(1)
   c=RECT();u.GetClientRect(h,ctypes.byref(c));pt=POINT(0,0);u.ClientToScreen(h,ctypes.byref(pt));print(backend,c.right,c.bottom)
   def snap(name):ImageGrab.grab(bbox=(pt.x,pt.y,pt.x+c.right,pt.y+c.bottom)).save(out/name)
   def click(x,y):
    pos=x|(y<<16);u.PostMessageW(h,0x0200,0,pos);u.PostMessageW(h,0x0201,1,pos);u.PostMessageW(h,0x0202,0,pos);time.sleep(1)
   click(35,65);snap('paused-1.png');time.sleep(1);snap('paused-2.png');click(50,160);click(50,415);snap('beam-logo.png');click(50,160);click(45,223);snap('basis-primary.png');click(50,160);click(45,250);click(40,290);snap('avt-secondary-mlab8.png')
   print('process',p.poll())
  finally:p.terminate();p.wait()
