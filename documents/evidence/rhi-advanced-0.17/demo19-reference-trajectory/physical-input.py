import os,time,subprocess,pathlib,ctypes
from PIL import ImageGrab
root=pathlib.Path.cwd();ref=root/'临时/实验/nvidia-donut-samples';out=root/'documents/evidence/rhi-advanced-0.17/demo19-reference-trajectory';u=ctypes.windll.user32;u.SetProcessDPIAware()
class RECT(ctypes.Structure):_fields_=[('left',ctypes.c_long),('top',ctypes.c_long),('right',ctypes.c_long),('bottom',ctypes.c_long)]
class POINT(ctypes.Structure):_fields_=[('x',ctypes.c_long),('y',ctypes.c_long)]
for backend in ['reference','dx12','vulkan']:
 folder=out/backend;folder.mkdir(exist_ok=True);e=os.environ.copy();e['FLUXEL_DX12_DEBUG']='1'
 if backend=='reference':
  exe=ref/'bin/rt_particles.exe';args=['-dx12'];cwd=ref/'bin';state=folder/'state.txt';state.unlink(missing_ok=True);e['FLUXEL_REFERENCE_FRAME']='360';e['FLUXEL_REFERENCE_STATE']=str(state)
 else:
  exe=root/'target/debug/examples/19_ray_query_particles.exe';args=['--backend',backend];cwd=folder;e['FLUXEL_RAY_QUERY_PRESENT']='1';e['FLUXEL_RAY_QUERY_TRACE']='1'
  if backend=='vulkan':e['VK_INSTANCE_LAYERS']='VK_LAYER_KHRONOS_validation'
 with (folder/'live.log').open('w') as log:
  p=subprocess.Popen([str(exe),*args],cwd=cwd,env=e,stdout=log,stderr=subprocess.STDOUT)
  try:
   time.sleep(7);hs=[]
   @ctypes.WINFUNCTYPE(ctypes.c_bool,ctypes.c_void_p,ctypes.c_void_p)
   def cb(h,_):
    pid=ctypes.c_ulong();u.GetWindowThreadProcessId(ctypes.c_void_p(h),ctypes.byref(pid))
    if pid.value==p.pid and u.IsWindowVisible(ctypes.c_void_p(h)):hs.append(h)
    return True
   u.EnumWindows(cb,0);assert hs;h=ctypes.c_void_p(hs[0]);r=RECT();c=RECT();u.GetWindowRect(h,ctypes.byref(r));u.GetClientRect(h,ctypes.byref(c));u.SetWindowPos(h,ctypes.c_void_p(-1),0,0,1000+r.right-r.left-c.right,709+r.bottom-r.top-c.bottom,0x40);u.SetForegroundWindow(h);time.sleep(2);u.GetClientRect(h,ctypes.byref(c));pt=POINT(0,0);u.ClientToScreen(h,ctypes.byref(pt));assert (c.right,c.bottom)==(1000,709)
   if backend=='reference':
    for _ in range(200):
     if state.exists():break
     time.sleep(.1)
    assert state.exists()
   else:
    u.keybd_event(0x20,0,0,0);u.keybd_event(0x20,0,2,0);time.sleep(2)
   def snap(n):ImageGrab.grab(bbox=(pt.x,pt.y,pt.x+c.right,pt.y+c.bottom)).save(folder/n)
   def move(x,y):u.SetCursorPos(pt.x+x,pt.y+y);time.sleep(.3)
   def drag(button,x1,y1,x2,y2):
    move(x1,y1);down,up=(2,4) if button=='left' else (32,64);u.mouse_event(down,0,0,0,0);time.sleep(.3);move(x2,y2);time.sleep(.6);u.mouse_event(up,0,0,0,0);time.sleep(1)
   snap('initial.png');drag('left',650,350,730,390);snap('orbit.png');drag('middle',650,350,700,380);snap('pan.png');u.keybd_event(0x12,0,0,0);drag('middle',650,350,650,410);u.keybd_event(0x12,0,2,0);time.sleep(1);snap('alt-pan.png');move(650,350);u.mouse_event(0x0800,0,0,120,0);time.sleep(1);snap('zoom.png');print(backend,'alive',p.poll(),flush=True)
  finally:p.terminate();p.wait()
