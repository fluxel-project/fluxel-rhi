import os,time,subprocess,pathlib,ctypes,sys
from PIL import ImageGrab
root=pathlib.Path.cwd();ref=root/'临时/实验/nvidia-donut-samples';out=root/'target/demo19-candidate-count';out.mkdir(exist_ok=True);state=out/'state.txt';tlas=out/'tlas.txt';state.unlink(missing_ok=True);tlas.unlink(missing_ok=True)
e=os.environ.copy();e['FLUXEL_REFERENCE_FRAME']=sys.argv[1];e['FLUXEL_REFERENCE_STATE']=str(state);e['FLUXEL_REFERENCE_TLAS']=str(tlas);p=subprocess.Popen([str(ref/'bin/rt_particles.exe'),'-dx12'],cwd=ref/'bin',env=e);u=ctypes.windll.user32;u.SetProcessDPIAware()
try:
 for i in range(200):
  if tlas.exists():break
  time.sleep(.1)
 else:raise Exception('TLAS dump missing')
 matches=[]
 @ctypes.WINFUNCTYPE(ctypes.c_bool,ctypes.c_void_p,ctypes.c_void_p)
 def cb(hwnd,_):
  pid=ctypes.c_ulong();u.GetWindowThreadProcessId(ctypes.c_void_p(hwnd),ctypes.byref(pid))
  if pid.value==p.pid and u.IsWindowVisible(ctypes.c_void_p(hwnd)):matches.append(hwnd)
  return True
 u.EnumWindows(cb,0)
 class RECT(ctypes.Structure):_fields_=[('left',ctypes.c_long),('top',ctypes.c_long),('right',ctypes.c_long),('bottom',ctypes.c_long)]
 class POINT(ctypes.Structure):_fields_=[('x',ctypes.c_long),('y',ctypes.c_long)]
 h=ctypes.c_void_p(matches[0]);r=RECT();u.GetWindowRect(h,ctypes.byref(r));c=RECT();u.GetClientRect(h,ctypes.byref(c));ok=u.SetWindowPos(h,ctypes.c_void_p(-1),0,0,1000+(r.right-r.left-c.right),709+(r.bottom-r.top-c.bottom),0x40);print('setwindow',ok);time.sleep(4);u.GetClientRect(h,ctypes.byref(c));pt=POINT(0,0);u.ClientToScreen(h,ctypes.byref(pt));print('rect',pt.x,pt.y,c.right,c.bottom);ImageGrab.grab(bbox=(pt.x,pt.y,pt.x+c.right,pt.y+c.bottom)).save(out/'candidates.png')
finally:p.terminate();p.wait()
