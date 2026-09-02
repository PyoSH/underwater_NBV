import json, sys
import numpy as np
def sq(x):
    x=np.asarray(x,float); return x[:,0,:] if x.ndim==3 else x
for p in sys.argv[1:]:
    d=json.load(open(p)); t=np.asarray(d["t"],float); dt=t[1]-t[0]
    a=sq(d["action"]); w=sq(d["omega_b"]); eu=sq(d["euler_deg"]); va=sq(d["v_actual"])
    s=int(2.0/dt)   # 2초 이후
    print(f"=== {p.split('/')[-1]}  window {t[s]:.1f}~{t[-1]:.1f}s  N={len(t)-s}")
    print(f"  euler range roll[{eu[s:,0].min():+.0f},{eu[s:,0].max():+.0f}] "
          f"pitch[{eu[s:,1].min():+.0f},{eu[s:,1].max():+.0f}] yaw[{eu[s:,2].min():+.0f},{eu[s:,2].max():+.0f}] deg")
    print(f"  u mean={va[s:,0].mean():+.3f} std={va[s:,0].std():.3f}   |v| mean={np.linalg.norm(va[s:],axis=-1).mean():.3f}")
    for name, sig in [("a_roll",a[s:,3]),("a_pitch",a[s:,4]),("a_yaw",a[s:,5]),
                      ("a_heave",a[s:,2]),("w_p",w[s:,0]),("w_q",w[s:,1]),("w_r",w[s:,2])]:
        x=sig-sig.mean()
        # zero crossing rate -> 기본 주파수 (반주기 개수/2)
        zc=(np.diff(np.sign(x))!=0).sum()
        f_zc=zc/2/((len(x)-1)*dt)
        X=np.abs(np.fft.rfft(x*np.hanning(len(x))))**2
        f=np.fft.rfftfreq(len(x),dt); X[0]=0
        k=X.argmax()
        # 중심 주파수 (파워 가중)
        fc=(f*X).sum()/X.sum()
        print(f"  {name:8s} rms={x.std():6.3f}  f_peak={f[k]:5.2f}Hz  f_zc={f_zc:5.2f}Hz  f_centroid={fc:5.2f}Hz")
