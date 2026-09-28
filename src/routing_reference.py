"""NumPy reference for UCRR routing/calibration; not a neural training pipeline."""
import numpy as np

def blocks(x, size=16):
    x=np.asarray(x,dtype=np.float64)
    n,h,w=x.shape
    assert h%size==0 and w%size==0
    return x.reshape(n,h//size,size,w//size,size).mean(axis=(2,4))

def route(reference,candidate,score,valid,size=16,threshold=0):
    mask=(np.asarray(score)>threshold)&(blocks(valid,size)>=0.5)
    full=np.repeat(np.repeat(mask,size,axis=1),size,axis=2)
    return np.where(full,candidate,reference),mask

def calibrate(reference,candidate,score,valid,truth,size=16,budget=.005):
    utility=blocks(np.abs(reference-truth)-np.abs(candidate-truth),size)
    records=[]
    for threshold in np.linspace(-1,1,101):
        _,mask=route(reference,candidate,score,valid,size,threshold)
        harm=float(np.mean(mask*np.maximum(-utility,0)))
        gain=float(np.mean(mask*np.maximum(utility,0)))
        records.append((float(threshold),harm,gain,float(mask.mean())))
    feasible=[row for row in records if row[1]<=budget]
    best=max(feasible,key=lambda row:(row[2],row[0]))
    return best,records

def self_check():
    rng=np.random.default_rng(17)
    ref=rng.random((3,32,32));can=rng.random(ref.shape)
    y=(rng.random(ref.shape)>.5).astype(float)
    score=rng.uniform(-1,1,(3,2,2));valid=np.ones_like(ref)
    p,a=route(ref,can,score,valid)
    u=blocks(np.abs(ref-y)-np.abs(can-y))
    h=np.mean(a*np.maximum(-u,0));g=np.mean(a*np.maximum(u,0))
    assert np.isclose(np.mean(np.abs(p-y))-np.mean(np.abs(ref-y)),h-g)
    missing,_=route(ref,can,score,np.zeros_like(ref))
    assert np.array_equal(missing,ref)
    none,_=route(ref,can,score,valid,threshold=1)
    assert np.array_equal(none,ref)
    best,_=calibrate(ref,can,score,valid,y,budget=0)
    assert best[1]==0
    print('PASS: error decomposition, exact missing-depth fallback, all-RGB endpoint, calibration feasibility')

if __name__=='__main__': self_check()
