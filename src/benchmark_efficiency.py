"""Measure batch-1 inference latency, throughput, parameters, and peak memory."""
from __future__ import annotations
import argparse,json,platform,time
from pathlib import Path
import numpy as np
import torch
from models import RGBReference,RGBDCandidate,UtilityRouter,route_predictions

def params(m): return sum(p.numel() for p in m.parameters())

def profile_flops(fn, device):
    """Return PyTorch-profiler FLOP estimates for one batch-1 forward pass."""
    activities=[torch.profiler.ProfilerActivity.CPU]
    if device.type=='cuda': activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.inference_mode(), torch.profiler.profile(
            activities=activities, record_shapes=True, with_flops=True) as prof:
        fn()
        if device.type=='cuda': torch.cuda.synchronize()
    total=sum(int(event.flops or 0) for event in prof.key_averages())
    return {'flops':total,'gflops':total/1e9,
            'method':'torch.profiler with_flops; supported operators only; batch size 1'}

def bench(fn,device,warmup,iters):
    with torch.inference_mode():
        for _ in range(warmup): fn()
        if device.type=='cuda': torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats(device)
        times=[]
        for _ in range(iters):
            if device.type=='cuda':
                st=torch.cuda.Event(enable_timing=True);en=torch.cuda.Event(enable_timing=True);st.record();fn();en.record();en.synchronize();times.append(st.elapsed_time(en))
            else:
                t=time.perf_counter();fn();times.append((time.perf_counter()-t)*1000)
        peak=torch.cuda.max_memory_allocated(device)/1024**2 if device.type=='cuda' else None
    a=np.asarray(times)
    return {'latency_ms_mean':float(a.mean()),'latency_ms_median':float(np.median(a)),
            'latency_ms_p95':float(np.quantile(a,.95)),'throughput_fps':float(1000/a.mean()),
            'peak_cuda_allocated_mib':peak}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--size',type=int,default=320)
    ap.add_argument('--threshold',type=float,required=True);ap.add_argument('--warmup',type=int,default=10)
    ap.add_argument('--iterations',type=int,default=100);ap.add_argument('--out',required=True)
    a=ap.parse_args();ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    size=ck.get('size',a.size);region=ck.get('region_size',16)
    rgbnet=RGBReference(False);rgbdnet=RGBDCandidate(False);router=UtilityRouter(region)
    rgbnet.load_state_dict(ck['reference']);rgbdnet.load_state_dict(ck['candidate']);router.load_state_dict(ck['router'])
    rgbnet.to(device).eval();rgbdnet.to(device).eval();router.to(device).eval()
    rgb=torch.randn(1,3,size,size,device=device);depth=torch.rand(1,1,size,size,device=device);valid=torch.ones_like(depth)
    def run_rgb(): return rgbnet(rgb)
    def run_rgbd(): return rgbdnet(rgb,depth,valid)
    def run_full():
        pr=rgbnet(rgb);pc=rgbdnet(rgb,depth,valid);s=router(pr,pc,depth,valid)
        return route_predictions(pr,pc,s,valid,a.threshold,region)[0]
    result={'checkpoint':str(Path(a.checkpoint).resolve()),'seed':ck.get('seed'),'input_size':[size,size],
            'batch_size':1,'device':str(device),'gpu':torch.cuda.get_device_name(0) if device.type=='cuda' else None,
            'torch':torch.__version__,'platform':platform.platform(),'iterations':a.iterations,'warmup':a.warmup,
            'parameter_count':{'rgb_reference':params(rgbnet),'rgbd_candidate':params(rgbdnet),'router':params(router),
                               'two_detectors_plus_router':params(rgbnet)+params(rgbdnet)+params(router)},
            'compute':{'rgb_reference':profile_flops(run_rgb,device),
                       'rgbd_candidate':profile_flops(run_rgbd,device),
                       'full_ucrr':profile_flops(run_full,device)},
            'timings':{'rgb_reference':bench(run_rgb,device,a.warmup,a.iterations),
                       'rgbd_candidate':bench(run_rgbd,device,a.warmup,a.iterations),
                       'full_ucrr':bench(run_full,device,a.warmup,a.iterations)},
            'threshold':a.threshold,'note':'Synchronized batch-1 inference; full_ucrr includes both detectors and the regional router.'}
    p=Path(a.out);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(result,indent=2),encoding='utf8');print(p)

if __name__=='__main__':main()
