"""16 real concurrent HTTP clients, fixed 60-second closed-loop load."""
import argparse
import asyncio
import time
import numpy as np
import httpx
from bayan.models.training import ROOT,write_json

async def run(url,duration,concurrency):
    async with httpx.AsyncClient(timeout=120,limits=httpx.Limits(max_connections=concurrency,max_keepalive_connections=concurrency)) as client:
        health=await client.get(url+'/health');health.raise_for_status()
        payload={'text':'الخدمة ممتازة ولكن التأخير طويل'}
        for _ in range(20):
            response=await client.post(url+'/v1/classify',json=payload);response.raise_for_status()
        stop=time.perf_counter()+duration
        async def worker():
            results=[]
            while time.perf_counter()<stop:
                start=time.perf_counter()
                try:
                    response=await client.post(url+'/v1/classify',json=payload)
                    ok=response.status_code==200 and 'topic' in response.json()
                except (httpx.HTTPError,ValueError):ok=False
                results.append(((time.perf_counter()-start)*1000,ok))
            return results
        start=time.perf_counter();batches=await asyncio.gather(*(worker() for _ in range(concurrency)));elapsed=time.perf_counter()-start
    results=[r for batch in batches for r in batch];latencies=[r[0] for r in results]
    return {'duration_seconds':duration,'elapsed_seconds':elapsed,'concurrency':concurrency,'requests':len(results),'errors':sum(not r[1] for r in results),
        'p50_ms':float(np.median(latencies)),'p99_ms':float(np.percentile(latencies,99)), 'requests_per_second':len(results)/elapsed,
        'target_met':np.percentile(latencies,99)<=40 and all(r[1] for r in results),'canaries':health.json()['canaries'],
        'payload':payload,'method':'httpx persistent clients; closed-loop 16 concurrent requests for 60 seconds, equivalent load shape to supplied hey command; includes HTTP, queueing, preprocessing and model inference'}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--url',default='http://127.0.0.1:8000');parser.add_argument('--duration',type=int,default=60);parser.add_argument('--concurrency',type=int,default=16);args=parser.parse_args()
    report=asyncio.run(run(args.url,args.duration,args.concurrency));write_json(ROOT/'artifacts/lab7/http_load.json',report);print(report)

if __name__=='__main__':main()
