"""HTTP throughput benchmark with a fresh prompt namespace on every run."""
import argparse,asyncio,base64,datetime,hashlib,json,os,statistics,time,uuid
from pathlib import Path
import httpx

parser=argparse.ArgumentParser()
parser.add_argument("--concurrency",type=int,default=64)
parser.add_argument("--requests",type=int,default=64)
parser.add_argument("--size",default="512x512")
parser.add_argument("--unique-prompts",action="store_true")
parser.add_argument("--diffusion-cache",choices=("none","exact","taylorseer"),default="exact")
parser.add_argument("--output",default="benchmark-v2.json")
parser.add_argument("--base-url",default="http://127.0.0.1:8036")
args=parser.parse_args()
if min(args.concurrency,args.requests)<1 or args.concurrency>256:
    parser.error("Use positive request counts and concurrency <=256")
key=os.environ["FLUX2KLEIN_API_KEY"]
run_id=uuid.uuid4().hex
async def main():
    semaphore=asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(trust_env=False,timeout=900,limits=httpx.Limits(
            max_connections=args.concurrency,max_keepalive_connections=args.concurrency)) as client:
        before=(await client.get(args.base_url+"/health")).json()
        if not before.get("ready"):
            raise RuntimeError("Service is not ready")
        async def one(index,warmup=False):
            async with semaphore:
                prompt="A red ceramic teapot on a wooden table in soft daylight"
                if args.unique_prompts:
                    prompt+=f", a card with reference code {run_id}-{index}"
                started=time.perf_counter()
                response=await client.post(args.base_url+"/v1/images/generations",
                    headers={"Authorization":"Bearer "+key},json={
                        "model":"flux2klein","prompt":prompt,"seed":900000+index,
                        "size":args.size,"num_inference_steps":4,"diffusion_cache":args.diffusion_cache,
                        "response_format":"b64_json"})
                seconds=time.perf_counter()-started
                if response.status_code!=200:
                    return {"status":response.status_code,"seconds":seconds,"error":response.text[:300]}
                item=response.json()["data"][0]
                data=base64.b64decode(item["b64_json"])
                return {"status":200,"seconds":seconds,"sha256":hashlib.sha256(data).hexdigest(),
                        "bytes":len(data),**item["timings"]}
        # A distinct warmup prompt avoids warming any measured unique prompt.
        await one(999999,True)
        started=time.perf_counter()
        rows=await asyncio.gather(*(one(i) for i in range(args.requests)))
        wall=time.perf_counter()-started
        after=(await client.get(args.base_url+"/health")).json()
        ok=[r for r in rows if r["status"]==200]
        lat=sorted(r["seconds"] for r in ok)
        def percentile(p):
            return lat[min(len(lat)-1,int((len(lat)-1)*p))] if lat else None
        summary={
            "tested_at":datetime.datetime.now().isoformat(),"run_id":run_id,
            "size":args.size,"concurrency":args.concurrency,"requests":args.requests,
            "successes":len(ok),"unique_images":len({r["sha256"] for r in ok}),
            "unique_prompts":args.unique_prompts,"prompt_cache_hits":sum(r["prompt_cache_hit"] for r in ok),
            "diffusion_cache":args.diffusion_cache,"wall_seconds":wall,"images_per_second":len(ok)/wall,
            "latency_p50_seconds":percentile(.5),"latency_p95_seconds":percentile(.95),
            "latency_max_seconds":max(lat) if lat else None,
            "batch_sizes":sorted({r["batch_size"] for r in ok}),
            "oom_splits_delta":after.get("oom_splits",0)-before.get("oom_splits",0),
            "health_before":before,"health_after":after,"rows":rows}
        Path(args.output).write_text(json.dumps(summary,indent=2))
        print(json.dumps({k:v for k,v in summary.items() if k not in ("rows","health_before","health_after")},indent=2),flush=True)
asyncio.run(main())


