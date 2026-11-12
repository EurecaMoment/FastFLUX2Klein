"""CPU contract tests using the real HTTP API and a deterministic fake GPU engine."""
import asyncio,base64,io,json,os,sys,time
from pathlib import Path
os.environ["FLUX2KLEIN_API_KEY"]="test-key"
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.argv=["server.py","--test-no-model"]
import httpx
from PIL import Image
import torch
import server
from optimizations import DiffusionController

checks=[]
def check(name,condition):
    print('CHECK',name,bool(condition),flush=True)
    checks.append({"check":name,"passed":bool(condition)})
    if not condition:
        raise AssertionError(name)

calls=[]
def fake_infer(jobs):
    calls.append([job.signature for job in jobs])
    return [(Image.new("RGB",(job.width,job.height),(job.seed%255,40*len(job.images),90)),
             {"batch_size":len(jobs),"seed":job.seed,"reference_images":len(job.images),
              "diffusion_cache":job.diffusion_cache,"predicted_blocks":int(job.diffusion_cache=="taylorseer")})
            for job in jobs]
server.engine.infer=fake_infer
server.engine.ready=True

def blob(color):
    out=io.BytesIO()
    Image.new("RGB",(512,512),color).save(out,"PNG")
    return out.getvalue()
red,blue=blob("red"),blob("blue")
headers={"Authorization":"Bearer test-key"}

async def main():
    async with server.app.router.lifespan_context(server.app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app),base_url="http://test") as client:
            r=await client.get("/v1/models")
            check("authentication_401_openai_error",r.status_code==401 and "error" in r.json())
            r=await client.get("/v1/models",headers=headers)
            check("model_list",r.status_code==200 and r.json()["data"][0]["id"]=="flux2klein")
            r=await client.post("/v1/images/generations",headers=headers,json={"prompt":"test","size":"512x512","seed":42,"n":2})
            d=r.json()
            check("generation_multi_output",r.status_code==200 and len(d["data"])==2)
            check("generation_distinct_seeds",d["data"][0]["b64_json"]!=d["data"][1]["b64_json"])
            r=await client.post("/v1/images/edits",headers=headers,
                data={"model":"flux2klein","prompt":"make it blue","size":"512x512","seed":"7"},
                files={"image":("red.png",red,"image/png")})
            check("single_edit_multipart",r.status_code==200 and r.json()["data"][0]["timings"]["reference_images"]==1)
            r=await client.post("/v1/images/edits",headers=headers,
                data={"prompt":"combine","size":"512x512","diffusion_cache":"taylorseer","n":"2"},
                files=[("image[]",("red.png",red,"image/png")),("image[]",("blue.png",blue,"image/png"))])
            check("multiple_edit_multipart",r.status_code==200 and len(r.json()["data"])==2 and r.json()["data"][0]["timings"]["reference_images"]==2)
            check("cache_option_reaches_engine",r.json()["data"][0]["timings"]["diffusion_cache"]=="taylorseer")
            encoded=base64.b64encode(red).decode()
            r=await client.post("/v1/images/edits",headers=headers,json={
                "prompt":"edit","size":"auto","images":[{"image_url":"data:image/png;base64,"+encoded}]})
            check("json_data_url_edit",r.status_code==200 and r.json()["size"]=="512x512")
            r=await client.post("/v1/images/generations",headers=headers,json={
                "prompt":"test","size":"512x512","response_format":"url","output_format":"jpeg","seed":11})
            url=r.json()["data"][0]["url"]
            x=await client.get(url)
            check("download_url_auth",x.status_code==401)
            x=await client.get(url,headers=headers)
            check("jpeg_download",x.status_code==200 and x.headers["content-type"]=="image/jpeg" and Image.open(io.BytesIO(x.content)).size==(512,512))
            # Remove only this test-created file, verified under this test staging root.
            ident=url.rsplit("/",1)[1]
            path=(server.OUT/ident).resolve()
            assert path.parent==server.OUT.resolve()
            path.unlink()
            for name,path,payload,status in [
                ("invalid_size","/v1/images/generations",{"prompt":"x","size":"513x512"},400),
                ("invalid_n","/v1/images/generations",{"prompt":"x","n":9},400),
                ("unknown_model","/v1/images/generations",{"prompt":"x","model":"unknown"},404),
                ("blank_prompt","/v1/images/generations",{"prompt":"  "},400),
                ("invalid_cache","/v1/images/generations",{"prompt":"x","diffusion_cache":"unknown"},400),
                ("missing_reference","/v1/images/edits",{"prompt":"x"},400),
                ("corrupt_base64","/v1/images/edits",{"prompt":"x","images":["%%%"]},400),
                ("corrupt_image","/v1/images/edits",{"prompt":"x","images":[base64.b64encode(b"not an image").decode()]},400),
                ("unsupported_mask","/v1/images/edits",{"prompt":"x","images":[encoded],"mask":"x"},400),
                ("too_many_images","/v1/images/edits",{"prompt":"x","images":[encoded]*9},400),
                ("ambiguous_image_fields","/v1/images/edits",{"prompt":"x","images":[encoded],"image":encoded},400),
            ]:
                r=await client.post(path,headers=headers,json=payload)
                check(name,r.status_code==status and "error" in r.json())
            check("compatible_batch_grouping",all(len(set(batch))==1 for batch in calls))
            check("request_counters_released",server.inflight==0 and server.requests_active==0)
            check("openapi_edit_route","/v1/images/edits" in (await client.get("/openapi.json")).json()["paths"])
    # Validate cache lifetime and actual prediction with small CPU modules.
    class IdentityBlock(torch.nn.Module):
        def forward(self,hidden_states,*args,**kwargs):
            return hidden_states*1.01+0.1
    class SmallTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.context_embedder=torch.nn.Linear(2,2,bias=False)
            self.pos_embed=torch.nn.Identity()
            self.single_transformer_blocks=torch.nn.ModuleList([IdentityBlock() for _ in range(24)])
    transformer=SmallTransformer()
    controller=DiffusionController(transformer)
    value=torch.ones(1,2)
    with controller.context("exact",4) as state:
        a=transformer.context_embedder(value)
        b=transformer.context_embedder(value)
        check("exact_cache_reuses_invariant",torch.equal(a,b) and state["memo_hits"]==1)
    check("exact_cache_scope_restored",controller.active is None)
    with controller.context("taylorseer",4) as state:
        for step in range(4):
            for block in transformer.single_transformer_blocks:
                block(value)
            controller.on_step_end(None,step,None,{})
        check("taylor_prediction_exercised",state["predicted_blocks"]==12)
    with controller.context("taylorseer",4) as state:
        check("taylor_state_reset",not state["history"] and state["step"]==0)
    print(json.dumps({"passed":len(checks),"checks":checks},indent=2))
asyncio.run(main())


