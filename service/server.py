"""OpenAI Images API for resident FLUX.2 Klein generation and reference editing."""
import os
os.environ.setdefault("HF_HUB_OFFLINE","1")
os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS","4")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
import argparse
import asyncio
import base64
import binascii
import hashlib
import hmac
import io
import json
import logging
import re
import secrets
import subprocess
import sys
import time
import uuid
import warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
torch._dynamo.config.cache_size_limit=256
torch._dynamo.config.accumulated_cache_size_limit=2048
from native_attention import install
install()
from engine import Engine, Job, VERSION
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from starlette.datastructures import UploadFile
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from PIL import Image, ImageOps, UnidentifiedImageError
import uvicorn

parser=argparse.ArgumentParser()
parser.add_argument("--host",default="0.0.0.0")
parser.add_argument("--port",type=int,default=8036)
parser.add_argument("--model",default=os.environ.get("FLUX2_MODEL_PATH",str(Path.home()/"models/FLUX.2-klein-9B")))
parser.add_argument("--served-model-name",default="flux2klein")
parser.add_argument("--gpu-memory-gb",type=float,default=40)
parser.add_argument("--max-concurrent-requests",type=int,default=256)
parser.add_argument("--max-batch-size",type=int,default=16)
parser.add_argument("--max-batch-size-1024",type=int,default=4)
parser.add_argument("--batch-wait-ms",type=float,default=8)
parser.add_argument("--compile",choices=("regional","none"),default="regional")
parser.add_argument("--compile-vae",action=argparse.BooleanOptionalAction,default=True)
parser.add_argument("--fuse-qkv",action=argparse.BooleanOptionalAction,default=False)
parser.add_argument("--park-unused-text-weights",action=argparse.BooleanOptionalAction,default=True)
parser.add_argument("--warm-edits",action=argparse.BooleanOptionalAction,default=True)
parser.add_argument("--cuda-graph",choices=("auto","off"),default="auto")
parser.add_argument("--text-batch-size",type=int,default=4)
parser.add_argument("--cpu-threads",type=int,default=8)
parser.add_argument("--prompt-cache-gb",type=float,default=1)
parser.add_argument("--reference-cache-gb",type=float,default=1)
parser.add_argument("--max-reference-images",type=int,default=8)
parser.add_argument("--max-reference-pixels",type=int,default=2*1024*1024)
parser.add_argument("--request-timeout",type=float,default=600)
parser.add_argument("--test-no-model",action="store_true",help=argparse.SUPPRESS)
args=parser.parse_args()
ROOT=Path(__file__).resolve().parent
OUT=ROOT/"outputs"
OUT.mkdir(exist_ok=True)
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR",str(Path(os.environ.get("XDG_CACHE_HOME",str(Path.home()/".cache")))/"fastflux2klein/inductor"))
API_KEY=os.environ.get("FLUX2KLEIN_API_KEY","")
if not API_KEY:
    raise RuntimeError("Set FLUX2KLEIN_API_KEY before starting the API")
log=logging.getLogger("flux2klein")
MAX_IMAGE_BYTES=16*1024*1024
MAX_BODY_BYTES=64*1024*1024
Image.MAX_IMAGE_PIXELS=16*1024*1024

class Generation(BaseModel):
    model_config=ConfigDict(extra="forbid")
    model: str="flux2klein"
    prompt: str=Field(min_length=1,max_length=10000)
    n: int=Field(default=1,ge=1,le=8)
    size: str="1024x1024"
    response_format: Literal["b64_json","url"]="b64_json"
    seed: int | None=Field(default=None,ge=0,le=2**63-1)
    num_inference_steps: int=Field(default=4,ge=1,le=50)
    cache_prompt: bool=True
    cache_reference: bool=True
    diffusion_cache: Literal["none","exact","taylorseer"]="exact"
    cache_interval: int=Field(default=2,ge=2,le=8)
    cache_warmup_steps: int=Field(default=2,ge=2,le=10)
    cache_blocks: int=Field(default=12,ge=1,le=24)
    output_format: Literal["png","jpeg","webp"]="png"
    output_compression: int=Field(default=90,ge=1,le=100)
    user: str | None=Field(default=None,max_length=256)

    @field_validator("prompt")
    @classmethod
    def nonblank(cls,value):
        if not value.strip():
            raise ValueError("prompt must contain text")
        return value

    @field_validator("size")
    @classmethod
    def size_supported(cls,value):
        if value=="auto":
            return value
        match=re.fullmatch(r"([0-9]{3,4})x([0-9]{3,4})",value)
        if not match:
            raise ValueError("size must be WIDTHxHEIGHT or auto")
        width,height=map(int,match.groups())
        if min(width,height)<256 or max(width,height)>1536 or width%16 or height%16 or width*height>1024*1024:
            raise ValueError("dimensions must be multiples of 16, 256..1536, with at most 1048576 pixels")
        return value

engine=Engine(args)
gpu_pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix="gpu")
encode_pool=ThreadPoolExecutor(max_workers=8,thread_name_prefix="image-io")
pending=deque()
changed=asyncio.Event()
inflight=0
requests_active=0
encode_tasks=set()
upload_semaphore=asyncio.Semaphore(8)
started_at=time.time()
memory_status={"timestamp":0.,"gb":None}

def authorize(authorization):
    if not authorization or not hmac.compare_digest(authorization,"Bearer "+API_KEY):
        raise HTTPException(401,"Invalid API key")

def openai_error(status,message,param=None,code=None,headers=None):
    error_type="server_error" if status>=500 else "invalid_request_error"
    if status==401:
        error_type="authentication_error"
    if status==429:
        error_type="rate_limit_error"
    return JSONResponse({"error":{"message":message,"type":error_type,"param":param,"code":code}},
                        status_code=status,headers=headers)

def validate_body(data):
    try:
        return Generation.model_validate(data)
    except ValidationError as exc:
        first=exc.errors(include_url=False,include_context=False)[0]
        param=".".join(map(str,first["loc"]))
        raise HTTPException(400,{"message":first["msg"],"param":param,"code":"invalid_parameter"})

def parse_data_image(value):
    if isinstance(value,dict):
        value=value.get("image_url",value.get("b64_json"))
        if isinstance(value,dict):
            value=value.get("url")
    if not isinstance(value,str):
        raise HTTPException(400,{"message":"Each image must be an uploaded file or a base64/data URL","param":"image"})
    if value.startswith("data:"):
        if not re.match(r"^data:image/(png|jpeg|jpg|webp);base64,",value,re.I):
            raise HTTPException(400,{"message":"Unsupported image data URL","param":"image"})
        value=value.split(",",1)[1]
    if len(value)>((MAX_IMAGE_BYTES+2)//3)*4:
        raise HTTPException(413,"Input image exceeds 16 MiB")
    try:
        return base64.b64decode(value,validate=True)
    except (ValueError,binascii.Error):
        raise HTTPException(400,{"message":"Invalid base64 image","param":"image"})

def decode_images(blobs):
    images=[]
    keys=[]
    total_pixels=0
    for blob in blobs:
        if not blob or len(blob)>MAX_IMAGE_BYTES:
            raise HTTPException(413,"Each image must be nonempty and at most 16 MiB")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error",Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(blob)) as source:
                    if source.format not in ("PNG","JPEG","WEBP"):
                        raise HTTPException(400,{"message":"Supported images are PNG, JPEG, WEBP","param":"image"})
                    if source.width*source.height>Image.MAX_IMAGE_PIXELS:
                        raise HTTPException(413,"Input image has too many pixels")
                    image=ImageOps.exif_transpose(source).convert("RGB")
            if min(image.size)<16:
                raise HTTPException(400,{"message":"Reference dimensions must be at least 16 pixels","param":"image"})
            # Match the installed pipeline's own per-reference area limit.
            if image.width*image.height>1024*1024:
                scale=(1024*1024/(image.width*image.height))**.5
                image=image.resize((max(16,int(image.width*scale)//16*16),
                                    max(16,int(image.height*scale)//16*16)),Image.Resampling.LANCZOS)
            total_pixels+=image.width*image.height
            if total_pixels>args.max_reference_pixels:
                raise HTTPException(400,{"message":"Total reference image area exceeds 2097152 pixels after resizing","param":"image"})
            digest=hashlib.sha256()
            digest.update(f"{image.width}x{image.height}:RGB:".encode())
            digest.update(image.tobytes())
            images.append(image)
            keys.append(digest.hexdigest())
        except HTTPException:
            raise
        except (UnidentifiedImageError,OSError,ValueError,Image.DecompressionBombError,Image.DecompressionBombWarning):
            raise HTTPException(400,{"message":"Cannot decode the reference image","param":"image"})
    return tuple(images),tuple(keys)

def dimensions(body,images):
    if body.size!="auto":
        return tuple(map(int,body.size.split("x")))
    if not images:
        return 1024,1024
    image=images[0]
    width=max(256,image.width//16*16)
    height=max(256,image.height//16*16)
    scale=min(1.,(1024*1024/(width*height))**.5,1536/max(width,height))
    return max(256,int(width*scale)//16*16),max(256,int(height*scale)//16*16)

def encode_image(image,job):
    if isinstance(image,torch.Tensor):
        image=Image.fromarray(image.numpy())
    stream=io.BytesIO()
    if job.output_format=="png":
        image.save(stream,format="PNG",compress_level=1)
    elif job.output_format=="jpeg":
        image.convert("RGB").save(stream,format="JPEG",quality=job.output_compression)
    else:
        image.save(stream,format="WEBP",quality=job.output_compression,method=2)
    payload=stream.getvalue()
    ident=uuid.uuid4().hex
    if job.response_format=="url":
        (OUT/(ident+"."+job.output_format)).write_bytes(payload)
        return ident,None
    return ident,base64.b64encode(payload).decode("ascii")

async def complete(job,image,metrics):
    global inflight
    try:
        if not job.future.cancelled():
            ident,b64=await asyncio.get_running_loop().run_in_executor(encode_pool,encode_image,image,job)
            metrics["server_total_seconds"]=round(time.perf_counter()-job.submitted,6)
            if not job.future.done():
                job.future.set_result((ident,b64,metrics))
    except Exception as exc:
        if not job.future.done():
            job.future.set_exception(exc)
    finally:
        inflight-=1

async def dispatcher():
    global inflight
    loop=asyncio.get_running_loop()
    while True:
        if not pending:
            changed.clear()
            await changed.wait()
        while pending and pending[0].future.cancelled():
            pending.popleft()
            inflight-=1
        if not pending:
            continue
        # Accumulate briefly only when the oldest job has not already waited.
        remaining=args.batch_wait_ms/1000-(time.perf_counter()-pending[0].submitted)
        if remaining>0:
            await asyncio.sleep(remaining)
        first=pending.popleft()
        if first.future.cancelled():
            inflight-=1
            continue
        batch=[first]
        limit=engine.batch_limit(first)
        for job in list(pending):
            if len(batch)>=limit:
                break
            if not job.future.cancelled() and job.signature==first.signature:
                pending.remove(job)
                batch.append(job)
        try:
            results=await loop.run_in_executor(gpu_pool,engine.infer,batch)
            for job,(image,metrics) in zip(batch,results):
                task=asyncio.create_task(complete(job,image,metrics))
                encode_tasks.add(task)
                task.add_done_callback(encode_tasks.discard)
        except Exception as exc:
            log.exception("GPU batch failed")
            for job in batch:
                if not job.future.done():
                    job.future.set_exception(exc)
                inflight-=1

@asynccontextmanager
async def lifespan(app):
    loop=asyncio.get_running_loop()
    async def initialize():
        try:
            await loop.run_in_executor(gpu_pool,engine.initialize)
        except Exception as exc:
            engine.error=f"{type(exc).__name__}: {exc}"
            engine.phase="failed"
            log.exception("Model initialization failed")
    init=asyncio.create_task(initialize()) if not args.test_no_model else None
    dispatch=asyncio.create_task(dispatcher())
    try:
        yield
    finally:
        if init:
            init.cancel()
        dispatch.cancel()
        for job in pending:
            if not job.future.done():
                job.future.cancel()
        if encode_tasks:
            await asyncio.gather(*encode_tasks,return_exceptions=True)
        gpu_pool.shutdown(wait=False)
        encode_pool.shutdown(wait=False)

app=FastAPI(title="FLUX.2 Klein OpenAI Images API",version=VERSION,lifespan=lifespan)

@app.exception_handler(HTTPException)
async def http_error(request,exc):
    detail=exc.detail
    if isinstance(detail,dict):
        return openai_error(exc.status_code,detail.get("message",str(detail)),
                            detail.get("param"),detail.get("code"),exc.headers)
    return openai_error(exc.status_code,str(detail),headers=exc.headers)

@app.exception_handler(RequestValidationError)
async def request_error(request,exc):
    first=exc.errors()[0]
    return openai_error(400,first["msg"],".".join(map(str,first["loc"][1:])),"invalid_parameter")

@app.middleware("http")
async def request_limits(request,call_next):
    if request.method=="POST":
        try:
            length=int(request.headers.get("content-length","0"))
        except ValueError:
            return openai_error(400,"Invalid Content-Length")
        if length>MAX_BODY_BYTES:
            return openai_error(413,"Request body exceeds 64 MiB")
        # Count streaming/chunked bodies as well; do not rely on Content-Length.
        received=0
        original=request._receive
        async def limited_receive():
            nonlocal received
            message=await original()
            received+=len(message.get("body",b""))
            if received>MAX_BODY_BYTES:
                raise HTTPException(413,"Request body exceeds 64 MiB")
            return message
        request._receive=limited_receive
    response=await call_next(request)
    response.headers["X-FLUX-Version"]=VERSION
    return response

@app.get("/health")
def health():
    now=time.monotonic()
    if engine.ready and now-memory_status["timestamp"]>=2:
        try:
            output=subprocess.check_output(
                ["nvidia-smi","--query-compute-apps=pid,used_gpu_memory","--format=csv,noheader,nounits"],
                text=True,timeout=3)
            for line in output.splitlines():
                pid,used=line.split(",")
                if int(pid)==os.getpid():
                    memory_status["gb"]=int(used)*1024**2/1e9
            memory_status["timestamp"]=now
        except (OSError,ValueError,subprocess.SubprocessError):
            pass
    return JSONResponse({
        "ready":engine.ready,"phase":engine.phase,"error":engine.error,"version":VERSION,
        "model":args.served_model_name,"visible_gpu_devices":os.environ.get("CUDA_VISIBLE_DEVICES","all"),
        "memory_budget_gb":args.gpu_memory_gb,
        "process_gpu_memory_gb":memory_status["gb"],
        "queued":len(pending),"inflight_images":inflight,"active_requests":requests_active,
        "max_concurrent_requests":args.max_concurrent_requests,
        "batch_limits":{"512":args.max_batch_size,"1024":args.max_batch_size_1024},
        "compile":args.compile,
        "prompt_cache_entries":len(engine.prompt_cache.entries),
        "prompt_cache_ram_gb":round(engine.prompt_cache.bytes/1e9,3),
        "reference_cache_entries":len(engine.reference_cache.entries),
        "reference_cache_ram_gb":round(engine.reference_cache.bytes/1e9,3),
        "features":{"generation":True,"single_image_edit":True,"multi_reference_edit":True,
                    "max_reference_images":args.max_reference_images,
                    "max_reference_total_pixels":args.max_reference_pixels,
                    "diffusion_cache_modes":["none","exact","taylorseer"]},
        "optimizations":engine.optimizations,"startup_checks":engine.startup_checks,
        "uptime_seconds":round(time.time()-started_at),**engine.stats,
    },status_code=200 if engine.ready else 503)

@app.get("/v1/models")
def models(authorization:str|None=Header(default=None)):
    authorize(authorization)
    return {"object":"list","data":[{"id":args.served_model_name,"object":"model",
                                    "created":0,"owned_by":"local"}]}

@app.get("/v1/files/{ident}.{extension}")
def image_file(ident:str,extension:str,authorization:str|None=Header(default=None)):
    authorize(authorization)
    if not re.fullmatch(r"[0-9a-f]{32}",ident) or extension not in ("png","jpeg","webp"):
        raise HTTPException(404,"Image not found")
    path=OUT/(ident+"."+extension)
    if not path.is_file():
        raise HTTPException(404,"Image not found")
    return FileResponse(path,media_type={"png":"image/png","jpeg":"image/jpeg","webp":"image/webp"}[extension])

async def submit(body,request,images=(),image_keys=()):
    global inflight,requests_active
    if body.model!=args.served_model_name:
        raise HTTPException(404,{"message":"Unknown model","param":"model","code":"model_not_found"})
    if not engine.ready:
        raise HTTPException(503,engine.error or "Model is warming up")
    if requests_active>=args.max_concurrent_requests or inflight+body.n>args.max_concurrent_requests:
        raise HTTPException(429,"Request queue is full",headers={"Retry-After":"1"})
    width,height=dimensions(body,images)
    loop=asyncio.get_running_loop()
    submitted=time.perf_counter()
    request_id="img-"+uuid.uuid4().hex
    seed=body.seed if body.seed is not None else secrets.randbits(63)
    jobs=[Job(
        body.prompt,width,height,body.num_inference_steps,(seed+i)%(2**63),body.cache_prompt,
        submitted,future=loop.create_future(),images=images,image_keys=image_keys,
        cache_reference=body.cache_reference,diffusion_cache=body.diffusion_cache,
        cache_interval=body.cache_interval,cache_warmup_steps=body.cache_warmup_steps,
        cache_blocks=body.cache_blocks,response_format=body.response_format,
        output_format=body.output_format,output_compression=body.output_compression,
        request_id=request_id) for i in range(body.n)]
    requests_active+=1
    pending.extend(jobs)
    inflight+=len(jobs)
    changed.set()
    try:
        # Poll disconnection while waiting, to cancel queued work from abandoned clients.
        aggregate=asyncio.gather(*(job.future for job in jobs))
        deadline=loop.time()+args.request_timeout
        while not aggregate.done():
            remaining=deadline-loop.time()
            if remaining<=0:
                aggregate.cancel()
                raise HTTPException(504,"Inference timed out")
            await asyncio.wait((aggregate,),timeout=min(1.,remaining))
            if await request.is_disconnected():
                aggregate.cancel()
                raise HTTPException(499,"Client disconnected")
        results=await aggregate
        data=[]
        for ident,b64,metrics in results:
            item=({"b64_json":b64} if body.response_format=="b64_json" else
                  {"url":str(request.base_url).rstrip("/")+f"/v1/files/{ident}.{body.output_format}"})
            item["timings"]=metrics
            data.append(item)
        return {"created":int(time.time()),"data":data,"model":args.served_model_name,
                "request_id":request_id,"size":f"{width}x{height}"}
    except torch.OutOfMemoryError as exc:
        raise HTTPException(503,str(exc))
    except HTTPException:
        raise
    except Exception:
        log.exception("Inference request failed")
        raise HTTPException(500,"Image inference failed; inspect service logs")
    finally:
        for job in jobs:
            if not job.future.done():
                job.future.cancel()
        requests_active-=1

@app.post("/v1/images/generations")
async def generate(body:Generation,request:Request,authorization:str|None=Header(default=None)):
    authorize(authorization)
    return await submit(body,request)

EDIT_BODY={"requestBody":{"required":True,"content":{
    "multipart/form-data":{"schema":{"type":"object","required":["prompt","image"],"properties":{
        "image":{"type":"array","items":{"type":"string","format":"binary"}},
        "prompt":{"type":"string"},"model":{"type":"string","default":"flux2klein"},
        "size":{"type":"string","default":"1024x1024"},
        "diffusion_cache":{"type":"string","enum":["none","exact","taylorseer"],"default":"exact"},
        "seed":{"type":"integer"},"n":{"type":"integer","default":1},
        "num_inference_steps":{"type":"integer","default":4}}}},
    "application/json":{"schema":{"type":"object","required":["prompt","images"],
        "properties":{"prompt":{"type":"string"},"images":{"type":"array",
        "items":{"oneOf":[{"type":"string"},{"type":"object"}]}}}}}
}}}

@app.post("/v1/images/edits",openapi_extra=EDIT_BODY)
async def edit(request:Request,authorization:str|None=Header(default=None)):
    authorize(authorization)
    async with upload_semaphore:
        content_type=request.headers.get("content-type","")
        blobs=[]
        data={}
        if content_type.startswith("multipart/form-data"):
            try:
                form=await request.form(max_files=args.max_reference_images+1,max_fields=32)
            except Exception as exc:
                raise HTTPException(400,{"message":"Invalid multipart image upload","param":"image"})
            try:
                uploads=[]
                for name,value in form.multi_items():
                    if isinstance(value,UploadFile):
                        if name not in ("image","image[]","images","images[]"):
                            raise HTTPException(400,{"message":f"Unsupported file field: {name}","param":name})
                        uploads.append(value)
                    else:
                        if name in data:
                            raise HTTPException(400,{"message":f"Repeated parameter: {name}","param":name})
                        data[name]=value
                if not 1<=len(uploads)<=args.max_reference_images:
                    raise HTTPException(400,{"message":f"Supply 1..{args.max_reference_images} reference images","param":"image"})
                for upload in uploads:
                    blobs.append(await upload.read(MAX_IMAGE_BYTES+1))
            finally:
                await form.close()
        elif content_type.startswith("application/json"):
            try:
                data=await request.json()
            except (ValueError,json.JSONDecodeError):
                raise HTTPException(400,"Invalid JSON body")
            if not isinstance(data,dict):
                raise HTTPException(400,"JSON body must be an object")
            if "images" in data and "image" in data:
                raise HTTPException(400,{"message":"Supply either image or images","param":"image"})
            raw=data.pop("images") if "images" in data else data.pop("image",None)
            raw=raw if isinstance(raw,list) else [raw] if raw is not None else []
            if not 1<=len(raw)<=args.max_reference_images:
                raise HTTPException(400,{"message":f"Supply 1..{args.max_reference_images} reference images","param":"image"})
            blobs=[parse_data_image(value) for value in raw]
        else:
            raise HTTPException(415,"Use multipart/form-data or application/json")
        body=validate_body(data)
        images,image_keys=await asyncio.get_running_loop().run_in_executor(encode_pool,decode_images,blobs)
    return await submit(body,request,images,image_keys)

if __name__=="__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app,host=args.host,port=args.port,workers=1,limit_concurrency=1024,
                backlog=2048,timeout_keep_alive=30)

