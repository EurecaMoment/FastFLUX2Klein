"""GPU engine: bounded batches, exact host caches, and opt-in diffusion prediction."""
import gc
import hashlib
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import torch
from diffusers import Flux2KleinPipeline
from optimizations import TensorLRU, DiffusionController, TransformerGraph, install_feature_encoder, park_unused_text_weights, fuse_qkv_without_duplicate_weights

log = logging.getLogger("flux2klein.engine")
VERSION = "2.0.0"


@dataclass
class Job:
    prompt: str
    width: int
    height: int
    steps: int
    seed: int
    cache_prompt: bool
    submitted: float
    future: object = None
    images: tuple = ()
    image_keys: tuple = ()
    cache_reference: bool = True
    diffusion_cache: str = "exact"
    cache_interval: int = 2
    cache_warmup_steps: int = 2
    cache_blocks: int = 12
    response_format: str = "b64_json"
    output_format: str = "png"
    output_compression: int = 90
    request_id: str = ""

    @property
    def signature(self):
        return (self.width, self.height, self.steps, self.image_keys, self.cache_reference, self.diffusion_cache,
                self.cache_interval, self.cache_warmup_steps, self.cache_blocks)


class Engine:
    def __init__(self, config):
        self.config = config
        self.pipe = None
        self.ready = False
        self.error = None
        self.phase = "starting"
        self.prompt_cache = TensorLRU(config.prompt_cache_gb * 1024**3)
        self.reference_cache = TensorLRU(config.reference_cache_gb * 1024**3)
        self.reduced_limits = {}
        self.stats = {"batches": 0, "images": 0, "oom_splits": 0, "failed_batches": 0,
                      "last_batch_size": 0, "prompt_cache_hits": 0, "prompt_cache_misses": 0,
                      "reference_cache_hits": 0, "predicted_blocks": 0}
        self.optimizations = {"early_text_features": False, "qkv_fusion": False,
                              "batched_text_encoder": True, "cuda_graph": False,
                              "compile": config.compile, "compile_vae": config.compile_vae,
                              "precision": "bfloat16", "text_sequence_length": 512}
        self.graph_runner = None
        self.startup_checks = {}

    def batch_limit(self, job):
        output_pixels = job.width * job.height
        reference_pixels = sum(im.width * im.height for im in job.images)
        pixels = output_pixels + reference_pixels
        base = self.config.max_batch_size if output_pixels <= 512*512 else self.config.max_batch_size_1024
        # Control the joint attention context, not just the output resolution.
        limit = max(1, min(base, (512*512*self.config.max_batch_size)//max(pixels,1)))
        if job.diffusion_cache == "taylorseer":
            limit = min(limit, 2 if pixels <= 512*512 else 1)
        return min(limit, self.reduced_limits.get(job.signature, limit))

    @torch.inference_mode()
    def initialize(self):
        self.phase = "loading_model"
        torch.set_num_threads(self.config.cpu_threads)
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(
            (self.config.gpu_memory_gb-1)*1e9/torch.cuda.get_device_properties(0).total_memory)
        self.pipe = Flux2KleinPipeline.from_pretrained(
            self.config.model, torch_dtype=torch.bfloat16, local_files_only=True).to("cuda")
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.vae.to(memory_format=torch.channels_last)
        # Required inference weights stay resident; unused weights may be parked once.
        sample = "A red ceramic teapot on a wooden table"
        self.phase = "checking_text_feature_equivalence"
        original = self.pipe.encode_prompt(sample, device=torch.device("cuda"))[0]
        original_encoder = self.pipe._get_qwen3_prompt_embeds
        install_feature_encoder(self.pipe)
        fast = self.pipe.encode_prompt(sample, device=torch.device("cuda"))[0]
        identical = torch.equal(original, fast)
        self.startup_checks["early_text_features_bitwise_equal"] = identical
        self.startup_checks["early_text_features_max_abs"] = float((original-fast).abs().max())
        if identical:
            self.optimizations["early_text_features"] = True
            if self.config.park_unused_text_weights:
                released = park_unused_text_weights(self.pipe)
                self.optimizations["unused_text_weights_cpu_gb"] = round(released/1e9,3)
                check = self.pipe.encode_prompt(sample,device=torch.device("cuda"))[0]
                if not torch.equal(original,check):
                    raise RuntimeError("Unused-weight parking changed the required text features")
                self.startup_checks["unused_weight_parking_bitwise_equal"] = True
                del check
        else:
            self.pipe._get_qwen3_prompt_embeds = original_encoder
            log.warning("Using original text encoder because feature equivalence failed")
        del original, fast
        if self.config.fuse_qkv:
            self.optimizations["qkv_fusion"] = bool(fuse_qkv_without_duplicate_weights(self.pipe.transformer))
        self.diffusion = DiffusionController(self.pipe.transformer)
        self.phase = "compiling"
        if self.config.compile == "regional":
            self.pipe.transformer.compile_repeated_blocks(
                fullgraph=True, mode="max-autotune-no-cudagraphs")
        if self.config.compile_vae:
            self.pipe.vae.decode = torch.compile(self.pipe.vae.decode, fullgraph=False, mode="default")
        for size, batches in [(512, (1,2,self.config.max_batch_size)),
                              (1024, (1,self.config.max_batch_size_1024))]:
            for batch in sorted(set(batches)):
                self.phase = f"warming_{size}_batch_{batch}"
                log.info("WARMUP size=%d batch=%d", size, batch)
                for _ in range(2):
                    result = self.infer([Job(sample,size,size,4,42+i,True,time.perf_counter()) for i in range(batch)])
                    effective = max(metrics["batch_size"] for _, metrics in result)
                    if effective < batch:
                        setting = "max_batch_size" if size == 512 else "max_batch_size_1024"
                        setattr(self.config, setting, min(getattr(self.config, setting), effective))
                        log.info("ADAPTED_BATCH_LIMIT size=%d limit=%d", size, effective)
        # Warm one reference-image shape without storing any output pictures.
        if self.config.warm_edits:
            from PIL import Image
            ref = Image.new("RGB",(512,512),(180,80,40))
            key = hashlib.sha256(ref.tobytes()).hexdigest()
            self.phase = "warming_edit"
            self.infer([Job("Make the object blue",512,512,4,42,True,time.perf_counter(),
                            images=(ref,),image_keys=(key,))])
        if self.config.cuda_graph == "auto":
            self.phase = "evaluating_cuda_graph"
            self.probe_cuda_graph(sample)
        self.ready = True
        self.phase = "ready"
        self.startup_checks["warmup_oom_splits"] = self.stats["oom_splits"]
        self.stats.update(batches=0,images=0,last_batch_size=0,oom_splits=0)
        log.info("MODEL_READY version=%s allocated_gb=%.3f reserved_gb=%.3f optimizations=%s",
                 VERSION,torch.cuda.memory_allocated()/1e9,torch.cuda.memory_reserved()/1e9,
                 self.optimizations)

    @torch.inference_mode()
    def probe_cuda_graph(self,sample):
        """Keep a capture only after equivalence, speed and full-batch memory checks."""
        trial = None
        previous_limits = dict(self.reduced_limits)
        report = {}
        try:
            torch.cuda.empty_cache()
            embeddings = self.prompt_cache.get(sample).to("cuda")
            latents,ids = self.pipe.prepare_latents(
                batch_size=1,num_latents_channels=32,height=512,width=512,
                dtype=torch.bfloat16,device=torch.device("cuda"),
                generator=torch.Generator("cuda").manual_seed(42))
            inputs = {"hidden_states":latents,"encoder_hidden_states":embeddings,
                      "timestep":torch.ones(1,device="cuda",dtype=torch.bfloat16),
                      "img_ids":ids,"txt_ids":self.pipe._prepare_text_ids(embeddings).to("cuda"),
                      "guidance":None,"joint_attention_kwargs":None,"return_dict":False}
            trial = TransformerGraph(self.pipe.transformer,self.diffusion,inputs)
            def timings(fn,count=5):
                values=[]
                for _ in range(count):
                    torch.cuda.synchronize()
                    started=time.perf_counter()
                    result=fn()
                    torch.cuda.synchronize()
                    values.append(time.perf_counter()-started)
                return result,sorted(values)[len(values)//2]
            expected,normal_seconds=timings(lambda:trial.original_forward(**inputs)[0])
            trial.capture()
            actual,graph_seconds=timings(lambda:trial.forward(**inputs)[0])
            equal=torch.equal(expected,actual)
            report.update(normal_seconds=normal_seconds,graph_seconds=graph_seconds,
                          bitwise_equal=equal,speedup=normal_seconds/graph_seconds)
            if not equal or graph_seconds>=normal_seconds*.97:
                raise RuntimeError("CUDA Graph did not meet equivalence and 3% speedup criteria")
            trial.install()
            self.graph_runner=trial
            for size, limit in [(512,self.config.max_batch_size),(1024,self.config.max_batch_size_1024)]:
                check=self.infer([Job(sample,size,size,4,500+i,True,time.perf_counter())
                                  for i in range(limit)])
                if any(metrics["batch_size"]!=limit for _,metrics in check):
                    raise RuntimeError("CUDA Graph reduced the supported full-batch memory headroom")
            self.optimizations["cuda_graph"]=True
            report["retained"]=True
            log.info("CUDA_GRAPH_ACCEPTED %s",report)
        except Exception as exc:
            if trial is not None:
                trial.uninstall()
            self.graph_runner=None
            self.reduced_limits=previous_limits
            report.update(retained=False,reason=f"{type(exc).__name__}: {exc}"[:500])
            log.info("CUDA_GRAPH_NOT_RETAINED %s",report)
        finally:
            self.startup_checks["cuda_graph"]=report
            trial = None
            gc.collect()
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def _encode_text_batch(self, prompts):
        try:
            embeddings,_ = self.pipe.encode_prompt(
                prompts,device=torch.device("cuda"),max_sequence_length=512)
            return [x.unsqueeze(0) for x in embeddings]
        except torch.OutOfMemoryError:
            pass
        gc.collect()
        torch.cuda.empty_cache()
        if len(prompts) == 1:
            raise torch.OutOfMemoryError("Text encoding exceeds the configured GPU budget")
        middle = len(prompts)//2
        return self._encode_text_batch(prompts[:middle])+self._encode_text_batch(prompts[middle:])

    @torch.inference_mode()
    def embeddings(self, jobs):
        hits = [False]*len(jobs)
        values = {}
        missing = []
        positions = {}
        for i,job in enumerate(jobs):
            if job.cache_prompt:
                cached = self.prompt_cache.get(job.prompt)
                if cached is not None:
                    values[("cached",job.prompt)] = cached
                    hits[i] = True
                    continue
            if job.prompt not in positions:
                positions[job.prompt] = []
                missing.append(job.prompt)
            positions[job.prompt].append(i)
        fresh = {}
        for start in range(0,len(missing),self.config.text_batch_size):
            prompts = missing[start:start+self.config.text_batch_size]
            encoded = self._encode_text_batch(prompts)
            for prompt,value in zip(prompts,encoded):
                fresh[prompt] = value
                if any(jobs[i].cache_prompt for i in positions[prompt]):
                    self.prompt_cache.put(prompt,value)
        # Assemble one host-to-device transfer for all cached embeddings.
        cpu_keys = list(values)
        if cpu_keys:
            host = torch.cat([values[key] for key in cpu_keys],dim=0).pin_memory()
            device = host.to("cuda",non_blocking=True)
            for i,key in enumerate(cpu_keys):
                values[key] = device[i:i+1]
        result = [values[("cached",job.prompt)] if hit else fresh[job.prompt]
                  for job,hit in zip(jobs,hits)]
        self.stats["prompt_cache_hits"] += sum(hits)
        self.stats["prompt_cache_misses"] += len(jobs)-sum(hits)
        return torch.cat(result,dim=0), hits, len(missing)

    @contextmanager
    def reference_context(self, job):
        counters = {"reference_cache_hits": 0,"reference_cache_misses": 0}
        if not job.images:
            yield counters
            return
        original = self.pipe._encode_vae_image
        index = 0
        def encode(image,generator):
            nonlocal index
            key = job.image_keys[index]
            index += 1
            cached = self.reference_cache.get(key) if job.cache_reference else None
            if cached is not None:
                counters["reference_cache_hits"] += 1
                return cached.to(device=image.device,dtype=self.pipe.vae.dtype,non_blocking=True)
            counters["reference_cache_misses"] += 1
            latent = original(image,generator)
            if job.cache_reference:
                self.reference_cache.put(key,latent)
            return latent
        self.pipe._encode_vae_image = encode
        try:
            yield counters
        finally:
            self.pipe._encode_vae_image = original
            self.stats["reference_cache_hits"] += counters["reference_cache_hits"]

    @torch.inference_mode()
    def infer(self, jobs):
        limit=self.batch_limit(jobs[0])
        if len(jobs)>limit:
            results=[]
            for start in range(0,len(jobs),limit):
                results.extend(self.infer(jobs[start:start+limit]))
            return results
        try:
            return self._infer(jobs)
        except torch.OutOfMemoryError:
            pass
        # Release the failed call's frames before splitting or retrying.
        gc.collect()
        torch.cuda.empty_cache()
        if len(jobs)==1:
            self.stats["failed_batches"] += 1
            raise torch.OutOfMemoryError("Image/reference context exceeds the 40GB GPU budget")
        self.stats["oom_splits"] += 1
        middle = len(jobs)//2
        self.reduced_limits[jobs[0].signature] = middle
        return self.infer(jobs[:middle])+self.infer(jobs[middle:])

    @torch.inference_mode()
    def _infer(self, jobs):
        first = jobs[0]
        if any(job.signature != first.signature for job in jobs):
            raise ValueError("Batch contains incompatible requests")
        if first.width*first.height <= 512*512:
            self.pipe.vae.disable_slicing()
        else:
            self.pipe.vae.enable_slicing()
        torch.cuda.synchronize()
        started = time.perf_counter()
        embeddings,hits,encoded_prompts = self.embeddings(jobs)
        torch.cuda.synchronize()
        text_seconds = time.perf_counter()-started
        generators = [torch.Generator(device="cuda").manual_seed(job.seed) for job in jobs]
        graph_hits_before = self.graph_runner.hits if self.graph_runner is not None else 0
        with self.reference_context(first) as references:
            with self.diffusion.context(
                first.diffusion_cache,first.steps,interval=first.cache_interval,
                warmup=first.cache_warmup_steps,blocks=first.cache_blocks) as cache:
                image_tensor = self.pipe(
                    prompt_embeds=embeddings,image=list(first.images) if first.images else None,
                    width=first.width,height=first.height,num_inference_steps=first.steps,
                    guidance_scale=1.0,generator=generators,output_type="pt",
                    callback_on_step_end=self.diffusion.on_step_end).images
                diffusion_metrics = {
                    "diffusion_cache":first.diffusion_cache,
                    "exact_cache_hits":cache["memo_hits"],
                    "predicted_blocks":cache["predicted_blocks"],
                    "approximate":bool(cache["predicted_blocks"]),
                }
        # Match Diffusers' float32 numpy rounding, but transfer RGB bytes instead
        # of float images. PIL construction and compression run in the I/O pool.
        pixels = image_tensor.float().mul(255).round().clamp_(0,255).to(torch.uint8).permute(0,2,3,1).contiguous()
        images = torch.empty_like(pixels,device="cpu",pin_memory=True)
        images.copy_(pixels,non_blocking=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-started
        self.stats["batches"] += 1
        self.stats["images"] += len(jobs)
        self.stats["last_batch_size"] = len(jobs)
        self.stats["predicted_blocks"] += diffusion_metrics["predicted_blocks"]
        metrics = {
            "batch_size":len(jobs),"batch_compute_seconds":round(elapsed,6),
            "amortized_compute_seconds_per_image":round(elapsed/len(jobs),6),
            "text_encode_seconds":round(text_seconds,6),"encoded_unique_prompts":encoded_prompts,
            "allocated_gb":round(torch.cuda.memory_allocated()/1e9,3),
            "reserved_gb":round(torch.cuda.memory_reserved()/1e9,3),
            "peak_allocated_gb":round(torch.cuda.max_memory_allocated()/1e9,3),
            "reference_images":len(first.images),**references,**diffusion_metrics}
        metrics["cuda_graph_replays"] = (self.graph_runner.hits-graph_hits_before
                                         if self.graph_runner is not None else 0)
        return [(image,{**metrics,"prompt_cache_hit":hit,"seed":job.seed,
                        "queue_seconds":round(max(0,started-job.submitted),6)})
                for image,job,hit in zip(images,jobs,hits)]


