"""OpenAI SDK client for text generation, single-image and multi-reference edits."""
import argparse
import base64
from contextlib import ExitStack
import os
from pathlib import Path
from openai import OpenAI

parser=argparse.ArgumentParser()
parser.add_argument("--base-url",default="http://127.0.0.1:8036/v1")
parser.add_argument("--prompt",default="A red ceramic teapot on a wooden table in soft daylight")
parser.add_argument("--image",action="append",default=[],help="Reference file; repeat for multi-reference editing")
parser.add_argument("--edit",action="store_true",help=argparse.SUPPRESS)
parser.add_argument("--size",default="1024x1024")
parser.add_argument("--output",default=str(Path(__file__).resolve().parent/"outputs/client.png"))
parser.add_argument("--seed",type=int,default=42)
parser.add_argument("--steps",type=int,default=4)
parser.add_argument("--n",type=int,default=1)
parser.add_argument("--diffusion-cache",choices=("none","exact","taylorseer"),default="exact")
parser.add_argument("--cache-interval",type=int,default=2)
parser.add_argument("--cache-blocks",type=int,default=12)
parser.add_argument("--no-prompt-cache",action="store_true")
parser.add_argument("--no-reference-cache",action="store_true")
parser.add_argument("--output-format",choices=("png","jpeg","webp"),default="png")
args=parser.parse_args()
if args.edit and not args.image:
    parser.error("Editing requires at least one --image")
client=OpenAI(base_url=args.base_url,api_key=os.environ["FLUX2KLEIN_API_KEY"],timeout=600,max_retries=0)
options={"seed":args.seed,"num_inference_steps":args.steps,"diffusion_cache":args.diffusion_cache,
         "cache_interval":args.cache_interval,"cache_blocks":args.cache_blocks,
         "cache_prompt":not args.no_prompt_cache,"cache_reference":not args.no_reference_cache,
         "output_format":args.output_format}
parameters={"model":"flux2klein","prompt":args.prompt,"size":args.size,"n":args.n,
            "response_format":"b64_json","extra_body":options}
with ExitStack() as stack:
    if args.image:
        parameters["image"]=[stack.enter_context(open(Path(name).expanduser(),"rb")) for name in args.image]
        result=client.images.edit(**parameters)
    else:
        result=client.images.generate(**parameters)
for i,item in enumerate(result.data):
    path=Path(args.output).expanduser()
    if len(result.data)>1:
        path=path.with_name(path.stem+f"-{i+1}"+path.suffix)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(base64.b64decode(item.b64_json))
    print("Saved:",str(path))
    print("Timings:",item.model_extra.get("timings"))


