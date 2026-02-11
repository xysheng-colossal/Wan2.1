# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
from datetime import datetime
import logging
import os
import sys
import subprocess
import warnings
import random
import time
warnings.filterwarnings('ignore')

import torch
import torch_npu
torch_npu.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format=False
from torch_npu.contrib import transfer_to_npu
import torch.distributed as dist
from PIL import Image

import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import cache_video, cache_image, str2bool
from wan.distributed.parallel_mgr import ParallelConfig, init_parallel_env, finalize_parallel_env
from wan.distributed.tp_applicator import TensorParallelApplicator

from mindiesd import CacheConfig, CacheAgent


EXAMPLE_PROMPT = {
    "t2v-1.3B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2v-14B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2i-14B": {
        "prompt": "一个朴素端庄的美人",
    },
    "i2v-14B": {
        "prompt":
            "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "image":
            "examples/i2v_input.JPG",
    },
}


def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"

    # The default sampling steps are 40 for image-to-video tasks and 50 for text-to-video tasks.
    if args.sample_steps is None:
        args.sample_steps = 40 if "i2v" in args.task else 50
    else:
        assert args.sample_steps >= 1 , f"sample_steps should be >= 1, but get {args.sample_steps}"

    if args.sample_shift is None:
        args.sample_shift = 5.0
        if "i2v" in args.task and args.size in ["832*480", "480*832"]:
            args.sample_shift = 3.0
    else:
        assert args.sample_shift > 0.0 , f"sample_shift should be > 0, but get {args.sample_shift}"

    # The default number of frames are 1 for text-to-image tasks and 81 for other tasks.
    if args.frame_num is None:
        args.frame_num = 1 if "t2i" in args.task else 81
    else:
        assert args.frame_num > 1 and(args.frame_num - 1) % 4 == 0, f"frame_num should be 4n+1 (n>0), but get {args.frame_num}"

    # T2I frame_num check
    if "t2i" in args.task:
        assert args.frame_num == 1, f"Unsupport frame_num {args.frame_num} for task {args.task}"

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)
    
    if args.cfg_size < 1 or args.ulysses_size < 1 or args.ring_size < 1 or args.tp_size < 1:
        raise ValueError(f"cfg_size, ulysses_size, ring_size and tp_size must >= 1, \
                         but get cfg_size={args.cfg_size}, ulysses_size={args.ulysses_size}, ring_size={args.ring_size}, tp_size={args.tp_size}")

    if args.tp_size > 1:
        assert args.ulysses_size == 1 and args.ring_size == 1, \
            f"tp only supported when ulysses_size == 1, and ring_size == 1, but get ulysses_size {args.ulysses_size}, ring_size {args.ring_size}  "
    
    assert args.cfg_size in (1, 2), f"cfg_size only support 1 or 2, but get {args.cfg_size}"

    # Size check
    assert args.size in SUPPORTED_SIZES[args.task], \
        f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="t2v-14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="How many frames to sample from a image or video. The number should be 4n+1"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )
    parser.add_argument(
        "--profile_stage",
        action="store_true",
        default=False,
        help="Print detailed server-side stage timing instead of using offline profile traces.",
    )
    parser.add_argument(
        "--profile_stage_file",
        type=str,
        default=None,
        help="Append concise stage profile report to this file (RUN/TOTAL/STEP/SLOW format).",
    )
    parser.add_argument(
        "--profile_attn",
        action="store_true",
        default=False,
        help="Enable internal attention breakdown profiling.",
    )
    parser.add_argument(
        "--profile_attn_file",
        type=str,
        default=None,
        help="Append concise attention profile report to this file (ATTN_RUN/ATTN format).",
    )
    parser.add_argument(
        "--profile_block",
        action="store_true",
        default=False,
        help="Enable DiT block-level breakdown profiling.",
    )
    parser.add_argument(
        "--profile_block_file",
        type=str,
        default=None,
        help="Append concise block profile report to this file (BLOCK_RUN/BLOCK format).",
    )
    parser.add_argument(
        "--analyze_profile",
        action="store_true",
        default=False,
        help="After generation, analyze profile_stage_file and print bottlenecks/suggestions.",
    )
    parser.add_argument(
        "--perf_logic",
        type=str,
        default="optimized",
        choices=["legacy", "optimized"],
        help="Select inference logic for A/B perf comparison.",
    )
    parser.add_argument(
        "--serialize_comm",
        action="store_true",
        default=False,
        help="Enable lightweight serialization on FSDP all-gather scheduling only.",
    )
    parser.add_argument(
        "--cfg_fused_forward",
        action="store_true",
        default=False,
        help="For cfg_size=1, fuse cond/uncond into one DiT forward (batch=2) for perf A/B.",
    )
    parser.add_argument(
        "--cache_cross_kv",
        action="store_true",
        default=False,
        help="Cache cross-attention K/V projections across denoise steps.",
    )
    parser.add_argument(
        "--cfg_size",
        type=int,
        default=1,
        help="The size of the cfg parallelism in DiT.")
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--ring_size",
        type=int,
        default=1,
        help="The size of the ring attention parallelism in DiT.")
    parser.add_argument(
        "--tp_size",
        type=int,
        default=1,
        help="The size of the tensor parallelism in DiT.")
    parser.add_argument(
        "--vae_parallel",
        action="store_true",
        default=False,
        help="Whether to use parallel for vae.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated image or video to.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the image or video from.")
    parser.add_argument(
        "--use_prompt_extend",
        action="store_true",
        default=False,
        help="Whether to use prompt extend.")
    parser.add_argument(
        "--prompt_extend_method",
        type=str,
        default="local_qwen",
        choices=["dashscope", "local_qwen"],
        help="The prompt extend method to use.")
    parser.add_argument(
        "--prompt_extend_model",
        type=str,
        default=None,
        help="The prompt extend model to use.")
    parser.add_argument(
        "--prompt_extend_target_lang",
        type=str,
        default="zh",
        choices=["zh", "en"],
        help="The target language of prompt extend.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=-1,
        help="The seed to use for generating the image or video.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="The image to generate the video from.")
    parser.add_argument(
        "--sample_solver",
        type=str,
        default='unipc',
        choices=['unipc', 'dpm++'],
        help="The solver used to sample.")
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=5.0,
        help="Classifier free guidance scale.")
    parser.add_argument(
        "--quant_dit_path",
        type=str,
        help="Path to quantization description file (enables quantization if provided, format: quant_model_description_*.json)"
    )
    
    parser = add_attentioncache_args(parser)
    parser = add_rainfusion_args(parser)
    args = parser.parse_args()
    _validate_args(args)

    return args


def add_attentioncache_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="Attention Cache args")

    group.add_argument("--use_attentioncache", action='store_true')
    group.add_argument("--attentioncache_ratio", type=float, default=1.2)
    group.add_argument("--attentioncache_interval", type=int, default=4)
    group.add_argument("--start_step", type=int, default=12)
    group.add_argument("--end_step", type=int, default=37)

    return parser


def add_rainfusion_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="Rainfusion args")

    group.add_argument("--use_rainfusion", action='store_true', help="Whether to use sparse fa")
    group.add_argument("--sparsity", type=float, default=0.64, help="Sparsity of flash attention, greater means more speed")
    group.add_argument("--sparse_start_step", type=int, default=15)

    return parser


def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def _set_transformer_rainfusion_config(transformer, dit_fsdp, rainfusion_config):
    if dit_fsdp:
        transformer._fsdp_wrapped_module.rainfusion_config = rainfusion_config
    else:
        transformer.rainfusion_config = rainfusion_config


def _attach_cache_to_transformer(transformer, dit_fsdp, cache, args):
    if dit_fsdp:
        for block in transformer._fsdp_wrapped_module.blocks:
            block._fsdp_wrapped_module.cache = cache
            block._fsdp_wrapped_module.args = args
    else:
        for block in transformer.blocks:
            block.cache = cache
            block.args = args


def _get_cache_blocks_count(args, transformer):
    return len(transformer.blocks) * 2 // args.cfg_size


def _run_profile_analysis(profile_file):
    if not profile_file:
        logging.warning("Skip profile analysis because profile_stage_file is empty.")
        return

    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools", "analyze_perf.py")
    if not os.path.exists(script_path):
        logging.warning(f"Skip profile analysis because script not found: {script_path}")
        return

    if not os.path.exists(profile_file):
        logging.warning(f"Skip profile analysis because profile file not found: {profile_file}")
        return

    cmd = [sys.executable, script_path, profile_file]
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except Exception as exc:
        logging.warning(f"Profile analysis failed: {exc}")
        return

    if result.returncode != 0:
        logging.warning(
            f"Profile analysis exited with code {result.returncode}: {result.stderr.strip()}"
        )
        return

    output = result.stdout.strip()
    if output:
        logging.info("Profile analysis report:\n" + output)


def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)
    stream = torch.npu.Stream()

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="hccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            args.cfg_size > 1 or args.ulysses_size > 1 or args.ring_size > 1
        ), f"context parallel are not supported in non-distributed environments."
        assert not (
            args.vae_parallel
        ), f"vae parallel are not supported in non-distributed environments."

    if args.cfg_size > 1 or args.ulysses_size > 1 or args.ring_size > 1 or args.tp_size > 1:
        assert args.cfg_size * args.ulysses_size * args.ring_size * args.tp_size == world_size, f"The number of cfg_size, ulysses_size and ring_size should be equal to the world size."
        sp_degree = args.ulysses_size * args.ring_size
        parallel_config = ParallelConfig(
            sp_degree=sp_degree,
            ulysses_degree=args.ulysses_size,
            ring_degree=args.ring_size,
            tp_degree=args.tp_size,
            use_cfg_parallel=(args.cfg_size==2),
            world_size=world_size,
        )
        init_parallel_env(parallel_config)

    if args.tp_size > 1 and args.dit_fsdp:
        logging.info("DiT using Tensor Parallel, disabled dit_fsdp")
        args.dit_fsdp = False

    if args.use_prompt_extend:
        if args.prompt_extend_method == "dashscope":
            prompt_expander = DashScopePromptExpander(
                model_name=args.prompt_extend_model, is_vl="i2v" in args.task)
        elif args.prompt_extend_method == "local_qwen":
            prompt_expander = QwenPromptExpander(
                model_name=args.prompt_extend_model,
                is_vl="i2v" in args.task,
                device=rank)
        else:
            raise NotImplementedError(
                f"Unsupport prompt_extend_method: {args.prompt_extend_method}")

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, f"`num_heads` must be divisible by `ulysses_size`."

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")
    use_legacy_perf = args.perf_logic == "legacy"

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    rainfusion_config = {
        "sparsity": args.sparsity,
        "skip_timesteps": args.sparse_start_step,
        "grid_size": None,
        "atten_mask_all": None
    }
    if args.use_rainfusion and args.cfg_fused_forward:
        logging.warning(
            "`--use_rainfusion` is currently incompatible with `--cfg_fused_forward`; "
            "falling back to non-fused CFG forward."
        )
        args.cfg_fused_forward = False

    if "t2v" in args.task or "t2i" in args.task:
        if args.prompt is None:
            args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
        logging.info(f"Input prompt: {args.prompt}")
        if args.use_prompt_extend:
            logging.info("Extending prompt ...")
            if rank == 0:
                prompt_output = prompt_expander(
                    args.prompt,
                    tar_lang=args.prompt_extend_target_lang,
                    seed=args.base_seed)
                if prompt_output.status == False:
                    logging.info(
                        f"Extending prompt failed: {prompt_output.message}")
                    logging.info("Falling back to original prompt.")
                    input_prompt = args.prompt
                else:
                    input_prompt = prompt_output.prompt
                input_prompt = [input_prompt]
            else:
                input_prompt = [None]
            if dist.is_initialized():
                dist.broadcast_object_list(input_prompt, src=0)
            args.prompt = input_prompt[0]
            logging.info(f"Extended prompt: {args.prompt}")

        logging.info("Creating WanT2V pipeline.")
        wan_t2v = wan.WanT2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
            t5_cpu=args.t5_cpu,
            use_vae_parallel=args.vae_parallel,
            quant_dit_path=args.quant_dit_path,
            use_legacy_perf=use_legacy_perf,
            serialize_comm=args.serialize_comm,
        )

        transformer = wan_t2v.model

        if args.use_rainfusion:
            _set_transformer_rainfusion_config(
                transformer, args.dit_fsdp, rainfusion_config
            )

        if args.tp_size > 1:
            logging.info("Initializing tensor parallel...")
            applicator = TensorParallelApplicator(args.tp_size, device_map="cpu")
            applicator.apply_to_model(transformer)
        wan_t2v.model.to("npu")

        config = CacheConfig(
                method="attention_cache",
                blocks_count=_get_cache_blocks_count(args, transformer),
                steps_count=args.sample_steps
            )
        cache = CacheAgent(config)
        _attach_cache_to_transformer(transformer, args.dit_fsdp, cache, args)

        t2v_generate_kwargs = dict(
            input_prompt=args.prompt,
            size=SIZE_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model,
            legacy_model_to_each_step=use_legacy_perf,
            cfg_fused_forward=args.cfg_fused_forward,
            cache_cross_kv=args.cache_cross_kv,
        )
        attn_profile_file = args.profile_attn_file or args.profile_stage_file
        block_profile_file = args.profile_block_file or args.profile_stage_file

        logging.info(f"Warm up 2 steps...")
        video = wan_t2v.generate(
            t2v_generate_kwargs["input_prompt"],
            size=t2v_generate_kwargs["size"],
            frame_num=t2v_generate_kwargs["frame_num"],
            shift=t2v_generate_kwargs["shift"],
            sample_solver=t2v_generate_kwargs["sample_solver"],
            sampling_steps=2,
            guide_scale=t2v_generate_kwargs["guide_scale"],
            seed=t2v_generate_kwargs["seed"],
            offload_model=t2v_generate_kwargs["offload_model"],
            profile_stage=False,
            profile_stage_file=None,
            profile_attn=False,
            profile_attn_file=None,
            profile_block=False,
            profile_block_file=None,
            legacy_model_to_each_step=t2v_generate_kwargs["legacy_model_to_each_step"],
            cfg_fused_forward=t2v_generate_kwargs["cfg_fused_forward"],
            cache_cross_kv=t2v_generate_kwargs["cache_cross_kv"],
        )

        if args.use_attentioncache:
            config = CacheConfig(
                method="attention_cache",
                blocks_count=_get_cache_blocks_count(args, transformer),
                steps_count=args.sample_steps,
                step_start=args.start_step,
                step_interval=args.attentioncache_interval,
                step_end=args.end_step
            )
            cache = CacheAgent(config)
            _attach_cache_to_transformer(transformer, args.dit_fsdp, cache, args)

        logging.info(f"Generating video ...")
        stream.synchronize()
        begin = time.time()
        video = wan_t2v.generate(
            t2v_generate_kwargs["input_prompt"],
            size=t2v_generate_kwargs["size"],
            frame_num=t2v_generate_kwargs["frame_num"],
            shift=t2v_generate_kwargs["shift"],
            sample_solver=t2v_generate_kwargs["sample_solver"],
            sampling_steps=args.sample_steps,
            guide_scale=t2v_generate_kwargs["guide_scale"],
            seed=t2v_generate_kwargs["seed"],
            offload_model=t2v_generate_kwargs["offload_model"],
            profile_stage=args.profile_stage,
            profile_stage_file=args.profile_stage_file,
            profile_attn=args.profile_attn,
            profile_attn_file=attn_profile_file,
            profile_block=args.profile_block,
            profile_block_file=block_profile_file,
            legacy_model_to_each_step=t2v_generate_kwargs["legacy_model_to_each_step"],
            cfg_fused_forward=t2v_generate_kwargs["cfg_fused_forward"],
            cache_cross_kv=t2v_generate_kwargs["cache_cross_kv"],
        )
        stream.synchronize()
        end = time.time()
        logging.info(f"Generating video used time {end - begin: .4f}s")

    else:
        if args.prompt is None:
            args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
        if args.image is None:
            args.image = EXAMPLE_PROMPT[args.task]["image"]
        logging.info(f"Input prompt: {args.prompt}")
        logging.info(f"Input image: {args.image}")

        img = Image.open(args.image).convert("RGB")
        if args.use_prompt_extend:
            logging.info("Extending prompt ...")
            if rank == 0:
                prompt_output = prompt_expander(
                    args.prompt,
                    tar_lang=args.prompt_extend_target_lang,
                    image=img,
                    seed=args.base_seed)
                if prompt_output.status == False:
                    logging.info(
                        f"Extending prompt failed: {prompt_output.message}")
                    logging.info("Falling back to original prompt.")
                    input_prompt = args.prompt
                else:
                    input_prompt = prompt_output.prompt
                input_prompt = [input_prompt]
            else:
                input_prompt = [None]
            if dist.is_initialized():
                dist.broadcast_object_list(input_prompt, src=0)
            args.prompt = input_prompt[0]
            logging.info(f"Extended prompt: {args.prompt}")

        logging.info("Creating WanI2V pipeline.")
        wan_i2v = wan.WanI2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
            t5_cpu=args.t5_cpu,
            use_vae_parallel=args.vae_parallel,
            quant_dit_path=args.quant_dit_path,
            use_legacy_perf=use_legacy_perf,
            serialize_comm=args.serialize_comm,
        )

        transformer = wan_i2v.model

        if args.use_rainfusion:
            _set_transformer_rainfusion_config(
                transformer, args.dit_fsdp, rainfusion_config
            )

        if args.tp_size > 1:
            logging.info("Initializing tensor parallel...")
            applicator = TensorParallelApplicator(args.tp_size, device_map="cpu")
            applicator.apply_to_model(transformer)
        wan_i2v.model.to("npu")

        config = CacheConfig(
                method="attention_cache",
                blocks_count=_get_cache_blocks_count(args, transformer),
                steps_count=args.sample_steps
            )
        cache = CacheAgent(config)
        _attach_cache_to_transformer(transformer, args.dit_fsdp, cache, args)

        i2v_generate_kwargs = dict(
            input_prompt=args.prompt,
            image=img,
            max_area=MAX_AREA_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model,
        )
        attn_profile_file = args.profile_attn_file or args.profile_stage_file
        block_profile_file = args.profile_block_file or args.profile_stage_file

        logging.info(f"Warm up 2 steps...")
        video = wan_i2v.generate(
            i2v_generate_kwargs["input_prompt"],
            i2v_generate_kwargs["image"],
            max_area=i2v_generate_kwargs["max_area"],
            frame_num=i2v_generate_kwargs["frame_num"],
            shift=i2v_generate_kwargs["shift"],
            sample_solver=i2v_generate_kwargs["sample_solver"],
            sampling_steps=2,
            guide_scale=i2v_generate_kwargs["guide_scale"],
            seed=i2v_generate_kwargs["seed"],
            offload_model=i2v_generate_kwargs["offload_model"],
            profile_stage=False,
            profile_stage_file=None,
            profile_attn=False,
            profile_attn_file=None,
            profile_block=False,
            profile_block_file=None,
        )

        if args.use_attentioncache:
            config = CacheConfig(
                method="attention_cache",
                blocks_count=_get_cache_blocks_count(args, transformer),
                steps_count=args.sample_steps,
                step_start=args.start_step,
                step_interval=args.attentioncache_interval,
                step_end=args.end_step
            )
            cache = CacheAgent(config)
            _attach_cache_to_transformer(transformer, args.dit_fsdp, cache, args)

        logging.info("Generating video ...")
        stream.synchronize()
        begin = time.time()
        video = wan_i2v.generate(
            i2v_generate_kwargs["input_prompt"],
            i2v_generate_kwargs["image"],
            max_area=i2v_generate_kwargs["max_area"],
            frame_num=i2v_generate_kwargs["frame_num"],
            shift=i2v_generate_kwargs["shift"],
            sample_solver=i2v_generate_kwargs["sample_solver"],
            sampling_steps=args.sample_steps,
            guide_scale=i2v_generate_kwargs["guide_scale"],
            seed=i2v_generate_kwargs["seed"],
            offload_model=i2v_generate_kwargs["offload_model"],
            profile_stage=args.profile_stage,
            profile_stage_file=args.profile_stage_file,
            profile_attn=args.profile_attn,
            profile_attn_file=attn_profile_file,
            profile_block=args.profile_block,
            profile_block_file=block_profile_file,
        )

        stream.synchronize()
        end = time.time()
        logging.info(f"Generating video used time {end - begin: .4f}s")


    if rank == 0:
        if args.analyze_profile and args.profile_stage:
            _run_profile_analysis(args.profile_stage_file)

        if args.save_file is None:
            formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            formatted_prompt = args.prompt.replace(" ", "_").replace("/",
                                                                     "_")[:50]
            suffix = '.png' if "t2i" in args.task else '.mp4'
            args.save_file = f"{args.task}_{args.size.replace('*','x') if sys.platform=='win32' else args.size}_{args.cfg_size}_{args.ulysses_size}_{args.ring_size}_{args.tp_size}_{formatted_prompt}_{formatted_time}" + suffix

        if "t2i" in args.task:
            logging.info(f"Saving generated image to {args.save_file}")
            cache_image(
                tensor=video.squeeze(1)[None],
                save_file=args.save_file,
                nrow=1,
                normalize=True,
                value_range=(-1, 1))
        else:
            logging.info(f"Saving generated video to {args.save_file}")
            cache_video(
                tensor=video[None],
                save_file=args.save_file,
                fps= args.frame_num // 5,
                nrow=1,
                normalize=True,
                value_range=(-1, 1))
    logging.info("Finished.")


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
    finalize_parallel_env()
    
