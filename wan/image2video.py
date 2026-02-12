# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .modules.clip import CLIPModel
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .utils.stage_profile import StageProfiler
from .utils.attention_profile import (
    configure_attention_profiler,
    report_attention_profiler,
)
from .utils.block_profile import (
    configure_block_profiler,
    get_block_profiler,
    report_block_profiler,
)
from .vae_patch_parallel import VAE_patch_parallel, set_vae_patch_parallel
from wan.distributed.parallel_mgr import (
    get_sequence_parallel_world_size,
    get_classifier_free_guidance_world_size,
    get_classifier_free_guidance_rank,
    get_cfg_group,
)


class WanI2V:
    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
        init_on_cpu=True,
        use_vae_parallel=False,
        quant_dit_path=None,
        use_legacy_perf=False,
        serialize_comm=False,
    ):
        r"""
        Initializes the image-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.use_usp = use_usp
        self.t5_cpu = t5_cpu
        self.dit_fsdp = dit_fsdp

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(
            shard_model,
            device_id=device_id,
            use_legacy_behavior=use_legacy_perf,
            serialize_communication=serialize_comm,
        )
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu') if int(os.getenv('T5_LOAD_CPU', 0)) else self.device,
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None,
        )

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device,
            dtype=self.param_dtype)
        if use_vae_parallel:
            all_pp_group_ranks = []
            if dist.get_world_size() < 8 :
                all_pp_group_ranks.append(list(range(0, dist.get_world_size())))
                set_vae_patch_parallel(self.vae.model, dist.get_world_size(), 1, all_pp_group_ranks= all_pp_group_ranks, decoder_decode="decoder.forward")
                set_vae_patch_parallel(self.vae.model, dist.get_world_size(), 1, all_pp_group_ranks= all_pp_group_ranks, decoder_decode="encoder.forward")
            else:
                for i in range(0, dist.get_world_size() // 8):
                    all_pp_group_ranks.append(list(range(8 * i, 8 * (i + 1))))
                set_vae_patch_parallel(self.vae.model, 4, 2, all_pp_group_ranks= all_pp_group_ranks, decoder_decode="decoder.forward")
                set_vae_patch_parallel(self.vae.model, 4, 2, all_pp_group_ranks= all_pp_group_ranks, decoder_decode="encoder.forward")

        self.clip = CLIPModel(
            dtype=config.clip_dtype,
            device=self.device,
            checkpoint_path=os.path.join(checkpoint_dir,
                                         config.clip_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.clip_tokenizer))

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.model = WanModel.from_pretrained(checkpoint_dir, torch_dtype=self.param_dtype)
        if quant_dit_path:
            quant_dit_path = os.path.abspath(quant_dit_path)
            quant_dit_desc_path = os.path.join(quant_dit_path, "quant_model_description_w8a8_dynamic.json")
            if not os.path.exists(quant_dit_desc_path):
                raise FileNotFoundError(f"Quantization description file not found: {quant_dit_desc_path}")
            logging.info(f"Enabled quant, trying to load quantized DiT model from {quant_dit_path}...")
            from mindiesd import quantize
            quantize(
                model=self.model,
                quant_des_path=quant_dit_desc_path,
                use_nz=True
            )
            logging.info("Load quantized DiT model successfully")
        self.model.eval().requires_grad_(False)

        if t5_fsdp or dit_fsdp or use_usp:
            init_on_cpu = False

        if use_usp:
            from .distributed.xdit_context_parallel import (usp_attn_forward,
                                                            usp_dit_forward)
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        # else:
        #     if not init_on_cpu:
        #         self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt

    def generate(self,
                 input_prompt,
                 img,
                 max_area=720 * 1280,
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=40,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True,
                 profile_stage=False,
                 profile_stage_file=None,
                 profile_attn=False,
                 profile_attn_file=None,
                 profile_block=False,
                 profile_block_file=None):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            img (PIL.Image.Image):
                Input image tensor. Shape: [3, H, W]
            max_area (`int`, *optional*, defaults to 720*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM
            profile_stage (`bool`, *optional*, defaults to False):
                If True, prints detailed stage timing logs on server side
            profile_stage_file (`str`, *optional*, defaults to None):
                If set, append concise stage profiling report lines into this file
            profile_attn (`bool`, *optional*, defaults to False):
                If True, enable internal attention stage profiling
            profile_attn_file (`str`, *optional*, defaults to None):
                If set, append concise attention profiling lines into this file
            profile_block (`bool`, *optional*, defaults to False):
                If True, enable DiT block-level profiling
            profile_block_file (`str`, *optional*, defaults to None):
                If set, append concise block profiling lines into this file

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from max_area)
                - W: Frame width from max_area)
        """
        profiler = StageProfiler(
            enabled=profile_stage,
            device=self.device,
            rank=self.rank,
            name="I2V",
            output_file=profile_stage_file,
            run_info={
                "frame_num": frame_num,
                "sampling_steps": sampling_steps,
                "max_area": max_area,
                "offload_model": int(bool(offload_model)),
                "sample_solver": sample_solver,
            },
        )
        configure_attention_profiler(
            enabled=profile_attn,
            device=self.device,
            rank=self.rank,
            name="I2V",
            output_file=profile_attn_file,
        )
        configure_block_profiler(
            enabled=profile_block,
            device=self.device,
            rank=self.rank,
            name="I2V",
            output_file=profile_block_file,
        )
        block_profiler = get_block_profiler()
        total_start = profiler.start()

        t0 = profiler.start()
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)
        profiler.stop("image_preprocess", t0)

        F = frame_num
        h, w = img.shape[1:]
        aspect_ratio = h / w
        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
            self.patch_size[1] * self.patch_size[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
            self.patch_size[2] * self.patch_size[2])
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]
        profiler.update_run_info(width=w, height=h, latent_h=lat_h, latent_w=lat_w)

        max_seq_len = ((F - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        latent_frame_num = (F - 1) // self.vae_stride[0] + 1
        t0 = profiler.start()
        noise = torch.randn(
            16, 
            latent_frame_num, 
            lat_h, 
            lat_w, 
            dtype=torch.float32, 
            generator=seed_g, 
            device=self.device)
        profiler.stop("noise_init", t0)

        t0 = profiler.start()
        msk = torch.ones(1, F, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
        ],
                           dim=1)
        msk = msk.view(1, latent_frame_num, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]
        profiler.stop("mask_prepare", t0)

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        # preprocess
        if not self.t5_cpu:
            t0 = profiler.start()
            self.text_encoder.model.to(self.device)
            profiler.stop("t5_to_device", t0)

            t0 = profiler.start()
            context = self.text_encoder([input_prompt], self.device)
            profiler.stop("text_encode_pos", t0)

            t0 = profiler.start()
            context_null = self.text_encoder([n_prompt], self.device)
            profiler.stop("text_encode_neg", t0)
            if offload_model:
                t0 = profiler.start()
                self.text_encoder.model.cpu()
                profiler.stop("t5_offload_cpu", t0)
        else:
            t0 = profiler.start()
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            profiler.stop("text_encode_pos_cpu", t0)

            t0 = profiler.start()
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            profiler.stop("text_encode_neg_cpu", t0)

            t0 = profiler.start()
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]
            profiler.stop("text_context_to_device", t0)

        t0 = profiler.start()
        self.clip.model.to(self.device)
        clip_context = self.clip.visual([img[:, None, :, :]])
        if offload_model:
            self.clip.model.cpu()
        profiler.stop("clip_encode", t0)
        # NOTE 下面是源代码interpolate 在cpu 计算,下面移动到NPU上计算,有细微差异
        '''
        encode_input = torch.concat([
                    torch.nn.functional.interpolate(
                        img[None].cpu(), size=(h, w), mode='bicubic').transpose(
                            0, 1),
                    torch.zeros(3, F - 1, h, w)
                ],
            dim=1).to(self.device)
        '''
        # 这里在是NPU 上计算会快一些
        t0 = profiler.start()
        encode_input = torch.concat([
                    torch.nn.functional.interpolate(
                        img[None].to(self.device), size=(h, w), mode='bicubic').transpose(
                            0, 1),
                        torch.zeros(3, F - 1, h, w, device = self.device)],
                     dim=1)
        profiler.stop("vae_encode_input_prepare", t0)
        
        t0 = profiler.start()
        with VAE_patch_parallel():
            y = self.vae.encode([
                encode_input
            ])[0]
        profiler.stop("vae_encode", t0)

        t0 = profiler.start()
        y = torch.concat([msk, y])
        profiler.stop("cond_concat", t0)

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():
            t0 = profiler.start()

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")
            profiler.stop("scheduler_setup", t0)

            # sample videos
            latent = noise

            arg_c = {
                'context': [context[0]],
                'clip_fea': clip_context,
                'seq_len': max_seq_len,
                'y': [y],
                'block_profiler': block_profiler,
            }

            arg_null = {
                'context': context_null,
                'clip_fea': clip_context,
                'seq_len': max_seq_len,
                'y': [y],
                'block_profiler': block_profiler,
            }

            arg_all = {
                'context': [context[0]] if get_classifier_free_guidance_rank()==0 else context_null,
                'clip_fea': clip_context,
                'seq_len': max_seq_len,
                'y': [y],
                'block_profiler': block_profiler,
            }

            if offload_model:
                t0 = profiler.start()
                torch.cuda.empty_cache()
                profiler.stop("pre_denoise_empty_cache", t0)

            t0 = profiler.start()
            self.model.to(self.device)
            profiler.stop("dit_to_device", t0)

            denoise_loop_start = profiler.start()
            for t_idx, t in enumerate(tqdm(timesteps, disable=self.rank != 0)):
                step_start = profiler.start()
                latent_model_input = [latent.to(self.device)] if offload_model else [latent]
                timestep = t.unsqueeze(0)

                if get_classifier_free_guidance_world_size() == 2:
                    t0 = profiler.start()
                    noise_pred = self.model(
                        latent_model_input, t=timestep, **arg_all, t_idx=t_idx)[0]
                    if offload_model:
                        noise_pred = noise_pred.to(torch.device('cpu'))
                    profiler.stop("dit_forward", t0, per_step=True)

                    t0 = profiler.start()
                    noise_pred_cond, noise_pred_uncond = get_cfg_group().all_gather(
                        noise_pred, separate_tensors=True
                    )
                    profiler.stop("cfg_allgather", t0, per_step=True)
                    if offload_model:
                        t0 = profiler.start()
                        torch.cuda.empty_cache()
                        profiler.stop("step_empty_cache_after_gather", t0, per_step=True)
                else:
                    t0 = profiler.start()
                    noise_pred_cond = self.model(
                        latent_model_input, t=timestep, **arg_c, t_idx=t_idx)[0]
                    if offload_model:
                        noise_pred_cond = noise_pred_cond.to(torch.device('cpu'))
                    profiler.stop("dit_forward_cond", t0, per_step=True)
                    if offload_model:
                        t0 = profiler.start()
                        torch.cuda.empty_cache()
                        profiler.stop("step_empty_cache_after_cond", t0, per_step=True)

                    t0 = profiler.start()
                    noise_pred_uncond = self.model(
                        latent_model_input, t=timestep, **arg_null, t_idx=t_idx)[0]
                    if offload_model:
                        noise_pred_uncond = noise_pred_uncond.to(torch.device('cpu'))
                    profiler.stop("dit_forward_uncond", t0, per_step=True)
                    if offload_model:
                        t0 = profiler.start()
                        torch.cuda.empty_cache()
                        profiler.stop("step_empty_cache_after_uncond", t0, per_step=True)

                t0 = profiler.start()
                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)
                profiler.stop("cfg_combine", t0, per_step=True)

                t0 = profiler.start()
                if offload_model:
                    latent = latent.to(torch.device('cpu'))
                profiler.stop("latent_to_target_device", t0, per_step=True)

                t0 = profiler.start()
                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latent.unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                profiler.stop("scheduler_step", t0, per_step=True)
                latent = temp_x0.squeeze(0)

                t0 = profiler.start()
                x0 = [latent.to(self.device)] if offload_model else [latent]
                profiler.stop("x0_to_device", t0, per_step=True)
                del latent_model_input, timestep
                profiler.stop("denoise_step_total", step_start, per_step=True)
            profiler.stop("denoise_loop_total", denoise_loop_start)

            if offload_model:
                t0 = profiler.start()
                self.model.cpu()
                torch.cuda.empty_cache()
                profiler.stop("dit_offload_cpu", t0)
            t0 = profiler.start()
            with VAE_patch_parallel():
                videos = self.vae.decode(x0)
            profiler.stop("vae_decode", t0)

        del noise, latent
        del sample_scheduler

        if self.dit_fsdp:
            self.model._fsdp_wrapped_module.freqs_list = None
        else:
            self.model.freqs_list = None

        if offload_model:
            t0 = profiler.start()
            gc.collect()
            torch.cuda.synchronize()
            profiler.stop("post_gc_sync", t0)
        if dist.is_initialized():
            t0 = profiler.start()
            dist.barrier()
            profiler.stop("final_barrier", t0)

        profiler.stop("request_total", total_start)
        profiler.report()
        report_attention_profiler()
        report_block_profiler()

        return videos[0] if self.rank == 0 else None
