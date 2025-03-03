import torch
import logging
from typing import List, Optional, Union
import torch.distributed as dist
from yunchang import set_seq_parallel_pg

logger = logging.getLogger(__name__)

def init_distributed_environment(
    world_size: int = -1,
    rank: int = -1,
    distributed_init_method: str = "env://",
    local_rank: int = -1,
    backend: str = "hccl"
):
    logger.debug(
        "world_size=%d rank=%d local_rank=%d " "distributed_init_method=%s backend=%s",
        world_size,
        rank,
        local_rank,
        distributed_init_method,
        backend,
    )
    if not torch.distributed.is_initialized():
        assert distributed_init_method is not None, (
            "distributed_init_method must be provided when initializing "
            "distributed environment"
        )
        # this backend is used for WORLD
        torch.distributed.init_process_group(
            backend=backend,
            init_method=distributed_init_method,
            world_size=world_size,
            rank=rank,
        )
    torch.npu.set_device(rank)

def initialize_model_parallel(
    data_parallel_degree: int = 1,
    classifier_free_guidance_degree: int = 1,
    sequence_parallel_degree: int = 1,
    ulysses_degree: int = 1,
    ring_degree: int = 1,
    tensor_parallel_degree: int = 1,
    pipeline_parallel_degree: int = 1,
    vae_parallel_size: int = 0,
    backend: Optional[str] = None,
) -> None:

    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()
    backend = backend

    dit_parallel_size = (data_parallel_degree *
                     classifier_free_guidance_degree *
                     sequence_parallel_degree *
                     pipeline_parallel_degree *
                     tensor_parallel_degree)

    if world_size < dit_parallel_size:
        raise RuntimeError(
            f"world_size ({world_size}) is less than "
            f"tensor_parallel_degree ({tensor_parallel_degree}) x "
            f"pipeline_parallel_degree ({pipeline_parallel_degree}) x"
            f"sequence_parallel_degree ({sequence_parallel_degree}) x"
            f"classifier_free_guidance_degree "
            f"({classifier_free_guidance_degree}) x"
            f"data_parallel_degree ({data_parallel_degree})"
        )
    if world_size == 2 or world_size == 4 or world_size == 8:
        set_seq_parallel_pg(
            sp_ulysses_degree=ulysses_degree,
            sp_ring_degree=ring_degree,
            rank=dist.get_rank(),
            world_size=world_size
        )
    elif world_size == 16:
        set_seq_parallel_pg(
            sp_ulysses_degree=ulysses_degree,
            sp_ring_degree=ring_degree,
            rank=dist.get_rank(),
            world_size=world_size,
            use_ulysses_low=False
        )

def get_sequence_parallel_world_size():
    return dist.get_world_size()

def get_sequence_parallel_rank():
    return dist.get_rank()

def all_gather(
    input_: torch.Tensor, dim: int = 0, separate_tensors: bool = False
) -> Union[torch.Tensor, List[torch.Tensor]]:
    world_size = get_sequence_parallel_world_size()
    if world_size == 1:
        return input_
    assert (
        -input_.dim() <= dim < input_.dim()
    ), f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"
    if dim < 0:
        # Convert negative dim to positive.
        dim += input_.dim()
    # Allocate output tensor.
    input_size = list(input_.size())
    input_size[0] *= world_size
    output_tensor = torch.empty(
        input_size, dtype=input_.dtype, device=input_.device
    )
    # All-gather.
    torch.distributed.all_gather_into_tensor(
        output_tensor, input_
    )
    if dim != 0:
        input_size[0] //= world_size
        output_tensor = output_tensor.reshape([world_size, ] + input_size)
        output_tensor = output_tensor.movedim(0, dim)
    if separate_tensors:
        tensor_list = [
            output_tensor.view(-1)
            .narrow(0, input_.numel() * i, input_.numel())
            .view_as(input_)
            for i in range(world_size)
        ]
        return tensor_list
    else:
        input_size = list(input_.size())
        input_size[dim] = input_size[dim] * world_size
        # Reshape
        output_tensor = output_tensor.reshape(input_size)
        return output_tensor