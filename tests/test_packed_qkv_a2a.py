import importlib.util
import multiprocessing as mp
import queue
import socket
import traceback
import unittest
from pathlib import Path

try:
    import torch
    import torch.distributed as dist
except ModuleNotFoundError:
    torch = None
    dist = None


def _load_comm_module():
    comm_path = Path(__file__).resolve().parents[1] / "wan" / "distributed" / "comm.py"
    spec = importlib.util.spec_from_file_location("wan_distributed_comm_for_test", comm_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _run_packed_qkv_a2a_worker(rank, world_size, port, status_queue):
    try:
        dist.init_process_group(
            backend="gloo",
            rank=rank,
            world_size=world_size,
            init_method=f"tcp://127.0.0.1:{port}",
        )
        comm = _load_comm_module()

        bs = 2
        shard_seqlen = 3
        heads_per_rank = 2
        head_dim = 5
        head_count = world_size * heads_per_rank
        shape = (bs, shard_seqlen, head_count, head_dim)

        def make_tensor(offset):
            data = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32)
            return data.reshape(shape) + offset + rank * 100000

        q = make_tensor(0)
        k = make_tensor(1000)
        v = make_tensor(2000)

        ref_q = comm.all_to_all_4D(q, scatter_idx=2, gather_idx=1)
        ref_k = comm.all_to_all_4D(k, scatter_idx=2, gather_idx=1)
        ref_v = comm.all_to_all_4D(v, scatter_idx=2, gather_idx=1)
        packed_q, packed_k, packed_v = comm.all_to_all_4D_qkv_packed(q, k, v)

        torch.testing.assert_close(packed_q, ref_q, rtol=0, atol=0)
        torch.testing.assert_close(packed_k, ref_k, rtol=0, atol=0)
        torch.testing.assert_close(packed_v, ref_v, rtol=0, atol=0)
        status_queue.put(("ok", rank, ""))
    except RuntimeError as exc:
        message = str(exc)
        if "all_to_all" in message.lower() or "alltoall" in message.lower():
            status_queue.put(("skip", rank, message))
        else:
            status_queue.put(("fail", rank, traceback.format_exc()))
    except Exception:
        status_queue.put(("fail", rank, traceback.format_exc()))
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


class PackedQKVA2ATest(unittest.TestCase):

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_packed_qkv_a2a_matches_three_independent_collectives(self):
        if not dist.is_available():
            self.skipTest("torch.distributed is not available")

        world_size = 2
        port = _get_free_port()
        context = mp.get_context("spawn")
        status_queue = context.Queue()
        processes = [
            context.Process(
                target=_run_packed_qkv_a2a_worker,
                args=(rank, world_size, port, status_queue),
            )
            for rank in range(world_size)
        ]

        for process in processes:
            process.start()
        for process in processes:
            process.join(30)

        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()
                self.fail("packed QKV all-to-all test timed out")
            if process.exitcode != 0:
                self.fail(f"worker exited with code {process.exitcode}")

        statuses = []
        for _ in processes:
            try:
                statuses.append(status_queue.get_nowait())
            except queue.Empty:
                self.fail("worker exited without reporting status")

        skips = [message for status, _, message in statuses if status == "skip"]
        if skips:
            self.skipTest(skips[0])

        failures = [message for status, _, message in statuses if status == "fail"]
        if failures:
            self.fail("\n".join(failures))


if __name__ == "__main__":
    unittest.main()
