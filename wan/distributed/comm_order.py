import os
import torch

_LAST_A2A_DONE_EVENT = {}


def _get_device_api():
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.npu
    if torch.cuda.is_available():
        return torch.cuda
    return None


def _resolve_enabled():
    raw = os.getenv("WAN_FSDP_ALLGATHER_WAIT_A2A", "0").strip().lower()
    if raw in ("0", "off", "false", ""):
        return False
    if raw in ("1", "on", "true", "wait"):
        return True
    return False


def is_fsdp_allgather_wait_a2a_enabled():
    return _resolve_enabled()


def record_last_a2a_done_event():
    if not _resolve_enabled():
        return
    device_api = _get_device_api()
    if device_api is None:
        return
    event = device_api.Event()
    event.record(device_api.current_stream())
    _LAST_A2A_DONE_EVENT[device_api.current_device()] = event


def get_last_a2a_done_event():
    if not _resolve_enabled():
        return None
    device_api = _get_device_api()
    if device_api is None:
        return None
    return _LAST_A2A_DONE_EVENT.get(device_api.current_device())


def wait_stream_on_last_a2a_done(stream):
    event = get_last_a2a_done_event()
    if event is None:
        return False
    if stream is None or not hasattr(stream, "wait_event"):
        return False
    stream.wait_event(event)
    return True


def wait_current_stream_on_last_a2a_done():
    device_api = _get_device_api()
    if device_api is None:
        return False
    event = get_last_a2a_done_event()
    if event is None:
        return False
    device_api.current_stream().wait_event(event)
    return True
