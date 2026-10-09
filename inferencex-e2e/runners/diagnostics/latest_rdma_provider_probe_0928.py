"""Read-only libibverbs probe using the candidate image's own providers."""

import ctypes as ct

verbs = ct.CDLL("libibverbs.so.1", use_errno=True)
verbs.ibv_get_device_list.argtypes = [ct.POINTER(ct.c_int)]
verbs.ibv_get_device_list.restype = ct.POINTER(ct.c_void_p)
verbs.ibv_get_device_name.argtypes = [ct.c_void_p]
verbs.ibv_get_device_name.restype = ct.c_char_p
verbs.ibv_open_device.argtypes = [ct.c_void_p]
verbs.ibv_open_device.restype = ct.c_void_p
verbs.ibv_close_device.argtypes = [ct.c_void_p]
verbs.ibv_close_device.restype = ct.c_int
verbs.ibv_free_device_list.argtypes = [ct.POINTER(ct.c_void_p)]
count = ct.c_int()
devices = verbs.ibv_get_device_list(ct.byref(count))
assert devices, f"device enumeration failed, errno={ct.get_errno()}"
try:
    assert count.value == 8, f"expected eight devices, found {count.value}"
    for i in range(count.value):
        name = verbs.ibv_get_device_name(devices[i]).decode()
        ctx = verbs.ibv_open_device(devices[i])
        assert ctx, f"ibv_open_device({name}) failed, errno={ct.get_errno()}"
        assert verbs.ibv_close_device(ctx) == 0
        print("RDMA_PROVIDER_OPEN_CLOSE_OK", name, flush=True)
finally:
    verbs.ibv_free_device_list(devices)
