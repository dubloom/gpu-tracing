from fastapi import FastAPI
from ddtrace import tracer
import torch
import ctypes

app = FastAPI()


# Charger la bibliothèque
cuda_lib = ctypes.CDLL('/home/ubuntu/gpu-tracing/.build/lib_ddog_cuda.so')
class CudaTraced(ctypes.Structure):
    _fields_ = [("action", ctypes.c_char_p),
                ("start", ctypes.c_double),
                ("duration", ctypes.c_double)]

# Définir la structure CudaTracedArray
class CudaTracedArray(ctypes.Structure):
    _fields_ = [("data", ctypes.POINTER(CudaTraced)),
                ("size", ctypes.c_size_t)]

cuda_lib.get_cuda_actions.restype = CudaTracedArray

@app.get("/")
async def read_root():
    return {"message": "Hello World!"}

@tracer.wrap('torch.multiplication', service='torch')
def torch_mul():
    cuda_lib.new_tracing()
    try:
        device = torch.cuda.current_device()
        x = torch.randn(1024, 1024).to(device)
        y = torch.randn(1024, 1024).to(device)
        z = torch.matmul(x, y)
        norm = torch.norm(z).item()
        cuda_actions_array = cuda_lib.get_cuda_actions()
        for i in range(cuda_actions_array.size):
            cuda_action = cuda_actions_array.data[i]
            cuda_action_span = tracer.trace(cuda_action.action.decode("utf-8"), service='cuda')
            cuda_action_span.finish()
            cuda_action_span.start = cuda_action.start
            cuda_action_span.duration = cuda_action.duration
        return norm 
    except Exception as e:
        print(str(e))
        return -1

@app.get("/matmul")
async def mat_mul():
    norm = torch_mul()
    if norm == -1:
        return {"error": "there was en error during torch code execution"}
    else:
        return {"norm": norm}
