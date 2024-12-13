# dd-cuda-tracing

This folder is made of two examples:
- matmul.cuda. It is a raw CUDA file which will do a 1024x1024 random matrix multiplication on GPU and compare the result with the same computation on CPU.
- a python example. It is a FastAPI application which exposes one endpoint /matmul. It returns the norm of a 1024x1024 random matrix multiplication. 

## CUDA example

In this example, the library will intercept CUDA Runtime API calls, after calling them, it will send traces to Datadog Backend. The CUDA code will have to tell when the trace starts and end thanks to the `create_and_set_active_span("matmul");` and `end_active_span();`calls.

To run the test, run:
```bash
make tracing-lib
make example-cuda
```

## Python example 

In this example, I'm trying to build a real use case. The example is a FastAPI app with one endpoint /matmul. It will compute a 1024x1024 random matrix multiplication using PyTorch and CUDA. We are using a different library which will store the CUDA Runtime API calls, Python code will then be able to retrieve the different calls and create trace. 

To run the test, run:
```bash
source ../.gpu-tracing-venv/bin/activate
make lib
make example-integration
```

In another terminal, you can now use the command curl -i "http://localhost:8000/matmul" to call the endpoint and create the trace.