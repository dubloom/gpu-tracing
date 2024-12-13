# dd-nsys

dd-nsys is a wrapper around nsys, the NVIDIA tool used to profile and analyze CUDA program execution. 

dd-nsys will execute the nsys command and then will extract all the useful information to send them to datadog backend. At the moment, it only supports nsys profile

## dd-nsys example 

To run the example, you must first copy the .env.example into a .env and put your API_KEY

You can then run:
```bash
make build
make example-cuda
./dd-nsys profile examples/matmul
```

The metrics emitted all start by the prefix cuda_