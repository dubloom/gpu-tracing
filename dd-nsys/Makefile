example-cuda: examples/matmul.cu
	nvcc -g -G -o examples/matmul examples/matmul.cu -cudart shared

build:
	cd src && go build -o dd-nsys main.go && mv dd-nsys ..
	
clean:
	rm examples/matmul
	rm report*
	rm dd-nsys
