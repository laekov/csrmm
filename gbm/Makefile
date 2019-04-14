libcus.so : cusparse.cu
	nvcc --compiler-options -fPIC --std=c++11 -c $< -o $@.o
	nvcc -shared $@.o -lcusparse -lcublas -o $@

cusparse : cusparse.cu
	nvcc -O3 --std=c++11 $< -lcusparse -lcublas -o $@

exg_cusparse : exg_cusparse.cu
	nvcc -O3 --std=c++11 $< -lcusparse -lcublas -o $@
