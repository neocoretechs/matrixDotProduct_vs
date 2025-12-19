g++ -shared -fPIC \
    -o /usr/lib/jni/libllamagpu.so \
    /home/jg/matrixDotProduct/matrixDotProduct/aarch64/Release/llamagpu.o \
    -L/usr/local/cuda-11.4/lib64 \
    -lcublas -lcudart \
    /home/jg/llama.cpp/build/bin/libllama.so
