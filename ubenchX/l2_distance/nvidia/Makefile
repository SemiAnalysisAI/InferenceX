NVCC       := nvcc
CXXFLAGS   := -O2 -std=c++17

# Auto-detect SM count; override with  make NUM_SMS=148
NUM_SMS    ?= $(shell nvidia-smi --query-gpu=count --format=csv,noheader 2>/dev/null | head -1)

# Architecture flags — add more -gencode lines for older GPUs.
# Blackwell (sm_100a / sm_120a):
ARCH       ?= -gencode=arch=compute_100a,code=\"sm_100a,compute_100a\"

TARGET     := l2_pointer_chase
SRC        := l2_pointer_chase.cu

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CXXFLAGS) $(ARCH) -DNUM_SMS=$(NUM_SMS) -o $@ $<

run: $(TARGET)
	mkdir -p results
	./$(TARGET) > results/distance.csv

clean:
	rm -f $(TARGET)
	rm -rf results/*.csv
