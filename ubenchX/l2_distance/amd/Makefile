HIPCC      := hipcc
CXXFLAGS   := -O2 -std=c++17

# Auto-detect the GPU arch and CU count from rocminfo; override with
#   make ARCH=gfx950 NUM_CUS=256
ARCH       ?= $(shell rocminfo 2>/dev/null | awk '/^ *Name: *gfx/{print $$2; exit}')
NUM_CUS    ?= $(shell rocminfo 2>/dev/null | awk '/^ *Name: *gfx/{f=1} f && /Compute Unit:/{print $$3; exit}')

TARGET     := l2_pointer_chase
SRC        := l2_pointer_chase.hip

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRC)
ifeq ($(strip $(ARCH)),)
	$(error ARCH is empty: rocminfo found no gfx agent; pass ARCH=gfx950 NUM_CUS=256)
endif
ifeq ($(strip $(NUM_CUS)),)
	$(error NUM_CUS is empty: rocminfo found no Compute Unit count; pass NUM_CUS=256)
endif
	$(HIPCC) $(CXXFLAGS) --offload-arch=$(ARCH) -DNUM_CUS=$(NUM_CUS) -o $@ $<

run: $(TARGET)
	mkdir -p results
	./$(TARGET) > results/distance.csv

clean:
	rm -f $(TARGET)
	rm -rf results/*.csv
