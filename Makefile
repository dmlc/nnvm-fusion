ROOTDIR = $(CURDIR)

ifndef NNVM_PATH
	NNVM_PATH = $(ROOTDIR)/../..
endif

ifndef CUDA_PATH
	CUDA_PATH = /usr/local/cuda
endif

BUILD_PATH = $(NNVM_PATH)/build/nnvm-fuison
LIB_PATH = $(NNVM_PATH)/lib/nnvm-fusion
BIN_PATH = $(NNVM_PATH)/bin/nnvm-fuison

export CFLAGS =  -std=c++11 -Wall -O2 -msse2  -Wno-unknown-pragmas -funroll-loops\
     -Iinclude -I$(NNVM_PATH)/include -fPIC -I$(CUDA_PATH)/include

.PHONY: clean all

all: $(LIB_PATH)/libnnvm-fusion.a

SRC = $(wildcard src/*.cc src/*/*.cc)
ALL_OBJ = $(patsubst src/%.cc, $(BUILD_PATH)/%.o, $(SRC))
ALL_DEP = $(ALL_OBJ)

$(BUILD_PATH)/%.o: src/%.cc
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) -MM -MT $(BUILD_PATH)/$*.o $< >$(BUILD_PATH)/$*.d
	$(CXX) -c $(CFLAGS) -c $< -o $@

$(LIB_PATH)/libnnvm-fusion.a: $(ALL_DEP)
	@mkdir -p $(@D)
	ar crv $@ $(filter %.o, $?)

clean:
	$(RM) -rf $(BUILD_PATH) $(LIB_PATH) $(BIN_PATH) *~ */*~ */*/*~ */*/*/*~ */*.o */*/*.o */*/*/*.o cli_test $(LIB_PATH)/libnnvm-fusion.a

-include $(BUILD_PATH)/*.d
-include $(BUILD_PATH)/*/*.d
