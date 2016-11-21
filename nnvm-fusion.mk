CFLAGS += -I$(NNVM_FUSION_PATH)/include  -I$(CUDA_PATH)/include
NNVM_FUSION_SRC = $(wildcard $(NNVM_FUSION_PATH)/src/*.cc)
PLUGIN_OBJ += $(patsubst %.cc, build/%.o, $(NNVM_FUSION_SRC))
