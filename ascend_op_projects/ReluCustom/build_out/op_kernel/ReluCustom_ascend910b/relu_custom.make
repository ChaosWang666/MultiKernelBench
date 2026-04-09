all: relu_custom0
relu_custom0:
	cd /data/w00936672/MultiKernelBench/ascend_op_projects/ReluCustom/build_out/op_kernel/ReluCustom_ascend910b/kernel_0 && bash /data/w00936672/MultiKernelBench/ascend_op_projects/ReluCustom/build_out/op_kernel/ReluCustom_ascend910b/bin_param/ReluCustom-relu_custom-0.sh --kernel-src=$(CPP) $(PY) $(OUT) $(MAKE)