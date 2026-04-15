all: relu_custom0
relu_custom0:
	cd /mnt/workspace/claude_code/MultiKernelBench/ascend_op_projects/ReluCustom/build_out/op_kernel/ReluCustom_ascend910b/kernel_0 && bash /mnt/workspace/claude_code/MultiKernelBench/ascend_op_projects/ReluCustom/build_out/op_kernel/ReluCustom_ascend910b/bin_param/ReluCustom-relu_custom-0.sh --kernel-src=$(CPP) $(PY) $(OUT) $(MAKE)