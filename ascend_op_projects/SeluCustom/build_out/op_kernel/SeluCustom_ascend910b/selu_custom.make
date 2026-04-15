all: selu_custom0
selu_custom0:
	cd /mnt/workspace/claude_code/MultiKernelBench/ascend_op_projects/SeluCustom/build_out/op_kernel/SeluCustom_ascend910b/kernel_0 && bash /mnt/workspace/claude_code/MultiKernelBench/ascend_op_projects/SeluCustom/build_out/op_kernel/SeluCustom_ascend910b/bin_param/SeluCustom-selu_custom-0.sh --kernel-src=$(CPP) $(PY) $(OUT) $(MAKE)