all: lamb_custom0
lamb_custom0:
	cd /mnt/workspace/claude_code/MultiKernelBench/ascend_op_projects/LambCustom/build_out/op_kernel/LambCustom_ascend910b/kernel_0 && bash /mnt/workspace/claude_code/MultiKernelBench/ascend_op_projects/LambCustom/build_out/op_kernel/LambCustom_ascend910b/bin_param/LambCustom-lamb_custom-0.sh --kernel-src=$(CPP) $(PY) $(OUT) $(MAKE)