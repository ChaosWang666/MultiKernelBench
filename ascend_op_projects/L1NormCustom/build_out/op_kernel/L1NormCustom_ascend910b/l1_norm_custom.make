all: l1_norm_custom0
l1_norm_custom0:
	cd /mnt/workspace/claude_code/MultiKernelBench/ascend_op_projects/L1NormCustom/build_out/op_kernel/L1NormCustom_ascend910b/kernel_0 && bash /mnt/workspace/claude_code/MultiKernelBench/ascend_op_projects/L1NormCustom/build_out/op_kernel/L1NormCustom_ascend910b/bin_param/L1NormCustom-l1_norm_custom-0.sh --kernel-src=$(CPP) $(PY) $(OUT) $(MAKE)