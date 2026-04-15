all: softsign_custom0
softsign_custom0:
	cd /mnt/workspace/claude_code/MultiKernelBench/ascend_op_projects/SoftsignCustom/build_out/op_kernel/SoftsignCustom_ascend910b/kernel_0 && bash /mnt/workspace/claude_code/MultiKernelBench/ascend_op_projects/SoftsignCustom/build_out/op_kernel/SoftsignCustom_ascend910b/bin_param/SoftsignCustom-softsign_custom-0.sh --kernel-src=$(CPP) $(PY) $(OUT) $(MAKE)