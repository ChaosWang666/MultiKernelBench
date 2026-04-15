all: hardtanh_custom0
hardtanh_custom0:
	cd /mnt/workspace/claude_code/MultiKernelBench/ascend_op_projects/HardtanhCustom/build_out/op_kernel/HardtanhCustom_ascend910b/kernel_0 && bash /mnt/workspace/claude_code/MultiKernelBench/ascend_op_projects/HardtanhCustom/build_out/op_kernel/HardtanhCustom_ascend910b/bin_param/HardtanhCustom-hardtanh_custom-0.sh --kernel-src=$(CPP) $(PY) $(OUT) $(MAKE)