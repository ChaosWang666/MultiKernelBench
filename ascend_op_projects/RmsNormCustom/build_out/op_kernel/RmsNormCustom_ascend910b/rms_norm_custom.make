all: rms_norm_custom0
rms_norm_custom0:
	cd /mnt/workspace/claude_code/MultiKernelBench/ascend_op_projects/RmsNormCustom/build_out/op_kernel/RmsNormCustom_ascend910b/kernel_0 && bash /mnt/workspace/claude_code/MultiKernelBench/ascend_op_projects/RmsNormCustom/build_out/op_kernel/RmsNormCustom_ascend910b/bin_param/RmsNormCustom-rms_norm_custom-0.sh --kernel-src=$(CPP) $(PY) $(OUT) $(MAKE)