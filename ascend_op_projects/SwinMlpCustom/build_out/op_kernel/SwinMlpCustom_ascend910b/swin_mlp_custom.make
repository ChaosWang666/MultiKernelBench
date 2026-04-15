all: swin_mlp_custom0
swin_mlp_custom0:
	cd /mnt/workspace/claude_code/MultiKernelBench/ascend_op_projects/SwinMlpCustom/build_out/op_kernel/SwinMlpCustom_ascend910b/kernel_0 && bash /mnt/workspace/claude_code/MultiKernelBench/ascend_op_projects/SwinMlpCustom/build_out/op_kernel/SwinMlpCustom_ascend910b/bin_param/SwinMlpCustom-swin_mlp_custom-0.sh --kernel-src=$(CPP) $(PY) $(OUT) $(MAKE)