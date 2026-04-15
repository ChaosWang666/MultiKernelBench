all: mse_loss_custom0
mse_loss_custom0:
	cd /mnt/workspace/claude_code/MultiKernelBench/ascend_op_projects/MseLossCustom/build_out/op_kernel/MseLossCustom_ascend910b/kernel_0 && bash /mnt/workspace/claude_code/MultiKernelBench/ascend_op_projects/MseLossCustom/build_out/op_kernel/MseLossCustom_ascend910b/bin_param/MseLossCustom-mse_loss_custom-0.sh --kernel-src=$(CPP) $(PY) $(OUT) $(MAKE)