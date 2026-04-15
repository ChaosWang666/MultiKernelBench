all: softmax_custom0
softmax_custom0:
	cd /mnt/workspace/claude_code/MultiKernelBench/ascend_op_projects/SoftmaxCustom/build_out/op_kernel/SoftmaxCustom_ascend910b/kernel_0 && bash /mnt/workspace/claude_code/MultiKernelBench/ascend_op_projects/SoftmaxCustom/build_out/op_kernel/SoftmaxCustom_ascend910b/bin_param/SoftmaxCustom-softmax_custom-0.sh --kernel-src=$(CPP) $(PY) $(OUT) $(MAKE)