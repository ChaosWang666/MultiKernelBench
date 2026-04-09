all: softmax_custom0
softmax_custom0:
	cd /data/w00936672/MultiKernelBench/ascend_op_projects/SoftmaxCustom/build_out/op_kernel/SoftmaxCustom_ascend910b/kernel_0 && bash /data/w00936672/MultiKernelBench/ascend_op_projects/SoftmaxCustom/build_out/op_kernel/SoftmaxCustom_ascend910b/bin_param/SoftmaxCustom-softmax_custom-0.sh --kernel-src=$(CPP) $(PY) $(OUT) $(MAKE)