all: log_softmax_custom0
log_softmax_custom0:
	cd /data/w00936672/MultiKernelBench/ascend_op_projects/LogSoftmaxCustom/build_out/op_kernel/LogSoftmaxCustom_ascend910b/kernel_0 && bash /data/w00936672/MultiKernelBench/ascend_op_projects/LogSoftmaxCustom/build_out/op_kernel/LogSoftmaxCustom_ascend910b/bin_param/LogSoftmaxCustom-log_softmax_custom-0.sh --kernel-src=$(CPP) $(PY) $(OUT) $(MAKE)