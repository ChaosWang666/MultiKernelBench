all: elu_custom0
elu_custom0:
	cd /data/w00936672/MultiKernelBench/ascend_op_projects/EluCustom/build_out/op_kernel/EluCustom_ascend910b/kernel_0 && bash /data/w00936672/MultiKernelBench/ascend_op_projects/EluCustom/build_out/op_kernel/EluCustom_ascend910b/bin_param/EluCustom-elu_custom-0.sh --kernel-src=$(CPP) $(PY) $(OUT) $(MAKE)