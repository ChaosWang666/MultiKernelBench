all: hardsigmoid_custom0
hardsigmoid_custom0:
	cd /data/w00936672/MultiKernelBench/ascend_op_projects/HardsigmoidCustom/build_out/op_kernel/HardsigmoidCustom_ascend910b/kernel_0 && bash /data/w00936672/MultiKernelBench/ascend_op_projects/HardsigmoidCustom/build_out/op_kernel/HardsigmoidCustom_ascend910b/bin_param/HardsigmoidCustom-hardsigmoid_custom-0.sh --kernel-src=$(CPP) $(PY) $(OUT) $(MAKE)