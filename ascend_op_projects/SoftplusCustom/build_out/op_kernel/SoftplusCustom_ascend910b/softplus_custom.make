all: softplus_custom0
softplus_custom0:
	cd /data/w00936672/MultiKernelBench/ascend_op_projects/SoftplusCustom/build_out/op_kernel/SoftplusCustom_ascend910b/kernel_0 && bash /data/w00936672/MultiKernelBench/ascend_op_projects/SoftplusCustom/build_out/op_kernel/SoftplusCustom_ascend910b/bin_param/SoftplusCustom-softplus_custom-0.sh --kernel-src=$(CPP) $(PY) $(OUT) $(MAKE)