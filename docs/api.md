# API Reference

!!! note "Work in Progress"
    API documentation is being developed. For now, refer to the source code and examples.

## Modules

### commplax.module

- `scan_with(step, jit_backend)` - Create a scanner for equinox modules
- `pipe(*fns)` - Compose multiple step functions
- `step_at(idx, step_fn)` - Apply step function to specific module in tuple
- `allreduce(field, op)` - Reduce a field across ensemble

### commplax.equalizer

- `MIMOCell` - Adaptive MIMO equalizer
- `FOE` - Frequency offset estimator
- `CPR` - Carrier phase recovery

### commplax.adaptive_kernel

- `rls_cma()` - RLS-accelerated CMA
- `lms()` - LMS with optional bias
- `rls_lms()` - RLS-based LMS
- `cpr_partition_pll()` - Partition-based CPR for 16-QAM
- `cpane_ekf()` - EKF-based carrier phase estimator
