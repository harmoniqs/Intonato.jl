"""
    AbstractHardwareBackend

Abstract type for structured hardware interfaces.

Required methods (documented, not enforced):
- `upload_pulse!(backend, pulse)`
- `trigger!(backend)`
- `readout(backend) → raw data`
- `sample_rate(backend) → Float64`

Concrete backends are user-implemented for their specific hardware.
"""
abstract type AbstractHardwareBackend end
