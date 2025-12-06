
# for geminifs build test
from deepspeed.ops.op_builder import GeminiFSBuilder
assert GeminiFSBuilder().is_compatible(True)
assert GeminiFSBuilder().load(True)