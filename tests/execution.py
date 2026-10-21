"""Test fixture for native graphs with persistent, explicit tensor bindings.

Metadata is queried from the builder before compilation. This fixture is not part
of the package and never inspects a compiled graph or records its operations.
"""

import numpy as np


class GraphExecution:
    def __init__(self, context, builder, outputs, *, input_dtypes=None):
        self.input_dtypes = input_dtypes or {}
        self.context = context
        self.outputs = {
            name: context.create_tensor(
                builder.operand_shape(operand),
                builder.operand_data_type(operand),
                readable=True,
            )
            for name, operand in outputs.items()
        }
        self.graph = builder.build(outputs)
        self.inputs = {}

    def prepare(self, inputs):
        for name, value in inputs.items():
            value = np.asarray(value)
            if name not in self.inputs:
                self.inputs[name] = self.context.create_tensor(
                    list(value.shape),
                    self.input_dtypes.get(name, str(value.dtype)),
                    writable=True,
                )
            self.context.write_tensor(self.inputs[name], value)

    def run(self, inputs):
        self.prepare(inputs)
        self.context.dispatch(self.graph, self.inputs, self.outputs)
        return {
            name: self.context.read_tensor(tensor)
            for name, tensor in self.outputs.items()
        }
