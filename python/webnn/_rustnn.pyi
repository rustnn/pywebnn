"""Types for the direct RustNN bindings. No Python graph representation is exposed."""

from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

MLDimension: TypeAlias = int | dict[str, str | int]

class ML:
    def __init__(self) -> None: ...
    def create_context(
        self,
        power_preference: str = "default",
        accelerated: bool = True,
        backend: str = "auto",
    ) -> MLContext: ...

class MLContext:
    def create_graph_builder(self) -> MLGraphBuilder: ...
    def load_graph(self, path: str) -> MLGraph:
        """Load and compile through RustNN, resolving weights beside the graph file."""

    def create_tensor(
        self,
        shape: list[int],
        data_type: str,
        readable: bool = False,
        writable: bool = False,
    ) -> MLTensor: ...
    def dispatch(
        self, graph: MLGraph, inputs: dict[str, MLTensor], outputs: dict[str, MLTensor]
    ) -> None: ...
    def read_tensor(self, tensor: MLTensor) -> np.ndarray: ...
    def write_tensor(self, tensor: MLTensor, data: npt.ArrayLike) -> None: ...
    def resize_tensor(self, tensor: MLTensor, shape: list[int]) -> None: ...
    def set_tensor_capacity(self, tensor: MLTensor, max_shape: list[int]) -> None: ...
    @property
    def accelerated(self) -> bool: ...
    def op_support_limits(self) -> None:
        """Raises NotImplementedError while unimplemented in RustNN."""

class MLOperand:
    """Opaque native operand; query shape and dtype through its builder before build()."""

class MLGraphBuilder:
    """Create builders through MLContext.create_graph_builder()."""

    def input(self, name: str, shape: list[int], data_type: str) -> MLOperand: ...
    def constant(
        self,
        value: Any,
        shape: list[int] | None = None,
        data_type: str | None = None,
    ) -> MLOperand: ...
    def add(
        self, a: MLOperand, b: MLOperand, *, label: str | None = None
    ) -> MLOperand: ...
    def sub(
        self, a: MLOperand, b: MLOperand, *, label: str | None = None
    ) -> MLOperand: ...
    def mul(
        self, a: MLOperand, b: MLOperand, *, label: str | None = None
    ) -> MLOperand: ...
    def div(
        self, a: MLOperand, b: MLOperand, *, label: str | None = None
    ) -> MLOperand: ...
    def pow(
        self, a: MLOperand, b: MLOperand, *, label: str | None = None
    ) -> MLOperand: ...
    def max(
        self, a: MLOperand, b: MLOperand, *, label: str | None = None
    ) -> MLOperand: ...
    def min(
        self, a: MLOperand, b: MLOperand, *, label: str | None = None
    ) -> MLOperand: ...
    def matmul(
        self, a: MLOperand, b: MLOperand, *, label: str | None = None
    ) -> MLOperand: ...
    def gemm(
        self,
        a: MLOperand,
        b: MLOperand,
        c: MLOperand | None = None,
        alpha: float = 1.0,
        beta: float = 1.0,
        a_transpose: bool = False,
        b_transpose: bool = False,
        *,
        label: str | None = None,
    ) -> MLOperand: ...
    def conv2d(
        self,
        input: MLOperand,
        filter: MLOperand,
        strides: list[int] | None = None,
        dilations: list[int] | None = None,
        pads: list[int] | None = None,
        groups: int | None = None,
        input_layout: str | None = None,
        filter_layout: str | None = None,
        bias: MLOperand | None = None,
        *,
        label: str | None = None,
    ) -> MLOperand: ...
    def conv_transpose2d(
        self,
        input: MLOperand,
        filter: MLOperand,
        strides: list[int] | None = None,
        dilations: list[int] | None = None,
        pads: list[int] | None = None,
        output_padding: list[int] | None = None,
        output_sizes: list[int] | None = None,
        groups: int | None = None,
        input_layout: str | None = None,
        filter_layout: str | None = None,
        bias: MLOperand | None = None,
        *,
        label: str | None = None,
    ) -> MLOperand: ...
    def average_pool2d(
        self,
        input: MLOperand,
        window_dimensions: list[int] | None = None,
        strides: list[int] | None = None,
        dilations: list[int] | None = None,
        pads: list[int] | None = None,
        layout: str | None = None,
        output_shape_rounding: str | None = None,
        output_sizes: list[int] | None = None,
        *,
        label: str | None = None,
    ) -> MLOperand: ...
    def max_pool2d(
        self,
        input: MLOperand,
        window_dimensions: list[int] | None = None,
        strides: list[int] | None = None,
        dilations: list[int] | None = None,
        pads: list[int] | None = None,
        layout: str | None = None,
        output_shape_rounding: str | None = None,
        output_sizes: list[int] | None = None,
        *,
        label: str | None = None,
    ) -> MLOperand: ...
    def l2_pool2d(
        self,
        input: MLOperand,
        window_dimensions: list[int] | None = None,
        strides: list[int] | None = None,
        dilations: list[int] | None = None,
        pads: list[int] | None = None,
        layout: str | None = None,
        output_shape_rounding: str | None = None,
        output_sizes: list[int] | None = None,
        *,
        label: str | None = None,
    ) -> MLOperand: ...
    def global_average_pool(
        self,
        input: MLOperand,
        layout: str | None = None,
        *,
        label: str | None = None,
    ) -> MLOperand: ...
    def global_max_pool(
        self,
        input: MLOperand,
        layout: str | None = None,
        *,
        label: str | None = None,
    ) -> MLOperand: ...
    def batch_normalization(
        self,
        input: MLOperand,
        mean: MLOperand,
        variance: MLOperand,
        scale: MLOperand | None = None,
        bias: MLOperand | None = None,
        epsilon: float = 1e-5,
        axis: int = 1,
        *,
        label: str | None = None,
    ) -> MLOperand: ...
    def instance_normalization(
        self,
        input: MLOperand,
        scale: MLOperand | None = None,
        bias: MLOperand | None = None,
        epsilon: float = 1e-5,
        layout: str | None = None,
        *,
        label: str | None = None,
    ) -> MLOperand: ...
    def layer_normalization(
        self,
        input: MLOperand,
        scale: MLOperand | None = None,
        bias: MLOperand | None = None,
        epsilon: float = 1e-5,
        axes: list[int] | None = None,
        *,
        label: str | None = None,
    ) -> MLOperand: ...
    def relu(self, x: MLOperand, *, label: str | None = None) -> MLOperand: ...
    def sigmoid(self, x: MLOperand, *, label: str | None = None) -> MLOperand: ...
    def tanh(self, x: MLOperand, *, label: str | None = None) -> MLOperand: ...
    def softmax(
        self, x: MLOperand, axis: int, *, label: str | None = None
    ) -> MLOperand: ...
    def abs(self, x: MLOperand, *, label: str | None = None) -> MLOperand: ...
    def ceil(self, x: MLOperand, *, label: str | None = None) -> MLOperand: ...
    def floor(self, x: MLOperand, *, label: str | None = None) -> MLOperand: ...
    def round_even(self, x: MLOperand, *, label: str | None = None) -> MLOperand: ...
    def neg(self, x: MLOperand, *, label: str | None = None) -> MLOperand: ...
    def sign(self, x: MLOperand, *, label: str | None = None) -> MLOperand: ...
    def exp(self, x: MLOperand, *, label: str | None = None) -> MLOperand: ...
    def log(self, x: MLOperand, *, label: str | None = None) -> MLOperand: ...
    def sqrt(self, x: MLOperand, *, label: str | None = None) -> MLOperand: ...
    def reciprocal(self, x: MLOperand, *, label: str | None = None) -> MLOperand: ...
    def sin(self, x: MLOperand, *, label: str | None = None) -> MLOperand: ...
    def cos(self, x: MLOperand, *, label: str | None = None) -> MLOperand: ...
    def tan(self, x: MLOperand, *, label: str | None = None) -> MLOperand: ...
    def erf(self, x: MLOperand, *, label: str | None = None) -> MLOperand: ...
    def identity(self, x: MLOperand, *, label: str | None = None) -> MLOperand: ...
    def equal(
        self, a: MLOperand, b: MLOperand, *, label: str | None = None
    ) -> MLOperand: ...
    def greater(
        self, a: MLOperand, b: MLOperand, *, label: str | None = None
    ) -> MLOperand: ...
    def greater_or_equal(
        self,
        a: MLOperand,
        b: MLOperand,
        *,
        label: str | None = None,
    ) -> MLOperand: ...
    def lesser(
        self, a: MLOperand, b: MLOperand, *, label: str | None = None
    ) -> MLOperand: ...
    def lesser_or_equal(
        self,
        a: MLOperand,
        b: MLOperand,
        *,
        label: str | None = None,
    ) -> MLOperand: ...
    def not_equal(
        self,
        a: MLOperand,
        b: MLOperand,
        *,
        label: str | None = None,
) -> MLOperand: ...
    def logical_not(
        self, x: MLOperand, *, label: str | None = None
    ) -> MLOperand: ...
    def logical_not(self, x: MLOperand, *, label: str | None = None) -> MLOperand: ...
    def logical_and(
        self,
        a: MLOperand,
        b: MLOperand,
        *,
        label: str | None = None,
    ) -> MLOperand: ...
    def logical_or(
        self,
        a: MLOperand,
        b: MLOperand,
        *,
        label: str | None = None,
    ) -> MLOperand: ...
    def logical_xor(
        self,
        a: MLOperand,
        b: MLOperand,
        *,
        label: str | None = None,
    ) -> MLOperand: ...
    def dequantize_linear(
        self,
        input: MLOperand,
        scale: MLOperand,
        zero_point: MLOperand | None = None,
        *,
        label: str | None = None,
    ) -> MLOperand: ...
    def quantize_linear(
        self,
        input: MLOperand,
        scale: MLOperand,
        zero_point: MLOperand | None = None,
        *,
        label: str | None = None,
    ) -> MLOperand: ...
    def reshape(
        self,
        x: MLOperand,
        new_shape: list[MLDimension],
        *,
        label: str | None = None,
    ) -> MLOperand: ...
    def reduce_sum(
        self,
        input: MLOperand,
        axes: list[int] | None = None,
        keep_dimensions: bool = False,
        *,
        label: str | None = None,
    ) -> MLOperand: ...
    def reduce_mean(
        self,
        input: MLOperand,
        axes: list[int] | None = None,
        keep_dimensions: bool = False,
        *,
        label: str | None = None,
    ) -> MLOperand: ...
    def reduce_max(
        self,
        input: MLOperand,
        axes: list[int] | None = None,
        keep_dimensions: bool = False,
        *,
        label: str | None = None,
    ) -> MLOperand: ...
    def reduce_min(
        self,
        input: MLOperand,
        axes: list[int] | None = None,
        keep_dimensions: bool = False,
        *,
        label: str | None = None,
    ) -> MLOperand: ...
    def reduce_product(
        self,
        input: MLOperand,
        axes: list[int] | None = None,
        keep_dimensions: bool = False,
        *,
        label: str | None = None,
    ) -> MLOperand: ...
    def reduce_l1(
        self,
        input: MLOperand,
        axes: list[int] | None = None,
        keep_dimensions: bool = False,
        *,
        label: str | None = None,
    ) -> MLOperand: ...
    def reduce_l2(
        self,
        input: MLOperand,
        axes: list[int] | None = None,
        keep_dimensions: bool = False,
        *,
        label: str | None = None,
    ) -> MLOperand: ...
    def reduce_log_sum(
        self,
        input: MLOperand,
        axes: list[int] | None = None,
        keep_dimensions: bool = False,
        *,
        label: str | None = None,
    ) -> MLOperand: ...
    def reduce_log_sum_exp(
        self,
        input: MLOperand,
        axes: list[int] | None = None,
        keep_dimensions: bool = False,
        *,
        label: str | None = None,
    ) -> MLOperand: ...
    def reduce_sum_square(
        self,
        input: MLOperand,
        axes: list[int] | None = None,
        keep_dimensions: bool = False,
        *,
        label: str | None = None,
    ) -> MLOperand: ...
    def transpose(
        self,
        input: MLOperand,
        permutation: list[int] | None = None,
        *,
        label: str | None = None,
    ) -> MLOperand: ...
    def concat(
        self,
        inputs: list[MLOperand],
        axis: int,
        *,
        label: str | None = None,
    ) -> MLOperand: ...
    def slice(
        self,
        input: MLOperand,
        starts: list[int],
        sizes: list[MLDimension],
        strides: list[int] | None = None,
        *,
        label: str | None = None,
    ) -> MLOperand: ...
    def expand(
        self,
        input: MLOperand,
        new_shape: list[MLDimension],
        *,
        label: str | None = None,
    ) -> MLOperand: ...
    def gather(
        self,
        input: MLOperand,
        indices: MLOperand,
        axis: int = 0,
        *,
        label: str | None = None,
    ) -> MLOperand: ...
    def gather_elements(
        self,
        input: MLOperand,
        indices: MLOperand,
        axis: int = 0,
        *,
        label: str | None = None,
    ) -> MLOperand: ...
    def gather_nd(
        self,
        input: MLOperand,
        indices: MLOperand,
        *,
        label: str | None = None,
    ) -> MLOperand: ...
    def split(
        self,
        input: MLOperand,
        splits: Any,
        axis: int = 0,
        *,
        label: str | None = None,
    ) -> list[MLOperand]: ...
    def where_(
        self,
        condition: MLOperand,
        true_value: MLOperand,
        false_value: MLOperand,
        *,
        label: str | None = None,
    ) -> MLOperand: ...
    def pad(
        self,
        input: MLOperand,
        padding: list[int],
        mode: str | None = None,
        value: Any | None = None,
        *,
        label: str | None = None,
    ) -> MLOperand: ...
    def gelu(self, input: MLOperand, *, label: str | None = None) -> MLOperand: ...
    def squeeze(
        self,
        input: MLOperand,
        axes: list[int] | None = None,
        *,
        label: str | None = None,
    ) -> MLOperand: ...
    def unsqueeze(
        self,
        input: MLOperand,
        axes: list[int] | None = None,
        *,
        label: str | None = None,
    ) -> MLOperand: ...
    def arg_max(
        self,
        input: MLOperand,
        axis: int,
        keep_dimensions: bool = False,
        output_data_type: str | None = None,
        *,
        label: str | None = None,
    ) -> MLOperand: ...
    def arg_min(
        self,
        input: MLOperand,
        axis: int,
        keep_dimensions: bool = False,
        output_data_type: str | None = None,
        *,
        label: str | None = None,
    ) -> MLOperand: ...
    def cast(
        self,
        input: MLOperand,
        data_type: str,
        *,
        label: str | None = None,
    ) -> MLOperand: ...
    def save(self, outputs: dict[str, MLOperand], path: str) -> None: ...
    def build(self, outputs: dict[str, MLOperand]) -> MLGraph: ...
    def scatter_elements(
        self,
        input: MLOperand,
        indices: MLOperand,
        updates: MLOperand,
        axis: int = 0,
        *,
        label: str | None = None,
    ) -> MLOperand: ...
    def scatter_nd(
        self,
        input: MLOperand,
        indices: MLOperand,
        updates: MLOperand,
        *,
        label: str | None = None,
    ) -> MLOperand: ...
    def tile(
        self,
        input: MLOperand,
        repetitions: list[int],
        *,
        label: str | None = None,
    ) -> MLOperand: ...
    def triangular(
        self,
        input: MLOperand,
        upper: bool = True,
        diagonal: int = 0,
        *,
        label: str | None = None,
    ) -> MLOperand: ...
    def hard_sigmoid(
        self,
        input: MLOperand,
        alpha: float = 0.2,
        beta: float = 0.5,
        *,
        label: str | None = None,
    ) -> MLOperand: ...
    def hard_swish(
        self, input: MLOperand, *, label: str | None = None
    ) -> MLOperand: ...
    def softplus(
        self, input: MLOperand, *, label: str | None = None
    ) -> MLOperand: ...
    def softsign(
        self, input: MLOperand, *, label: str | None = None
    ) -> MLOperand: ...
    def softplus(self, input: MLOperand, *, label: str | None = None) -> MLOperand: ...
    def softsign(self, input: MLOperand, *, label: str | None = None) -> MLOperand: ...
    def clamp(
        self,
        input: MLOperand,
        min_value: Any | None = None,
        max_value: Any | None = None,
        *,
        label: str | None = None,
    ) -> MLOperand: ...
    def elu(
        self,
        input: MLOperand,
        alpha: float = 1.0,
        *,
        label: str | None = None,
    ) -> MLOperand: ...
    def leaky_relu(
        self,
        input: MLOperand,
        alpha: float = 0.01,
        *,
        label: str | None = None,
    ) -> MLOperand: ...
    def prelu(
        self,
        input: MLOperand,
        slope: MLOperand,
        *,
        label: str | None = None,
    ) -> MLOperand: ...
    def linear(
        self,
        input: MLOperand,
        alpha: float = 1.0,
        beta: float = 0.0,
        *,
        label: str | None = None,
    ) -> MLOperand: ...
    def is_nan(self, input: MLOperand, *, label: str | None = None) -> MLOperand: ...
    def is_infinite(
        self, input: MLOperand, *, label: str | None = None
    ) -> MLOperand: ...
    def shape(self, input: MLOperand, *, label: str | None = None) -> MLOperand: ...
    def cumulative_sum(
        self,
        input: MLOperand,
        axis: int,
        exclusive: bool = False,
        reversed: bool = False,
        *,
        label: str | None = None,
    ) -> MLOperand: ...
    def reverse(
        self,
        input: MLOperand,
        axes: list[int] | None = None,
        *,
        label: str | None = None,
    ) -> MLOperand: ...
    def resample2d(
        self,
        input: MLOperand,
        sizes: list[int] | None = None,
        scales: list[float] | None = None,
        mode: str = "nearest-neighbor",
        axes: list[int] | None = None,
        *,
        label: str | None = None,
    ) -> MLOperand: ...
    def gru(
        self,
        input: MLOperand,
        weight: MLOperand,
        recurrent_weight: MLOperand,
        steps: int,
        hidden_size: int,
        bias: MLOperand | None = None,
        recurrent_bias: MLOperand | None = None,
        initial_hidden_state: MLOperand | None = None,
        reset_after: bool = False,
        return_sequence: bool = False,
        direction: str = "forward",
        layout: str = "zrn",
        activations: list[str] | None = None,
        *,
        label: str | None = None,
    ) -> list[MLOperand]: ...
    def gru_cell(
        self,
        input: MLOperand,
        weight: MLOperand,
        recurrent_weight: MLOperand,
        hidden_state: MLOperand,
        hidden_size: int,
        bias: MLOperand | None = None,
        recurrent_bias: MLOperand | None = None,
        reset_after: bool = False,
        layout: str = "zrn",
        activations: list[str] | None = None,
        *,
        label: str | None = None,
    ) -> MLOperand: ...
    def lstm(
        self,
        input: MLOperand,
        weight: MLOperand,
        recurrent_weight: MLOperand,
        steps: int,
        hidden_size: int,
        bias: MLOperand | None = None,
        recurrent_bias: MLOperand | None = None,
        peephole_weight: MLOperand | None = None,
        initial_hidden_state: MLOperand | None = None,
        initial_cell_state: MLOperand | None = None,
        return_sequence: bool = False,
        direction: str = "forward",
        layout: str = "iofg",
        activations: list[str] | None = None,
        *,
        label: str | None = None,
    ) -> list[MLOperand]: ...
    def lstm_cell(
        self,
        input: MLOperand,
        weight: MLOperand,
        recurrent_weight: MLOperand,
        hidden_state: MLOperand,
        cell_state: MLOperand,
        hidden_size: int,
        bias: MLOperand | None = None,
        recurrent_bias: MLOperand | None = None,
        peephole_weight: MLOperand | None = None,
        layout: str = "iofg",
        activations: list[str] | None = None,
        *,
        label: str | None = None,
    ) -> list[MLOperand]: ...
    def operand_shape(self, operand: MLOperand) -> list[int]: ...
    def operand_data_type(self, operand: MLOperand) -> str: ...

class MLGraph:
    """Native compiled graph passed to MLContext.dispatch()."""

    @property
    def input_descriptors(self) -> dict[str, dict[str, Any]]: ...
    @property
    def output_descriptors(self) -> dict[str, dict[str, Any]]: ...

class MLTensor:
    @property
    def data_type(self) -> str: ...
    @property
    def shape(self) -> list[int]: ...
    @property
    def readable(self) -> bool: ...
    @property
    def writable(self) -> bool: ...
    def destroy(self) -> None:
        """Raises NotImplementedError while unimplemented in RustNN."""
