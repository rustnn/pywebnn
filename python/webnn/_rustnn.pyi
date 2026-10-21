"""Types for the direct RustNN bindings. No Python graph representation is exposed."""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import numpy.typing as npt

MLDimension = Union[int, Dict[str, Union[str, int]]]

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
        ...

    def create_tensor(
        self,
        shape: List[int],
        data_type: str,
        readable: bool = False,
        writable: bool = False,
    ) -> MLTensor: ...
    def dispatch(
        self, graph: MLGraph, inputs: Dict[str, MLTensor], outputs: Dict[str, MLTensor]
    ) -> None: ...
    def read_tensor(self, tensor: MLTensor) -> np.ndarray: ...
    def write_tensor(self, tensor: MLTensor, data: npt.ArrayLike) -> None: ...
    def resize_tensor(self, tensor: MLTensor, shape: List[int]) -> None: ...
    def set_tensor_capacity(self, tensor: MLTensor, max_shape: List[int]) -> None: ...
    @property
    def accelerated(self) -> bool: ...
    def op_support_limits(self) -> None:
        """Raises NotImplementedError while unimplemented in RustNN."""
        ...

class MLOperand:
    """Opaque native operand; query shape and dtype through its builder before build()."""

class MLGraphBuilder:
    """Create builders through MLContext.create_graph_builder()."""

    def input(self, name: str, shape: List[int], data_type: str) -> MLOperand: ...
    def constant(
        self,
        value: Any,
        shape: Optional[List[int]] = None,
        data_type: Optional[str] = None,
    ) -> MLOperand: ...
    def add(
        self, a: MLOperand, b: MLOperand, *, label: Optional[str] = None
    ) -> MLOperand: ...
    def sub(
        self, a: MLOperand, b: MLOperand, *, label: Optional[str] = None
    ) -> MLOperand: ...
    def mul(
        self, a: MLOperand, b: MLOperand, *, label: Optional[str] = None
    ) -> MLOperand: ...
    def div(
        self, a: MLOperand, b: MLOperand, *, label: Optional[str] = None
    ) -> MLOperand: ...
    def pow(
        self, a: MLOperand, b: MLOperand, *, label: Optional[str] = None
    ) -> MLOperand: ...
    def max(
        self, a: MLOperand, b: MLOperand, *, label: Optional[str] = None
    ) -> MLOperand: ...
    def min(
        self, a: MLOperand, b: MLOperand, *, label: Optional[str] = None
    ) -> MLOperand: ...
    def matmul(
        self, a: MLOperand, b: MLOperand, *, label: Optional[str] = None
    ) -> MLOperand: ...
    def gemm(
        self,
        a: MLOperand,
        b: MLOperand,
        c: Optional[MLOperand] = None,
        alpha: float = 1.0,
        beta: float = 1.0,
        a_transpose: bool = False,
        b_transpose: bool = False,
        *,
        label: Optional[str] = None,
    ) -> MLOperand: ...
    def conv2d(
        self,
        input: MLOperand,
        filter: MLOperand,
        strides: Optional[List[int]] = None,
        dilations: Optional[List[int]] = None,
        pads: Optional[List[int]] = None,
        groups: Optional[int] = None,
        input_layout: Optional[str] = None,
        filter_layout: Optional[str] = None,
        bias: Optional[MLOperand] = None,
        *,
        label: Optional[str] = None,
    ) -> MLOperand: ...
    def conv_transpose2d(
        self,
        input: MLOperand,
        filter: MLOperand,
        strides: Optional[List[int]] = None,
        dilations: Optional[List[int]] = None,
        pads: Optional[List[int]] = None,
        output_padding: Optional[List[int]] = None,
        output_sizes: Optional[List[int]] = None,
        groups: Optional[int] = None,
        input_layout: Optional[str] = None,
        filter_layout: Optional[str] = None,
        bias: Optional[MLOperand] = None,
        *,
        label: Optional[str] = None,
    ) -> MLOperand: ...
    def average_pool2d(
        self,
        input: MLOperand,
        window_dimensions: Optional[List[int]] = None,
        strides: Optional[List[int]] = None,
        dilations: Optional[List[int]] = None,
        pads: Optional[List[int]] = None,
        layout: Optional[str] = None,
        output_shape_rounding: Optional[str] = None,
        output_sizes: Optional[List[int]] = None,
        *,
        label: Optional[str] = None,
    ) -> MLOperand: ...
    def max_pool2d(
        self,
        input: MLOperand,
        window_dimensions: Optional[List[int]] = None,
        strides: Optional[List[int]] = None,
        dilations: Optional[List[int]] = None,
        pads: Optional[List[int]] = None,
        layout: Optional[str] = None,
        output_shape_rounding: Optional[str] = None,
        output_sizes: Optional[List[int]] = None,
        *,
        label: Optional[str] = None,
    ) -> MLOperand: ...
    def l2_pool2d(
        self,
        input: MLOperand,
        window_dimensions: Optional[List[int]] = None,
        strides: Optional[List[int]] = None,
        dilations: Optional[List[int]] = None,
        pads: Optional[List[int]] = None,
        layout: Optional[str] = None,
        output_shape_rounding: Optional[str] = None,
        output_sizes: Optional[List[int]] = None,
        *,
        label: Optional[str] = None,
    ) -> MLOperand: ...
    def global_average_pool(
        self,
        input: MLOperand,
        layout: Optional[str] = None,
        *,
        label: Optional[str] = None,
    ) -> MLOperand: ...
    def global_max_pool(
        self,
        input: MLOperand,
        layout: Optional[str] = None,
        *,
        label: Optional[str] = None,
    ) -> MLOperand: ...
    def batch_normalization(
        self,
        input: MLOperand,
        mean: MLOperand,
        variance: MLOperand,
        scale: Optional[MLOperand] = None,
        bias: Optional[MLOperand] = None,
        epsilon: float = 1e-5,
        axis: int = 1,
        *,
        label: Optional[str] = None,
    ) -> MLOperand: ...
    def instance_normalization(
        self,
        input: MLOperand,
        scale: Optional[MLOperand] = None,
        bias: Optional[MLOperand] = None,
        epsilon: float = 1e-5,
        layout: Optional[str] = None,
        *,
        label: Optional[str] = None,
    ) -> MLOperand: ...
    def layer_normalization(
        self,
        input: MLOperand,
        scale: Optional[MLOperand] = None,
        bias: Optional[MLOperand] = None,
        epsilon: float = 1e-5,
        axes: Optional[List[int]] = None,
        *,
        label: Optional[str] = None,
    ) -> MLOperand: ...
    def relu(self, x: MLOperand, *, label: Optional[str] = None) -> MLOperand: ...
    def sigmoid(self, x: MLOperand, *, label: Optional[str] = None) -> MLOperand: ...
    def tanh(self, x: MLOperand, *, label: Optional[str] = None) -> MLOperand: ...
    def softmax(
        self, x: MLOperand, axis: int, *, label: Optional[str] = None
    ) -> MLOperand: ...
    def abs(self, x: MLOperand, *, label: Optional[str] = None) -> MLOperand: ...
    def ceil(self, x: MLOperand, *, label: Optional[str] = None) -> MLOperand: ...
    def floor(self, x: MLOperand, *, label: Optional[str] = None) -> MLOperand: ...
    def round_even(self, x: MLOperand, *, label: Optional[str] = None) -> MLOperand: ...
    def neg(self, x: MLOperand, *, label: Optional[str] = None) -> MLOperand: ...
    def sign(self, x: MLOperand, *, label: Optional[str] = None) -> MLOperand: ...
    def exp(self, x: MLOperand, *, label: Optional[str] = None) -> MLOperand: ...
    def log(self, x: MLOperand, *, label: Optional[str] = None) -> MLOperand: ...
    def sqrt(self, x: MLOperand, *, label: Optional[str] = None) -> MLOperand: ...
    def reciprocal(self, x: MLOperand, *, label: Optional[str] = None) -> MLOperand: ...
    def sin(self, x: MLOperand, *, label: Optional[str] = None) -> MLOperand: ...
    def cos(self, x: MLOperand, *, label: Optional[str] = None) -> MLOperand: ...
    def tan(self, x: MLOperand, *, label: Optional[str] = None) -> MLOperand: ...
    def erf(self, x: MLOperand, *, label: Optional[str] = None) -> MLOperand: ...
    def identity(self, x: MLOperand, *, label: Optional[str] = None) -> MLOperand: ...
    def equal(
        self, a: MLOperand, b: MLOperand, *, label: Optional[str] = None
    ) -> MLOperand: ...
    def greater(
        self, a: MLOperand, b: MLOperand, *, label: Optional[str] = None
    ) -> MLOperand: ...
    def greater_or_equal(
        self,
        a: MLOperand,
        b: MLOperand,
        *,
        label: Optional[str] = None,
    ) -> MLOperand: ...
    def lesser(
        self, a: MLOperand, b: MLOperand, *, label: Optional[str] = None
    ) -> MLOperand: ...
    def lesser_or_equal(
        self,
        a: MLOperand,
        b: MLOperand,
        *,
        label: Optional[str] = None,
    ) -> MLOperand: ...
    def not_equal(
        self,
        a: MLOperand,
        b: MLOperand,
        *,
        label: Optional[str] = None,
    ) -> MLOperand: ...
    def logical_not(
        self, x: MLOperand, *, label: Optional[str] = None
    ) -> MLOperand: ...
    def logical_and(
        self,
        a: MLOperand,
        b: MLOperand,
        *,
        label: Optional[str] = None,
    ) -> MLOperand: ...
    def logical_or(
        self,
        a: MLOperand,
        b: MLOperand,
        *,
        label: Optional[str] = None,
    ) -> MLOperand: ...
    def logical_xor(
        self,
        a: MLOperand,
        b: MLOperand,
        *,
        label: Optional[str] = None,
    ) -> MLOperand: ...
    def dequantize_linear(
        self,
        input: MLOperand,
        scale: MLOperand,
        zero_point: Optional[MLOperand] = None,
        *,
        label: Optional[str] = None,
    ) -> MLOperand: ...
    def quantize_linear(
        self,
        input: MLOperand,
        scale: MLOperand,
        zero_point: Optional[MLOperand] = None,
        *,
        label: Optional[str] = None,
    ) -> MLOperand: ...
    def reshape(
        self,
        x: MLOperand,
        new_shape: List[MLDimension],
        *,
        label: Optional[str] = None,
    ) -> MLOperand: ...
    def reduce_sum(
        self,
        input: MLOperand,
        axes: Optional[List[int]] = None,
        keep_dimensions: bool = False,
        *,
        label: Optional[str] = None,
    ) -> MLOperand: ...
    def reduce_mean(
        self,
        input: MLOperand,
        axes: Optional[List[int]] = None,
        keep_dimensions: bool = False,
        *,
        label: Optional[str] = None,
    ) -> MLOperand: ...
    def reduce_max(
        self,
        input: MLOperand,
        axes: Optional[List[int]] = None,
        keep_dimensions: bool = False,
        *,
        label: Optional[str] = None,
    ) -> MLOperand: ...
    def reduce_min(
        self,
        input: MLOperand,
        axes: Optional[List[int]] = None,
        keep_dimensions: bool = False,
        *,
        label: Optional[str] = None,
    ) -> MLOperand: ...
    def reduce_product(
        self,
        input: MLOperand,
        axes: Optional[List[int]] = None,
        keep_dimensions: bool = False,
        *,
        label: Optional[str] = None,
    ) -> MLOperand: ...
    def reduce_l1(
        self,
        input: MLOperand,
        axes: Optional[List[int]] = None,
        keep_dimensions: bool = False,
        *,
        label: Optional[str] = None,
    ) -> MLOperand: ...
    def reduce_l2(
        self,
        input: MLOperand,
        axes: Optional[List[int]] = None,
        keep_dimensions: bool = False,
        *,
        label: Optional[str] = None,
    ) -> MLOperand: ...
    def reduce_log_sum(
        self,
        input: MLOperand,
        axes: Optional[List[int]] = None,
        keep_dimensions: bool = False,
        *,
        label: Optional[str] = None,
    ) -> MLOperand: ...
    def reduce_log_sum_exp(
        self,
        input: MLOperand,
        axes: Optional[List[int]] = None,
        keep_dimensions: bool = False,
        *,
        label: Optional[str] = None,
    ) -> MLOperand: ...
    def reduce_sum_square(
        self,
        input: MLOperand,
        axes: Optional[List[int]] = None,
        keep_dimensions: bool = False,
        *,
        label: Optional[str] = None,
    ) -> MLOperand: ...
    def transpose(
        self,
        input: MLOperand,
        permutation: Optional[List[int]] = None,
        *,
        label: Optional[str] = None,
    ) -> MLOperand: ...
    def concat(
        self,
        inputs: List[MLOperand],
        axis: int,
        *,
        label: Optional[str] = None,
    ) -> MLOperand: ...
    def slice(
        self,
        input: MLOperand,
        starts: List[int],
        sizes: List[MLDimension],
        strides: Optional[List[int]] = None,
        *,
        label: Optional[str] = None,
    ) -> MLOperand: ...
    def expand(
        self,
        input: MLOperand,
        new_shape: List[MLDimension],
        *,
        label: Optional[str] = None,
    ) -> MLOperand: ...
    def gather(
        self,
        input: MLOperand,
        indices: MLOperand,
        axis: int = 0,
        *,
        label: Optional[str] = None,
    ) -> MLOperand: ...
    def gather_elements(
        self,
        input: MLOperand,
        indices: MLOperand,
        axis: int = 0,
        *,
        label: Optional[str] = None,
    ) -> MLOperand: ...
    def gather_nd(
        self,
        input: MLOperand,
        indices: MLOperand,
        *,
        label: Optional[str] = None,
    ) -> MLOperand: ...
    def split(
        self,
        input: MLOperand,
        splits: Any,
        axis: int = 0,
        *,
        label: Optional[str] = None,
    ) -> List[MLOperand]: ...
    def where_(
        self,
        condition: MLOperand,
        true_value: MLOperand,
        false_value: MLOperand,
        *,
        label: Optional[str] = None,
    ) -> MLOperand: ...
    def pad(
        self,
        input: MLOperand,
        padding: List[int],
        mode: Optional[str] = None,
        value: Optional[Any] = None,
        *,
        label: Optional[str] = None,
    ) -> MLOperand: ...
    def gelu(self, input: MLOperand, *, label: Optional[str] = None) -> MLOperand: ...
    def squeeze(
        self,
        input: MLOperand,
        axes: Optional[List[int]] = None,
        *,
        label: Optional[str] = None,
    ) -> MLOperand: ...
    def unsqueeze(
        self,
        input: MLOperand,
        axes: Optional[List[int]] = None,
        *,
        label: Optional[str] = None,
    ) -> MLOperand: ...
    def arg_max(
        self,
        input: MLOperand,
        axis: int,
        keep_dimensions: bool = False,
        output_data_type: Optional[str] = None,
        *,
        label: Optional[str] = None,
    ) -> MLOperand: ...
    def arg_min(
        self,
        input: MLOperand,
        axis: int,
        keep_dimensions: bool = False,
        output_data_type: Optional[str] = None,
        *,
        label: Optional[str] = None,
    ) -> MLOperand: ...
    def cast(
        self,
        input: MLOperand,
        data_type: str,
        *,
        label: Optional[str] = None,
    ) -> MLOperand: ...
    def save(self, outputs: Dict[str, MLOperand], path: str) -> None: ...
    def build(self, outputs: Dict[str, MLOperand]) -> MLGraph: ...
    def scatter_elements(
        self,
        input: MLOperand,
        indices: MLOperand,
        updates: MLOperand,
        axis: int = 0,
        *,
        label: Optional[str] = None,
    ) -> MLOperand: ...
    def scatter_nd(
        self,
        input: MLOperand,
        indices: MLOperand,
        updates: MLOperand,
        *,
        label: Optional[str] = None,
    ) -> MLOperand: ...
    def tile(
        self,
        input: MLOperand,
        repetitions: List[int],
        *,
        label: Optional[str] = None,
    ) -> MLOperand: ...
    def triangular(
        self,
        input: MLOperand,
        upper: bool = True,
        diagonal: int = 0,
        *,
        label: Optional[str] = None,
    ) -> MLOperand: ...
    def hard_sigmoid(
        self,
        input: MLOperand,
        alpha: float = 0.2,
        beta: float = 0.5,
        *,
        label: Optional[str] = None,
    ) -> MLOperand: ...
    def hard_swish(
        self, input: MLOperand, *, label: Optional[str] = None
    ) -> MLOperand: ...
    def softplus(
        self, input: MLOperand, *, label: Optional[str] = None
    ) -> MLOperand: ...
    def softsign(
        self, input: MLOperand, *, label: Optional[str] = None
    ) -> MLOperand: ...
    def clamp(
        self,
        input: MLOperand,
        min_value: Optional[Any] = None,
        max_value: Optional[Any] = None,
        *,
        label: Optional[str] = None,
    ) -> MLOperand: ...
    def elu(
        self,
        input: MLOperand,
        alpha: float = 1.0,
        *,
        label: Optional[str] = None,
    ) -> MLOperand: ...
    def leaky_relu(
        self,
        input: MLOperand,
        alpha: float = 0.01,
        *,
        label: Optional[str] = None,
    ) -> MLOperand: ...
    def prelu(
        self,
        input: MLOperand,
        slope: MLOperand,
        *,
        label: Optional[str] = None,
    ) -> MLOperand: ...
    def linear(
        self,
        input: MLOperand,
        alpha: float = 1.0,
        beta: float = 0.0,
        *,
        label: Optional[str] = None,
    ) -> MLOperand: ...
    def is_nan(self, input: MLOperand, *, label: Optional[str] = None) -> MLOperand: ...
    def is_infinite(
        self, input: MLOperand, *, label: Optional[str] = None
    ) -> MLOperand: ...
    def shape(self, input: MLOperand, *, label: Optional[str] = None) -> MLOperand: ...
    def cumulative_sum(
        self,
        input: MLOperand,
        axis: int,
        exclusive: bool = False,
        reversed: bool = False,
        *,
        label: Optional[str] = None,
    ) -> MLOperand: ...
    def reverse(
        self,
        input: MLOperand,
        axes: Optional[List[int]] = None,
        *,
        label: Optional[str] = None,
    ) -> MLOperand: ...
    def resample2d(
        self,
        input: MLOperand,
        sizes: Optional[List[int]] = None,
        scales: Optional[List[float]] = None,
        mode: str = "nearest-neighbor",
        axes: Optional[List[int]] = None,
        *,
        label: Optional[str] = None,
    ) -> MLOperand: ...
    def gru(
        self,
        input: MLOperand,
        weight: MLOperand,
        recurrent_weight: MLOperand,
        steps: int,
        hidden_size: int,
        bias: Optional[MLOperand] = None,
        recurrent_bias: Optional[MLOperand] = None,
        initial_hidden_state: Optional[MLOperand] = None,
        reset_after: bool = False,
        return_sequence: bool = False,
        direction: str = "forward",
        layout: str = "zrn",
        activations: Optional[List[str]] = None,
        *,
        label: Optional[str] = None,
    ) -> List[MLOperand]: ...
    def gru_cell(
        self,
        input: MLOperand,
        weight: MLOperand,
        recurrent_weight: MLOperand,
        hidden_state: MLOperand,
        hidden_size: int,
        bias: Optional[MLOperand] = None,
        recurrent_bias: Optional[MLOperand] = None,
        reset_after: bool = False,
        layout: str = "zrn",
        activations: Optional[List[str]] = None,
        *,
        label: Optional[str] = None,
    ) -> MLOperand: ...
    def lstm(
        self,
        input: MLOperand,
        weight: MLOperand,
        recurrent_weight: MLOperand,
        steps: int,
        hidden_size: int,
        bias: Optional[MLOperand] = None,
        recurrent_bias: Optional[MLOperand] = None,
        peephole_weight: Optional[MLOperand] = None,
        initial_hidden_state: Optional[MLOperand] = None,
        initial_cell_state: Optional[MLOperand] = None,
        return_sequence: bool = False,
        direction: str = "forward",
        layout: str = "iofg",
        activations: Optional[List[str]] = None,
        *,
        label: Optional[str] = None,
    ) -> List[MLOperand]: ...
    def lstm_cell(
        self,
        input: MLOperand,
        weight: MLOperand,
        recurrent_weight: MLOperand,
        hidden_state: MLOperand,
        cell_state: MLOperand,
        hidden_size: int,
        bias: Optional[MLOperand] = None,
        recurrent_bias: Optional[MLOperand] = None,
        peephole_weight: Optional[MLOperand] = None,
        layout: str = "iofg",
        activations: Optional[List[str]] = None,
        *,
        label: Optional[str] = None,
    ) -> List[MLOperand]: ...
    def operand_shape(self, operand: MLOperand) -> List[int]: ...
    def operand_data_type(self, operand: MLOperand) -> str: ...

class MLGraph:
    """Native compiled graph passed to MLContext.dispatch()."""

    @property
    def input_descriptors(self) -> Dict[str, Dict[str, Any]]: ...
    @property
    def output_descriptors(self) -> Dict[str, Dict[str, Any]]: ...

class MLTensor:
    @property
    def data_type(self) -> str: ...
    @property
    def shape(self) -> List[int]: ...
    @property
    def readable(self) -> bool: ...
    @property
    def writable(self) -> bool: ...
    def destroy(self) -> None:
        """Raises NotImplementedError while unimplemented in RustNN."""
        ...
