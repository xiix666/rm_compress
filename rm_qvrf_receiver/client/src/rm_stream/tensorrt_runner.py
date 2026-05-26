"""Small TensorRT FP32-I/O NCHW runner for receiver-side pixel paths."""

from __future__ import annotations

import logging
import os
import sys
import ctypes
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)

_NP_TO_CTYPES = {
    np.dtype(np.float32): ctypes.c_float,
    np.dtype(np.float16): ctypes.c_uint16,
    np.dtype(np.uint8): ctypes.c_uint8,
    np.dtype(np.int8): ctypes.c_int8,
    np.dtype(np.int32): ctypes.c_int32,
}


def _cuda_check(result, what: str):
    from cuda.bindings import runtime as cudart

    status = result[0] if isinstance(result, tuple) else result
    if status == cudart.cudaError_t.cudaSuccess:
        return result[1:] if isinstance(result, tuple) else ()
    err = cudart.cudaGetErrorString(status)
    msg = err[1].decode("utf-8", errors="replace") if isinstance(err, tuple) and len(err) > 1 else str(status)
    raise RuntimeError(f"{what}: {msg}")


def _shape_str(shape: tuple[int, ...]) -> str:
    return "[" + ",".join(str(int(v)) for v in shape) + "]"


class TensorRtRunner:
    """Load one fixed-shape FP32 TensorRT engine and run host numpy input/output."""

    def __init__(
        self,
        engine_path: str | Path,
        expected_input_shape: tuple[int, ...],
        expected_output_shape: tuple[int, ...],
        device_id: int = 0,
        label: str = "tensor",
        check_finite: bool | None = None,
        input_dtype: np.dtype | type = np.float32,
        output_dtype: np.dtype | type = np.float32,
    ) -> None:
        import tensorrt as trt
        from cuda.bindings import runtime as cudart

        self.engine_path = Path(engine_path).resolve()
        self.expected_input_shape = tuple(int(v) for v in expected_input_shape)
        self.expected_output_shape = tuple(int(v) for v in expected_output_shape)
        self.device_id = int(device_id)
        self.label = label
        self.input_dtype = np.dtype(input_dtype)
        self.output_dtype = np.dtype(output_dtype)
        if check_finite is None:
            check_finite = os.environ.get("RM_STREAM_TRT_CHECK_FINITE", "").strip() in ("1", "true", "yes")
        self.check_finite = bool(check_finite)
        if not self.engine_path.exists():
            raise FileNotFoundError(f"TensorRT {label} engine not found: {self.engine_path}")
        if self.engine_path.stat().st_size <= 0:
            raise RuntimeError(f"TensorRT {label} engine is empty: {self.engine_path}")
        if self.device_id < 0:
            raise ValueError("TensorRT CUDA device id must be >= 0")

        count = _cuda_check(cudart.cudaGetDeviceCount(), "cudaGetDeviceCount")[0]
        if self.device_id >= count:
            raise RuntimeError(
                f"TensorRT CUDA device {self.device_id} unavailable; CUDA device count is {count}"
            )
        _cuda_check(cudart.cudaSetDevice(self.device_id), "cudaSetDevice")
        props = _cuda_check(cudart.cudaGetDeviceProperties(self.device_id), "cudaGetDeviceProperties")[0]
        self.device_name = props.name.decode("utf-8", errors="replace")
        self.compute_capability = f"{props.major}{props.minor}"

        class _Logger(trt.ILogger):
            def __init__(self):
                super().__init__()

            def log(self, severity, msg):
                if severity <= trt.ILogger.WARNING:
                    LOGGER.warning("[TensorRT] %s", msg)

        self._trt = trt
        self._cudart = cudart
        self._logger = _Logger()
        self._runtime = trt.Runtime(self._logger)
        data = self.engine_path.read_bytes()
        self._engine = self._runtime.deserialize_cuda_engine(data)
        if self._engine is None:
            raise RuntimeError(
                f"TensorRT {label} deserialize_cuda_engine failed for {self.engine_path}. "
                "Rebuild the engine for this TensorRT version/GPU/shape."
            )
        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise RuntimeError(f"TensorRT {label} create_execution_context failed")

        inputs: list[str] = []
        outputs: list[str] = []
        for idx in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(idx)
            mode = self._engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                inputs.append(name)
            elif mode == trt.TensorIOMode.OUTPUT:
                outputs.append(name)
        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError(f"TensorRT {label} engine must have one input and one output tensor")
        self.input_name = inputs[0]
        self.output_name = outputs[0]
        trt_dtype_by_np = {
            np.dtype(np.float32): trt.float32,
            np.dtype(np.float16): trt.float16,
            np.dtype(np.uint8): trt.uint8,
            np.dtype(np.int8): trt.int8,
            np.dtype(np.int32): trt.int32,
        }
        if self.input_dtype not in trt_dtype_by_np:
            raise RuntimeError(f"TensorRT {label} unsupported input dtype: {self.input_dtype}")
        if self.output_dtype not in trt_dtype_by_np:
            raise RuntimeError(f"TensorRT {label} unsupported output dtype: {self.output_dtype}")
        if self._engine.get_tensor_dtype(self.input_name) != trt_dtype_by_np[self.input_dtype]:
            raise RuntimeError(f"TensorRT {label} input must be {self.input_dtype}")
        if self._engine.get_tensor_dtype(self.output_name) != trt_dtype_by_np[self.output_dtype]:
            raise RuntimeError(f"TensorRT {label} output must be {self.output_dtype}")

        input_shape = tuple(int(v) for v in self._engine.get_tensor_shape(self.input_name))
        if any(v < 0 for v in input_shape):
            if not self._context.set_input_shape(self.input_name, self.expected_input_shape):
                raise RuntimeError(f"TensorRT {label} failed to set fixed input shape")
            input_shape = tuple(int(v) for v in self._context.get_tensor_shape(self.input_name))
        output_shape = tuple(int(v) for v in self._context.get_tensor_shape(self.output_name))
        if any(v < 0 for v in output_shape):
            output_shape = tuple(int(v) for v in self._engine.get_tensor_shape(self.output_name))
        self.input_shape = input_shape
        self.output_shape = output_shape
        if self.input_shape != self.expected_input_shape:
            raise RuntimeError(
                f"TensorRT {label} input shape mismatch: expected "
                f"{_shape_str(self.expected_input_shape)}, got {_shape_str(self.input_shape)}"
            )
        if self.output_shape != self.expected_output_shape:
            raise RuntimeError(
                f"TensorRT {label} output shape mismatch: expected "
                f"{_shape_str(self.expected_output_shape)}, got {_shape_str(self.output_shape)}"
            )

        self._input_nbytes = int(np.prod(self.input_shape)) * self.input_dtype.itemsize
        self._output_nbytes = int(np.prod(self.output_shape)) * self.output_dtype.itemsize
        self._input_host_ptr = None
        self._output_host_ptr = None
        self._input = self._alloc_host_array(self.input_shape, self._input_nbytes, "input")
        self._output = self._alloc_host_array(self.output_shape, self._output_nbytes, "output")
        self._input_device = _cuda_check(cudart.cudaMalloc(self._input_nbytes), "cudaMalloc input")[0]
        self._output_device = _cuda_check(cudart.cudaMalloc(self._output_nbytes), "cudaMalloc output")[0]
        self._stream = _cuda_check(cudart.cudaStreamCreate(), "cudaStreamCreate")[0]
        if not self._context.set_tensor_address(self.input_name, int(self._input_device)):
            raise RuntimeError(f"TensorRT {label} failed to bind input tensor")
        if not self._context.set_tensor_address(self.output_name, int(self._output_device)):
            raise RuntimeError(f"TensorRT {label} failed to bind output tensor")

        print(
            f"TensorRT {label} loaded: device=cuda:{self.device_id} name={self.device_name} "
            f"cc={self.compute_capability} engine={self.engine_path} "
            f"input={_shape_str(self.input_shape)} output={_shape_str(self.output_shape)}",
            file=sys.stderr,
            flush=True,
        )

    def _alloc_host_array(self, shape: tuple[int, ...], nbytes: int, name: str) -> np.ndarray:
        try:
            ptr = _cuda_check(
                self._cudart.cudaHostAlloc(
                    nbytes,
                    self._cudart.cudaHostAllocWriteCombined
                    if name == "input"
                    else self._cudart.cudaHostAllocDefault,
                ),
                f"cudaHostAlloc {name}",
            )[0]
            addr = int(ptr)
            dtype = self.input_dtype if name == "input" else self.output_dtype
            ctype = _NP_TO_CTYPES.get(dtype)
            if ctype is None:
                raise RuntimeError(f"unsupported pinned host dtype: {dtype}")
            buf_type = ctype * (nbytes // dtype.itemsize)
            arr = np.ctypeslib.as_array(buf_type.from_address(addr)).view(dtype).reshape(shape)
            setattr(self, f"_{name}_host_ptr", ptr)
            return arr
        except Exception as exc:
            LOGGER.warning("TensorRT %s: pinned %s host allocation failed: %s", self.label, name, exc)
            dtype = self.input_dtype if name == "input" else self.output_dtype
            return np.empty(shape, dtype=dtype)

    def input_host_array(self) -> np.ndarray:
        """Return the reusable pinned input buffer for callers that can fill it directly."""
        return self._input

    def output_host_array(self) -> np.ndarray:
        """Return the reusable pinned output buffer. Valid until the next inference."""
        return self._output

    @property
    def output_device_ptr(self) -> int:
        return int(self._output_device)

    @property
    def output_nbytes(self) -> int:
        return self._output_nbytes

    @property
    def stream_ptr(self) -> int:
        return int(self._stream)

    def synchronize(self) -> None:
        _cuda_check(self._cudart.cudaStreamSynchronize(self._stream), "cudaStreamSynchronize")

    def close(self) -> None:
        if getattr(self, "_input_device", None):
            _cuda_check(self._cudart.cudaFree(self._input_device), "cudaFree input")
            self._input_device = None
        if getattr(self, "_output_device", None):
            _cuda_check(self._cudart.cudaFree(self._output_device), "cudaFree output")
            self._output_device = None
        if getattr(self, "_stream", None):
            _cuda_check(self._cudart.cudaStreamDestroy(self._stream), "cudaStreamDestroy")
            self._stream = None
        if getattr(self, "_input_host_ptr", None):
            _cuda_check(self._cudart.cudaFreeHost(self._input_host_ptr), "cudaFreeHost input")
            self._input_host_ptr = None
        if getattr(self, "_output_host_ptr", None):
            _cuda_check(self._cudart.cudaFreeHost(self._output_host_ptr), "cudaFreeHost output")
            self._output_host_ptr = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def infer(self, arr: np.ndarray, *, copy_output: bool = True) -> np.ndarray:
        self.infer_device(arr)
        _cuda_check(
            self._cudart.cudaMemcpyAsync(
                int(self._output.ctypes.data),
                int(self._output_device),
                self._output_nbytes,
                self._cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                self._stream,
            ),
            "cudaMemcpyAsync D2H",
        )
        self.synchronize()
        if self.check_finite and np.issubdtype(self.output_dtype, np.floating) and not np.isfinite(self._output).all():
            raise RuntimeError(f"TensorRT {self.label} produced non-finite output")
        return self._output.copy() if copy_output else self._output

    def infer_device(self, arr: np.ndarray) -> int:
        """Run inference and leave output on device. Caller owns stream synchronization."""
        if arr.dtype != self.input_dtype or not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr, dtype=self.input_dtype)
        if tuple(arr.shape) != self.input_shape:
            raise RuntimeError(
                f"TensorRT {self.label} input shape mismatch: expected "
                f"{_shape_str(self.input_shape)}, got {_shape_str(tuple(arr.shape))}"
            )
        if int(arr.ctypes.data) != int(self._input.ctypes.data):
            np.copyto(self._input, arr, casting="no")
        _cuda_check(
            self._cudart.cudaMemcpyAsync(
                int(self._input_device),
                int(self._input.ctypes.data),
                self._input_nbytes,
                self._cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                self._stream,
            ),
            "cudaMemcpyAsync H2D",
        )
        if not self._context.execute_async_v3(int(self._stream)):
            raise RuntimeError(f"TensorRT {self.label} execute_async_v3 failed")
        return int(self._output_device)
