"""Convert a fully int8 TFLite classifier to an equivalent QDQ ONNX model.

The ONNX graph carries the TFLite model's exact int8 weights, int32 biases and
scales in NCHW layout, with Q/DQ pairs wherever TFLite requantizes, so TiGrIS
and ONNX Runtime execute the same quantized model. Supported operators:
CONV_2D, DEPTHWISE_CONV_2D, ADD, global AVERAGE_POOL_2D, RESHAPE,
FULLY_CONNECTED, SOFTMAX; anything else is refused.
"""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from tflite.ActivationFunctionType import ActivationFunctionType
from tflite.AddOptions import AddOptions
from tflite.BuiltinOperator import BuiltinOperator
from tflite.Conv2DOptions import Conv2DOptions
from tflite.DepthwiseConv2DOptions import DepthwiseConv2DOptions
from tflite.FullyConnectedOptions import FullyConnectedOptions
from tflite.Model import Model
from tflite.Pool2DOptions import Pool2DOptions
from tflite.TensorType import TensorType

OPERATORS = {getattr(BuiltinOperator, name): name for name in dir(BuiltinOperator) if not name.startswith("_")}


def convert(tflite_bytes, name):
    model = Model.GetRootAs(tflite_bytes, 0)
    graph = model.Subgraphs(0)
    nodes, initializers = [], []

    def shape(index):
        tensor = graph.Tensors(index)
        return [tensor.Shape(k) for k in range(tensor.ShapeLength())]

    def data(index):
        tensor = graph.Tensors(index)
        dtype = {TensorType.INT8: np.int8, TensorType.INT32: np.int32}[tensor.Type()]
        raw = model.Buffers(tensor.Buffer()).DataAsNumpy()
        return np.frombuffer(raw.tobytes(), dtype=dtype).reshape(shape(index))

    def quantization(index):
        q = graph.Tensors(index).Quantization()
        return (np.atleast_1d(np.asarray(q.ScaleAsNumpy(), dtype=np.float32)),
                np.atleast_1d(np.asarray(q.ZeroPointAsNumpy(), dtype=np.int64)))

    def constant(array, key):
        initializers.append(numpy_helper.from_array(array, key))
        return key

    def dequantized(key, values, scale, zero_point, axis, zp_dtype):
        constant(values, key)
        constant(scale if scale.size > 1 else np.float32(scale[0]), key + "_s")
        constant(zero_point.astype(zp_dtype) if zero_point.size > 1
                 else np.array(zero_point[0], dtype=zp_dtype), key + "_z")
        attributes = {"axis": axis} if axis is not None and scale.size > 1 else {}
        nodes.append(helper.make_node("DequantizeLinear", [key, key + "_s", key + "_z"],
                                      [key + "_dq"], **attributes))
        return key + "_dq"

    def requantized(source, index, tag):
        scale, zero_point = quantization(index)
        constant(np.float32(scale[0]), tag + "_as")
        constant(np.array(zero_point[0], dtype=np.int8), tag + "_az")
        nodes.append(helper.make_node("QuantizeLinear", [source, tag + "_as", tag + "_az"], [tag + "_q"]))
        nodes.append(helper.make_node("DequantizeLinear", [tag + "_q", tag + "_as", tag + "_az"], [tag + "_dq"]))
        return tag + "_dq"

    def activation(source, fused, tag):
        if fused == ActivationFunctionType.RELU:
            nodes.append(helper.make_node("Relu", [source], [tag + "_relu"]))
            return tag + "_relu"
        if fused == ActivationFunctionType.RELU6:
            constant(np.float32(0.0), tag + "_lo")
            constant(np.float32(6.0), tag + "_hi")
            nodes.append(helper.make_node("Clip", [source, tag + "_lo", tag + "_hi"], [tag + "_relu6"]))
            return tag + "_relu6"
        if fused != ActivationFunctionType.NONE:
            raise ValueError(f"unsupported fused activation {fused}")
        return source

    def options(operator, kind):
        opts = kind()
        opts.Init(operator.BuiltinOptions().Bytes, operator.BuiltinOptions().Pos)
        return opts

    input_index = graph.Inputs(0)
    in_shape = shape(input_index)
    graph_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, in_shape[3], in_shape[1], in_shape[2]])
    values = {input_index: requantized("input", input_index, "input")}

    for position in range(graph.OperatorsLength()):
        operator = graph.Operators(position)
        code = model.OperatorCodes(operator.OpcodeIndex())
        kind = OPERATORS[max(code.BuiltinCode(), code.DeprecatedBuiltinCode())]
        inputs = [operator.Inputs(j) for j in range(operator.InputsLength())]
        output = operator.Outputs(0)
        tag = f"l{position}"

        if kind in ("CONV_2D", "DEPTHWISE_CONV_2D"):
            depthwise = kind == "DEPTHWISE_CONV_2D"
            weights = data(inputs[1])
            weights = np.transpose(weights, (3, 0, 1, 2) if depthwise else (0, 3, 1, 2))
            w_scale, w_zero = quantization(inputs[1])
            b_scale, b_zero = quantization(inputs[2])
            opts = options(operator, DepthwiseConv2DOptions if depthwise else Conv2DOptions)
            kh, kw = weights.shape[2], weights.shape[3]
            sh, sw = opts.StrideH(), opts.StrideW()
            in_h, in_w = shape(inputs[0])[1:3]
            out_h, out_w = shape(output)[1:3]
            pad_h = max((out_h - 1) * sh + kh - in_h, 0)
            pad_w = max((out_w - 1) * sw + kw - in_w, 0)
            w = dequantized(tag + "_w", weights, w_scale, np.zeros_like(w_zero), 0, np.int8)
            b = dequantized(tag + "_b", data(inputs[2]).astype(np.int32), b_scale,
                            np.zeros_like(b_zero), 0, np.int32)
            nodes.append(helper.make_node(
                "Conv", [values[inputs[0]], w, b], [tag + "_conv"], kernel_shape=[kh, kw],
                strides=[sh, sw], pads=[pad_h // 2, pad_w // 2, pad_h - pad_h // 2, pad_w - pad_w // 2],
                group=weights.shape[0] if depthwise else 1))
            values[output] = requantized(
                activation(tag + "_conv", opts.FusedActivationFunction(), tag), output, tag)

        elif kind == "AVERAGE_POOL_2D":
            opts = options(operator, Pool2DOptions)
            in_h, in_w = shape(inputs[0])[1:3]
            if (opts.FilterHeight(), opts.FilterWidth()) != (in_h, in_w) or shape(output)[1:3] != [1, 1]:
                raise ValueError("only a global average pool is supported")
            # AveragePool over the whole window, not GlobalAveragePool: TFLite's
            # AVERAGE_POOL_2D divides the integer sum with rounding half away
            # from zero, which is AveragePool's int8 semantics; a global mean
            # requantizes like MEAN and can differ by one LSB.
            nodes.append(helper.make_node("AveragePool", [values[inputs[0]]], [tag + "_gap"],
                                          kernel_shape=[in_h, in_w], strides=[1, 1]))
            values[output] = requantized(
                activation(tag + "_gap", opts.FusedActivationFunction(), tag), output, tag)

        elif kind == "ADD":
            if shape(inputs[0]) != shape(inputs[1]):
                raise ValueError("only an ADD of equal shapes is supported")
            nodes.append(helper.make_node("Add", [values[inputs[0]], values[inputs[1]]], [tag + "_add"]))
            fused = options(operator, AddOptions).FusedActivationFunction()
            values[output] = requantized(activation(tag + "_add", fused, tag), output, tag)

        elif kind == "RESHAPE":
            if len(shape(output)) != 2:
                raise ValueError("only a flattening reshape is supported")
            nodes.append(helper.make_node("Flatten", [values[inputs[0]]], [tag + "_flat"], axis=1))
            values[output] = requantized(tag + "_flat", output, tag)

        elif kind == "FULLY_CONNECTED":
            weights = data(inputs[1])
            w_scale, w_zero = quantization(inputs[1])
            w = dequantized(tag + "_w", weights, w_scale, np.zeros_like(w_zero), 0, np.int8)
            if len(inputs) > 2 and inputs[2] >= 0:
                b_scale, b_zero = quantization(inputs[2])
                b = dequantized(tag + "_b", data(inputs[2]).astype(np.int32), b_scale,
                                np.zeros_like(b_zero), 0, np.int32)
            else:
                b = constant(np.zeros(weights.shape[0], dtype=np.float32), tag + "_fcb")
            nodes.append(helper.make_node("Gemm", [values[inputs[0]], w, b], [tag + "_gemm"], transB=1))
            fused = options(operator, FullyConnectedOptions).FusedActivationFunction()
            values[output] = requantized(activation(tag + "_gemm", fused, tag), output, tag)

        elif kind == "SOFTMAX":
            nodes.append(helper.make_node("Softmax", [values[inputs[0]]], [tag + "_softmax"], axis=1))
            values[output] = requantized(tag + "_softmax", output, tag)

        else:
            raise ValueError(f"unsupported TFLite operator {kind}")

    # The final requantization writes the graph output itself, so compiled plans
    # name their output "output".
    output_index = graph.Outputs(0)
    final = values[output_index]
    if any(final in node.input for node in nodes):
        raise ValueError("the model output must not feed another operator")
    producer = next(node for node in nodes if final in node.output)
    producer.output[list(producer.output).index(final)] = "output"
    graph_output = helper.make_tensor_value_info("output", TensorProto.FLOAT, shape(output_index))
    onnx_model = helper.make_model(
        helper.make_graph(nodes, name, [graph_input], [graph_output], initializers),
        opset_imports=[helper.make_opsetid("", 17)])
    onnx_model.ir_version = 8
    onnx.checker.check_model(onnx_model)
    return onnx_model
