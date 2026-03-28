# coding: utf-8
"""
基于地平线 HBRuntime 的音频分离推理脚本（纯 CPU，不使用 GPU）。

支持的模型文件类型：.onnx / .bc / .hbm
支持的输入文件类型：图片 / 音频 / numpy / raw binary

核心流程（音频场景）：
    1. 加载音频 -> 分块
    2. 每个分块做 STFT 预处理（需要 --config_path 和 --model_type）
    3. 送入 HBRuntime 推理：sess.run(None, {input_name: data})
    4. ISTFT 后处理还原为音频波形
    5. 拼接并保存为 wav

用法示例：

  音频分离（mel_band_roformer + ONNX）：

    python3 hbruntime_inference.py \
        --model model.onnx \
        --model_type mel_band_roformer \
        --config_path configs/config_vocals_mel_band_roformer.yaml \
        --input song.wav \
        --output output/

  音频分离（bs_roformer + HBIR 量化模型）：

    python3 hbruntime_inference.py \
        --model model.bc \
        --model_type bs_roformer \
        --config_path configs/config_vocals_bs_roformer.yaml \
        --input song.wav \
        --output output/

  通用推理（不需要 STFT，直接喂 npy 数据）：

    python3 hbruntime_inference.py \
        --model model.onnx \
        --input data.npy \
        --output result.npy

  单行写法：

    python3 hbruntime_inference.py --model model.onnx --model_type mel_band_roformer --config_path configs/config_vocals_mel_band_roformer.yaml --input song.wav --output output/
"""

import argparse
import os
import sys
import time
import numpy as np
from typing import List, Tuple, Optional, Dict

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# 将项目根目录加入 sys.path，以便 import models.preprocess
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Input loaders
# ---------------------------------------------------------------------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".wma"}
NUMPY_EXTS = {".npy", ".npz"}
RAW_EXTS = {".bin", ".raw", ".dat"}


def load_image(path: str, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    from PIL import Image
    img = Image.open(path).convert("RGB")
    if target_size is not None:
        img = img.resize(target_size, Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    arr = np.expand_dims(arr, axis=0)
    return arr


def load_audio(path: str, sample_rate: int = 44100, mono: bool = False) -> np.ndarray:
    import librosa
    audio, sr = librosa.load(path, sr=sample_rate, mono=mono)
    if audio.ndim == 1:
        audio = np.expand_dims(audio, axis=0)
    return audio.astype(np.float32)


def load_numpy(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        data = np.load(path)
        key = list(data.keys())[0]
        print(f"[INFO] 从 npz 中加载 key='{key}'")
        return data[key].astype(np.float32)
    return np.load(path).astype(np.float32)


def load_raw_binary(path: str, shape: Tuple[int, ...], dtype: str = "float32") -> np.ndarray:
    arr = np.fromfile(path, dtype=np.dtype(dtype))
    arr = arr.reshape(shape)
    return arr.astype(np.float32)


def load_input(
    path: str,
    target_size: Optional[Tuple[int, int]] = None,
    sample_rate: int = 44100,
    raw_shape: Optional[Tuple[int, ...]] = None,
    raw_dtype: str = "float32",
) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext in IMAGE_EXTS:
        print(f"[INFO] 检测到图片文件: {path}")
        return load_image(path, target_size)
    elif ext in AUDIO_EXTS:
        print(f"[INFO] 检测到音频文件: {path}")
        return load_audio(path, sample_rate=sample_rate)
    elif ext in NUMPY_EXTS:
        print(f"[INFO] 检测到 numpy 文件: {path}")
        return load_numpy(path)
    elif ext in RAW_EXTS:
        if raw_shape is None:
            raise ValueError(f"加载 {path} 需要 --input_shape，例如 --input_shape 1,3,224,224")
        print(f"[INFO] 检测到二进制文件: {path}")
        return load_raw_binary(path, shape=raw_shape, dtype=raw_dtype)
    else:
        raise ValueError(f"不支持的文件类型: {ext}")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str):
    import yaml
    from ml_collections import ConfigDict
    with open(config_path, 'r') as f:
        config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))
    return config


def create_preprocessor(model_type: str, config):
    from models.preprocess import (
        STFT, BS_roformer_processor, Mel_band_roformer_processor
    )
    if model_type == 'mel_band_roformer':
        return Mel_band_roformer_processor(**dict(config.model))
    elif model_type == 'bs_roformer':
        return BS_roformer_processor(**dict(config.model))
    else:
        return STFT(config.audio)


# ---------------------------------------------------------------------------
# Model info printing
# ---------------------------------------------------------------------------

def print_model_info(model_path: str) -> None:
    ext = os.path.splitext(model_path)[1].lower()
    file_size = os.path.getsize(model_path)

    MODEL_TYPE_MAP = {
        ".onnx": "ONNX 浮点模型",
        ".bc": "地平线 HBIR 量化模型",
        ".hbm": "地平线 HBM 板端部署模型",
    }

    print("=" * 70)
    print("[MODEL INFO] 模型基本信息")
    print("=" * 70)
    print(f"  文件路径    : {os.path.abspath(model_path)}")
    print(f"  文件名      : {os.path.basename(model_path)}")
    print(f"  文件格式    : {ext}")
    print(f"  文件大小    : {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
    print(f"  模型类型    : {MODEL_TYPE_MAP.get(ext, f'未知 ({ext})')}")

    if ext == ".onnx":
        _print_onnx_info(model_path)
    print("=" * 70)


def _print_onnx_info(model_path: str) -> None:
    try:
        import onnx
    except ImportError:
        print("  [WARN] 未安装 onnx 库，跳过解析")
        return

    model = onnx.load(model_path)
    graph = model.graph

    print(f"  IR 版本     : {model.ir_version}")
    opset_list = [f"{o.domain or 'ai.onnx'}:{o.version}" for o in model.opset_import]
    print(f"  Opset 版本  : {opset_list}")
    if model.producer_name:
        print(f"  生产者      : {model.producer_name} {model.producer_version or ''}")
    if graph.name:
        print(f"  图名称      : {graph.name}")

    op_counts = {}
    for node in graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
    print(f"  节点总数    : {len(graph.node)}")
    print(f"  算子种类    : {len(op_counts)}")

    initializer_names = {init.name for init in graph.initializer}

    print("-" * 70)
    print("[MODEL INFO] 模型输入")
    print("-" * 70)
    for inp in graph.input:
        if inp.name in initializer_names:
            continue
        print(f"  名称: {inp.name}  shape: {_get_tensor_shape(inp)}  dtype: {_get_tensor_dtype(inp)}")

    print("-" * 70)
    print("[MODEL INFO] 模型输出")
    print("-" * 70)
    for out in graph.output:
        print(f"  名称: {out.name}  shape: {_get_tensor_shape(out)}  dtype: {_get_tensor_dtype(out)}")

    total_params = sum(
        np.prod(list(init.dims)) for init in graph.initializer
    )
    if total_params > 0:
        print("-" * 70)
        print(f"[MODEL INFO] 参数: {len(graph.initializer)} 个权重张量, 共 {total_params:,} ({total_params / 1e6:.2f}M)")

    print("-" * 70)
    print("[MODEL INFO] 算子分布（按数量降序）")
    print("-" * 70)
    for op, count in sorted(op_counts.items(), key=lambda x: -x[1]):
        print(f"  {op:<25s} : {count:>5d}")


def _get_tensor_shape(tensor_info) -> str:
    try:
        shape = []
        for dim in tensor_info.type.tensor_type.shape.dim:
            if dim.dim_param:
                shape.append(dim.dim_param)
            elif dim.dim_value > 0:
                shape.append(str(dim.dim_value))
            else:
                shape.append("?")
        return f"[{', '.join(shape)}]" if shape else "[unknown]"
    except Exception:
        return "[unknown]"


def _get_tensor_dtype(tensor_info) -> str:
    DTYPE_MAP = {
        0: "UNDEFINED", 1: "FLOAT", 2: "UINT8", 3: "INT8",
        4: "UINT16", 5: "INT16", 6: "INT32", 7: "INT64",
        8: "STRING", 9: "BOOL", 10: "FLOAT16", 11: "DOUBLE",
        12: "UINT32", 13: "UINT64", 14: "COMPLEX64", 15: "COMPLEX128",
        16: "BFLOAT16",
    }
    try:
        elem_type = tensor_info.type.tensor_type.elem_type
        return DTYPE_MAP.get(elem_type, f"UNKNOWN({elem_type})")
    except Exception:
        return "UNKNOWN"


def print_session_info(sess) -> None:
    print("-" * 70)
    print("[HBRuntime] 模型运行时属性")
    print("-" * 70)
    attrs = {
        "input_num": "输入数量", "output_num": "输出数量",
        "input_names": "输入名称", "output_names": "输出名称",
        "input_types": "输入类型", "output_types": "输出类型",
        "input_shapes": "输入shape", "ouput_shapes": "输出shape",
    }
    for attr, label in attrs.items():
        try:
            print(f"  {label:<12s}: {getattr(sess, attr)}")
        except AttributeError:
            pass
    if not hasattr(sess, "ouput_shapes") and hasattr(sess, "output_shapes"):
        try:
            print(f"  {'输出shape':<12s}: {sess.output_shapes}")
        except Exception:
            pass
    print("-" * 70)


# ---------------------------------------------------------------------------
# Windowing (from utils.py)
# ---------------------------------------------------------------------------

def _get_windowing_array(chunk_size: int, fade_size: int):
    import torch
    fadein = torch.linspace(0, 1, fade_size)
    fadeout = torch.linspace(1, 0, fade_size)
    window = torch.ones(chunk_size)
    window[:fade_size] = fadein
    window[-fade_size:] = fadeout
    return window


# ---------------------------------------------------------------------------
# Audio source separation with STFT
# ---------------------------------------------------------------------------

def demix_with_hbruntime(
    sess,
    config,
    model_type: str,
    mix: np.ndarray,
    preprocessor,
) -> Dict[str, np.ndarray]:
    """
    完整的音频分离流程：分块 -> STFT -> HBRuntime 推理 -> ISTFT -> 拼接。
    参考 utils.py 中的 demix 函数。
    """
    import torch
    import torch.nn as nn

    device = "cpu"
    mix = torch.tensor(mix, dtype=torch.float32)

    chunk_size = config.audio.chunk_size
    num_overlap = config.inference.num_overlap
    batch_size = config.inference.batch_size
    fade_size = chunk_size // 10
    step = chunk_size // num_overlap
    border = chunk_size - step
    length_init = mix.shape[-1]

    if getattr(config.training, 'target_instrument', None):
        instruments = [config.training.target_instrument]
    else:
        instruments = list(config.training.instruments)
    num_instruments = len(instruments)

    windowing_array = _get_windowing_array(chunk_size, fade_size)

    if length_init > 2 * border and border > 0:
        mix = nn.functional.pad(mix, (border, border), mode="reflect")

    input_names = sess.input_names

    with torch.inference_mode():
        req_shape = (num_instruments,) + mix.shape
        result = torch.zeros(req_shape, dtype=torch.float32)
        counter = torch.zeros(req_shape, dtype=torch.float32)

        i = 0
        batch_data = []
        batch_locations = []
        total_chunks = 0

        while i < mix.shape[1]:
            part = mix[:, i:i + chunk_size]
            chunk_len = part.shape[-1]
            if chunk_len > chunk_size // 2:
                pad_mode = "reflect"
            else:
                pad_mode = "constant"
            part = nn.functional.pad(part, (0, chunk_size - chunk_len), mode=pad_mode, value=0)

            batch_data.append(part)
            batch_locations.append((i, chunk_len))
            i += step

            if len(batch_data) >= batch_size or i >= mix.shape[1]:
                arr = torch.stack(batch_data, dim=0)

                # STFT 预处理
                stft_input = preprocessor.stft(arr)
                stft_np = stft_input.cpu().numpy()

                # HBRuntime 推理
                input_feed = {input_names[0]: stft_np}
                output = sess.run(None, input_feed)

                # ISTFT 后处理
                output_tensor = torch.tensor(output[0])
                x = preprocessor.istft(output_tensor)

                # 窗口叠加
                window = windowing_array.clone()
                if i - step == 0:
                    window[:fade_size] = 1
                elif i >= mix.shape[1]:
                    window[-fade_size:] = 1

                for j, (start, seg_len) in enumerate(batch_locations):
                    result[..., start:start + seg_len] += x[j, ..., :seg_len] * window[..., :seg_len]
                    counter[..., start:start + seg_len] += window[..., :seg_len]

                total_chunks += len(batch_data)
                batch_data.clear()
                batch_locations.clear()

                print(f"\r[INFO] 已处理 {total_chunks} 个分块...", end="", flush=True)

        print()

        estimated_sources = result / counter
        estimated_sources = estimated_sources.cpu().numpy()
        np.nan_to_num(estimated_sources, copy=False, nan=0.0)

        if length_init > 2 * border and border > 0:
            estimated_sources = estimated_sources[..., border:-border]

    return {k: v for k, v in zip(instruments, estimated_sources)}


# ---------------------------------------------------------------------------
# Generic inference (no STFT)
# ---------------------------------------------------------------------------

def run_generic_inference(
    model_path: str,
    input_data: np.ndarray,
    print_info: bool = True,
) -> List[np.ndarray]:
    from horizon_tc_ui.hb_runtime import HBRuntime

    if print_info:
        print_model_info(model_path)

    print(f"\n[INFO] 加载模型: {model_path}")
    t0 = time.time()
    sess = HBRuntime(model_path)
    print(f"[INFO] 模型加载耗时: {time.time() - t0:.3f}s")

    if print_info:
        print_session_info(sess)

    input_names = sess.input_names
    input_feed = {input_names[0]: input_data}

    print(f"[INFO] 输入 '{input_names[0]}': shape={input_data.shape}, dtype={input_data.dtype}")

    t0 = time.time()
    outputs = sess.run(None, input_feed)
    print(f"[INFO] 推理耗时: {time.time() - t0:.3f}s")

    result_list = []
    if isinstance(outputs, (list, tuple)):
        for i, out in enumerate(outputs):
            arr = np.array(out) if not isinstance(out, np.ndarray) else out
            print(f"  输出[{i}]: shape={arr.shape}, dtype={arr.dtype}, "
                  f"min={arr.min():.6f}, max={arr.max():.6f}")
            result_list.append(arr)
    else:
        arr = np.array(outputs) if not isinstance(outputs, np.ndarray) else outputs
        print(f"  输出: shape={arr.shape}, dtype={arr.dtype}")
        result_list.append(arr)

    return result_list


# ---------------------------------------------------------------------------
# Output saving
# ---------------------------------------------------------------------------

def save_audio_results(
    results: Dict[str, np.ndarray],
    output_dir: str,
    file_name: str,
    sample_rate: int,
) -> None:
    import soundfile as sf
    os.makedirs(output_dir, exist_ok=True)
    for instr, audio in results.items():
        out_path = os.path.join(output_dir, f"{file_name}_{instr}.wav")
        sf.write(out_path, audio.T, sample_rate, subtype='FLOAT')
        print(f"[INFO] 已保存: {out_path}  shape={audio.shape}")


def save_numpy_output(outputs: List[np.ndarray], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    ext = os.path.splitext(output_path)[1].lower()
    if ext == ".npz":
        np.savez(output_path, **{f"output_{i}": o for i, o in enumerate(outputs)})
    elif ext == ".bin":
        for i, o in enumerate(outputs):
            p = output_path if len(outputs) == 1 else output_path.replace(".bin", f"_{i}.bin")
            o.tofile(p)
            print(f"[INFO] 已保存: {p}")
        return
    else:
        for i, o in enumerate(outputs):
            p = output_path if len(outputs) == 1 else output_path.replace(".npy", f"_{i}.npy")
            np.save(p, o)
            print(f"[INFO] 已保存: {p}")
        return
    print(f"[INFO] 已保存: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_shape(s: str) -> Tuple[int, ...]:
    return tuple(int(x.strip()) for x in s.split(","))


def parse_args():
    parser = argparse.ArgumentParser(
        description="基于地平线 HBRuntime 的推理脚本（纯 CPU）"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="模型文件路径（.onnx / .bc / .hbm）")
    parser.add_argument("--model_type", type=str, default=None,
                        help="模型类型：mel_band_roformer / bs_roformer / mdx23c 等。"
                             "提供后会自动做 STFT 预处理和 ISTFT 后处理")
    parser.add_argument("--config_path", type=str, default=None,
                        help="模型配置文件路径（.yaml），与 --model_type 配合使用")
    parser.add_argument("--input", type=str, required=True,
                        help="输入文件路径")
    parser.add_argument("--output", type=str, default="output",
                        help="输出路径：音频分离模式为目录，通用模式为文件")
    parser.add_argument("--input_shape", type=str, default=None,
                        help="输入 shape（仅 .bin 文件需要），格式如 1,3,224,224")
    parser.add_argument("--input_dtype", type=str, default="float32",
                        help="输入数据类型（仅 .bin 文件需要）")
    parser.add_argument("--image_size", type=str, default=None,
                        help="图片 resize 尺寸，格式如 224,224")
    parser.add_argument("--sample_rate", type=int, default=44100,
                        help="音频采样率，默认 44100")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.model):
        print(f"[ERROR] 模型文件不存在: {args.model}")
        sys.exit(1)
    if not os.path.isfile(args.input):
        print(f"[ERROR] 输入文件不存在: {args.input}")
        sys.exit(1)

    model_ext = os.path.splitext(args.model)[1].lower()
    if model_ext not in {".onnx", ".bc", ".hbm"}:
        print(f"[ERROR] 不支持的模型格式: {model_ext}")
        sys.exit(1)

    try:
        from horizon_tc_ui.hb_runtime import HBRuntime
    except ImportError:
        print("[ERROR] 无法导入 HBRuntime。\n"
              "正确导入: from horizon_tc_ui.hb_runtime import HBRuntime\n"
              "请确认已安装 horizon_tc_ui（地平线 OE 工具链）。")
        sys.exit(1)

    # ---- 音频分离模式（有 config_path 和 model_type） ----
    if args.config_path and args.model_type:
        if not os.path.isfile(args.config_path):
            print(f"[ERROR] 配置文件不存在: {args.config_path}")
            sys.exit(1)

        print(f"[INFO] 音频分离模式: model_type={args.model_type}")
        config = load_config(args.config_path)
        sample_rate = getattr(config.audio, 'sample_rate', args.sample_rate)

        print_model_info(args.model)

        print(f"\n[INFO] 加载模型...")
        t0 = time.time()
        sess = HBRuntime(args.model)
        print(f"[INFO] 模型加载耗时: {time.time() - t0:.3f}s")
        print_session_info(sess)

        print(f"[INFO] 创建 STFT 预处理器: {args.model_type}")
        preprocessor = create_preprocessor(args.model_type, config)

        print(f"[INFO] 加载音频: {args.input}")
        mix = load_audio(args.input, sample_rate=sample_rate)
        print(f"[INFO] 音频 shape={mix.shape}, 采样率={sample_rate}, "
              f"时长={mix.shape[-1] / sample_rate:.2f}s")

        print(f"[INFO] 开始分离推理...")
        total_start = time.time()
        results = demix_with_hbruntime(sess, config, args.model_type, mix, preprocessor)
        print(f"[INFO] 推理总耗时: {time.time() - total_start:.2f}s")

        file_name = os.path.splitext(os.path.basename(args.input))[0]
        save_audio_results(results, args.output, file_name, sample_rate)

    # ---- 通用模式（直接推理，无 STFT） ----
    else:
        if args.config_path or args.model_type:
            print("[WARN] --config_path 和 --model_type 需要同时提供才能启用 STFT 预处理，"
                  "当前进入通用推理模式")

        target_size = None
        if args.image_size:
            parts = args.image_size.split(",")
            target_size = (int(parts[0].strip()), int(parts[1].strip()))

        raw_shape = parse_shape(args.input_shape) if args.input_shape else None

        input_data = load_input(
            path=args.input,
            target_size=target_size,
            sample_rate=args.sample_rate,
            raw_shape=raw_shape,
            raw_dtype=args.input_dtype,
        )

        outputs = run_generic_inference(args.model, input_data)
        save_numpy_output(outputs, args.output)

    print("[INFO] 完成！")


if __name__ == "__main__":
    main()
