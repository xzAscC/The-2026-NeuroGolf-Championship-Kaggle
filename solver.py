#!/usr/bin/env python3
"""NeuroGolf 2026 Championship - ARC-AGI solver using minimal ONNX networks."""

import argparse
import json
import os
import time
import zipfile

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import onnxruntime as ort

# ── Constants ────────────────────────────────────────────────────────

BATCH = 1
CH = 10
GH = 30
GW = 30
GRID_SHAPE = [BATCH, CH, GH, GW]
DT = TensorProto.FLOAT
IR_VERSION = 10
OPSET = [helper.make_opsetid("", 11)]

ANALYTICAL_SOLVERS = [
    "identity",
    "constant",
    "color_map",
    "transpose",
    "flip",
    "rotate",
    "tile",
    "upscale",
    "concat",
    "spatial_gather",
    "crop",
]

# ── Core Utilities ───────────────────────────────────────────────────


def load_tasks_json(data_file):
    with open(data_file) as f:
        raw = json.load(f)
    tasks = {}
    for idx, key in enumerate(sorted(raw.keys())):
        tasks[idx] = {"hex": key, "data": raw[key]}
    return tasks


def to_onehot(grid):
    arr = np.zeros([1, CH, GH, GW], dtype=np.float32)
    for r, row in enumerate(grid):
        for c, v in enumerate(row):
            if 0 <= int(v) < CH:
                arr[0, int(v), r, c] = 1.0
    return arr


def validate(onnx_path, task_data):
    try:
        sess = ort.InferenceSession(onnx_path)
        inp_name = sess.get_inputs()[0].name
        for pair in task_data["train"] + task_data["test"]:
            inp = to_onehot(pair["input"])
            expected = to_onehot(pair["output"])
            out = sess.run(None, {inp_name: inp})[0]
            out = (out > 0.0).astype(np.float32)
            if not np.array_equal(out, expected):
                return False
        return True
    except Exception:
        return False


def mk(nodes, inits=None):
    inp_info = helper.make_tensor_value_info("input", DT, GRID_SHAPE)
    out_info = helper.make_tensor_value_info("output", DT, GRID_SHAPE)
    graph = helper.make_graph(nodes, "g", [inp_info], [out_info], inits or [])
    model = helper.make_model(graph, opset_imports=OPSET)
    model.ir_version = IR_VERSION
    return model


def get_exs(task_data):
    exs = []
    for pair in task_data["train"] + task_data["test"]:
        inp = np.array(pair["input"], dtype=np.int64)
        out = np.array(pair["output"], dtype=np.int64)
        exs.append((inp, out))
    return exs


def fixed_shapes(task_data):
    exs = get_exs(task_data)
    in_shapes = {e[0].shape for e in exs}
    out_shapes = {e[1].shape for e in exs}
    if len(in_shapes) == 1 and len(out_shapes) == 1:
        return (list(in_shapes)[0], list(out_shapes)[0])
    return None


# ── GatherElements helper ────────────────────────────────────────────


def _build_gather(idx_2d, mask_2d, const_oh=None):
    """Build GatherElements spatial remap model.

    idx_2d  : [GH, GW] int array mapping output (r,c) -> input flat index
    mask_2d : [GH, GW] float, 1.0 = valid gather, 0.0 = masked out
    const_oh: [1,CH,GH,GW] float constants added at masked positions (optional)
    """
    flat_idx = idx_2d.flatten().astype(np.int64)
    flat_mask = mask_2d.flatten().astype(np.float32)
    idx_t = np.tile(flat_idx.reshape(1, 1, 900), (1, CH, 1))
    mask_t = flat_mask.reshape(1, 1, 900)

    inits = [
        numpy_helper.from_array(idx_t, name="g_idx"),
        numpy_helper.from_array(mask_t, name="g_mask"),
        numpy_helper.from_array(np.array([1, CH, 900], dtype=np.int64), name="g_sf"),
        numpy_helper.from_array(np.array(GRID_SHAPE, dtype=np.int64), name="g_sg"),
    ]
    nodes = [
        helper.make_node("Reshape", ["input", "g_sf"], ["g_f"]),
        helper.make_node("GatherElements", ["g_f", "g_idx"], ["g_ga"], axis=2),
        helper.make_node("Mul", ["g_ga", "g_mask"], ["g_mu"]),
    ]
    if const_oh is not None:
        cf = const_oh.reshape(1, CH, 900).astype(np.float32)
        inits.append(numpy_helper.from_array(cf, name="g_const"))
        nodes.append(helper.make_node("Add", ["g_mu", "g_const"], ["g_out"]))
    else:
        nodes.append(helper.make_node("Identity", ["g_mu"], ["g_out"]))
    nodes.append(helper.make_node("Reshape", ["g_out", "g_sg"], ["output"]))
    return mk(nodes, inits)


# ── Analytical Solvers ───────────────────────────────────────────────


def s_identity(td):
    for inp, out in get_exs(td):
        if not np.array_equal(inp, out):
            return None
    return mk([helper.make_node("Identity", ["input"], ["output"])])


def s_constant(td):
    outputs = [pair["output"] for pair in td["train"] + td["test"]]
    if not outputs:
        return None
    ref = json.dumps(outputs[0], sort_keys=True)
    for o in outputs[1:]:
        if json.dumps(o, sort_keys=True) != ref:
            return None
    const_oh = to_onehot(outputs[0])
    inits = [
        numpy_helper.from_array(const_oh, name="c_val"),
        numpy_helper.from_array(np.array([0.0], dtype=np.float32), name="c_zero"),
    ]
    nodes = [
        helper.make_node("Mul", ["input", "c_zero"], ["c_z"]),
        helper.make_node("Add", ["c_z", "c_val"], ["output"]),
    ]
    return mk(nodes, inits)


def s_color_map(td):
    exs = get_exs(td)
    for inp, out in exs:
        if inp.shape != out.shape:
            return None
    cmap = {}
    for inp, out in exs:
        for r in range(inp.shape[0]):
            for c in range(inp.shape[1]):
                v = int(inp[r, c])
                e = int(out[r, c])
                if v in cmap:
                    if cmap[v] != e:
                        return None
                else:
                    cmap[v] = e
    W = np.zeros([CH, CH, 1, 1], dtype=np.float32)
    for src, dst in cmap.items():
        W[dst, src, 0, 0] = 1.0
    inits = [numpy_helper.from_array(W, name="cm_w")]
    nodes = [
        helper.make_node(
            "Conv",
            ["input", "cm_w"],
            ["output"],
            kernel_shape=[1, 1],
            strides=[1, 1],
            pads=[0, 0, 0, 0],
        )
    ]
    return mk(nodes, inits)


def s_transpose(td):
    for inp, out in get_exs(td):
        if not np.array_equal(out, inp.T):
            return None
    return mk([helper.make_node("Transpose", ["input"], ["output"], perm=[0, 1, 3, 2])])


def s_flip(td):
    exs = get_exs(td)
    # Horizontal flip
    ok = True
    for inp, out in exs:
        H, W = inp.shape
        for r in range(H):
            for c in range(W):
                if out[r, c] != inp[r, W - 1 - c]:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break
    if ok:
        shapes = {e[0].shape for e in exs}
        if len(shapes) == 1:
            H, W = list(shapes)[0]
            idx = np.zeros([GH, GW], dtype=np.int64)
            mask = np.zeros([GH, GW], dtype=np.float32)
            for r in range(GH):
                for c in range(GW):
                    if r < H and c < W:
                        idx[r, c] = r * GW + (W - 1 - c)
                        mask[r, c] = 1.0
                    else:
                        idx[r, c] = r * GW + c
            return _build_gather(idx, mask)

    # Vertical flip
    ok = True
    for inp, out in exs:
        H, W = inp.shape
        for r in range(H):
            for c in range(W):
                if out[r, c] != inp[H - 1 - r, c]:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break
    if ok:
        shapes = {e[0].shape for e in exs}
        if len(shapes) == 1:
            H, W = list(shapes)[0]
            idx = np.zeros([GH, GW], dtype=np.int64)
            mask = np.zeros([GH, GW], dtype=np.float32)
            for r in range(GH):
                for c in range(GW):
                    if r < H and c < W:
                        idx[r, c] = (H - 1 - r) * GW + c
                        mask[r, c] = 1.0
                    else:
                        idx[r, c] = r * GW + c
            return _build_gather(idx, mask)
    return None


def s_rotate(td):
    exs = get_exs(td)
    for angle in [90, 180, 270]:
        ok = True
        for inp, out in exs:
            Hi, Wi = inp.shape
            Ho, Wo = out.shape
            for r in range(Ho):
                for c in range(Wo):
                    if angle == 90:
                        sr, sc = Hi - 1 - c, r
                    elif angle == 180:
                        sr, sc = Hi - 1 - r, Wi - 1 - c
                    else:
                        sr, sc = c, Wi - 1 - r
                    if not (0 <= sr < Hi and 0 <= sc < Wi):
                        ok = False
                        break
                    if out[r, c] != inp[sr, sc]:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            in_shapes = {e[0].shape for e in exs}
            out_shapes = {e[1].shape for e in exs}
            if len(in_shapes) == 1 and len(out_shapes) == 1:
                Hi, Wi = list(in_shapes)[0]
                Ho, Wo = list(out_shapes)[0]
                idx = np.zeros([GH, GW], dtype=np.int64)
                mask = np.zeros([GH, GW], dtype=np.float32)
                for r in range(GH):
                    for c in range(GW):
                        if r < Ho and c < Wo:
                            if angle == 90:
                                sr, sc = Hi - 1 - c, r
                            elif angle == 180:
                                sr, sc = Hi - 1 - r, Wi - 1 - c
                            else:
                                sr, sc = c, Wi - 1 - r
                            idx[r, c] = sr * GW + sc
                            mask[r, c] = 1.0
                        else:
                            idx[r, c] = r * GW + c
                return _build_gather(idx, mask)
    return None


def s_tile(td):
    exs = get_exs(td)
    shapes_in = {e[0].shape for e in exs}
    shapes_out = {e[1].shape for e in exs}
    if len(shapes_in) != 1 or len(shapes_out) != 1:
        return None
    Hi, Wi = list(shapes_in)[0]
    Ho, Wo = list(shapes_out)[0]
    if Ho % Hi != 0 or Wo % Wi != 0:
        return None
    th, tw = Ho // Hi, Wo // Wi
    if th < 2 and tw < 2:
        return None
    for inp, out in exs:
        for r in range(Ho):
            for c in range(Wo):
                if out[r, c] != inp[r % Hi, c % Wi]:
                    return None
    # Build Slice → Tile → Pad
    inits = [
        numpy_helper.from_array(
            np.array([0, 0, 0, 0], dtype=np.int64), name="tl_starts"
        ),
        numpy_helper.from_array(
            np.array([1, CH, Hi, Wi], dtype=np.int64), name="tl_ends"
        ),
        numpy_helper.from_array(
            np.array([1, 1, th, tw], dtype=np.int64), name="tl_reps"
        ),
        numpy_helper.from_array(
            np.array([0, 0, 0, 0, 0, 0, GH - Ho, GW - Wo], dtype=np.int64),
            name="tl_pads",
        ),
    ]
    nodes = [
        helper.make_node("Slice", ["input", "tl_starts", "tl_ends"], ["tl_sl"]),
        helper.make_node("Tile", ["tl_sl", "tl_reps"], ["tl_ti"]),
        helper.make_node("Pad", ["tl_ti", "tl_pads"], ["output"], mode="constant"),
    ]
    return mk(nodes, inits)


def s_upscale(td):
    exs = get_exs(td)
    in_shapes = {e[0].shape for e in exs}
    out_shapes = {e[1].shape for e in exs}
    if len(in_shapes) != 1 or len(out_shapes) != 1:
        return None
    Hi, Wi = list(in_shapes)[0]
    Ho, Wo = list(out_shapes)[0]
    if Hi == 0 or Wi == 0:
        return None
    if Ho % Hi != 0 or Wo % Wi != 0:
        return None
    sh, sw = Ho // Hi, Wo // Wi
    if sh != sw or sh < 2:
        return None
    for inp, out in exs:
        for r in range(Ho):
            for c in range(Wo):
                if out[r, c] != inp[r // sh, c // sw]:
                    return None
    # Build gather model with out[r,c] -> in[r//s, c//s]
    idx = np.zeros([GH, GW], dtype=np.int64)
    mask = np.zeros([GH, GW], dtype=np.float32)
    for r in range(GH):
        for c in range(GW):
            if r < Ho and c < Wo:
                ir, ic = r // sh, c // sw
                idx[r, c] = ir * GW + ic
                mask[r, c] = 1.0
            else:
                idx[r, c] = r * GW + c
    return _build_gather(idx, mask)


def _transform_grid(grid, transform):
    """Apply a named transform to a 2D grid."""
    if transform == "identity":
        return grid
    if transform == "fliplr":
        return np.fliplr(grid)
    if transform == "flipud":
        return np.flipud(grid)
    if transform == "rot180":
        return np.rot90(grid, 2)
    return None


_TRANSFORMS = ["identity", "fliplr", "flipud", "rot180"]


def s_concat(td):
    exs = get_exs(td)
    in_shapes = {e[0].shape for e in exs}
    out_shapes = {e[1].shape for e in exs}
    if len(in_shapes) != 1 or len(out_shapes) != 1:
        return None
    Hi, Wi = list(in_shapes)[0]
    Ho, Wo = list(out_shapes)[0]
    if Hi == 0 or Wi == 0:
        return None

    # Try horizontal 2-tile: [f1(input) | f2(input)]
    if Wo == 2 * Wi and Ho == Hi:
        for f1 in _TRANSFORMS:
            for f2 in _TRANSFORMS:
                ok = True
                for inp, out in exs:
                    t1 = _transform_grid(inp, f1)
                    t2 = _transform_grid(inp, f2)
                    for r in range(Hi):
                        for c in range(Wi):
                            if out[r, c] != t1[r, c]:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        for r in range(Hi):
                            for c in range(Wi, 2 * Wi):
                                if out[r, c] != t2[r, c - Wi]:
                                    ok = False
                                    break
                            if not ok:
                                break
                    if not ok:
                        break
                if ok:
                    return _build_concat_model_h2(Hi, Wi, f1, f2)

    # Try vertical 2-tile
    if Ho == 2 * Hi and Wo == Wi:
        for f1 in _TRANSFORMS:
            for f2 in _TRANSFORMS:
                ok = True
                for inp, out in exs:
                    t1 = _transform_grid(inp, f1)
                    t2 = _transform_grid(inp, f2)
                    for r in range(Hi):
                        for c in range(Wi):
                            if out[r, c] != t1[r, c]:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        for r in range(Hi, 2 * Hi):
                            for c in range(Wi):
                                if out[r, c] != t2[r - Hi, c]:
                                    ok = False
                                    break
                            if not ok:
                                break
                    if not ok:
                        break
                if ok:
                    return _build_concat_model_v2(Hi, Wi, f1, f2)

    # Try 2x2 tile
    if Ho == 2 * Hi and Wo == 2 * Wi:
        for f1 in _TRANSFORMS:
            for f2 in _TRANSFORMS:
                for f3 in _TRANSFORMS:
                    for f4 in _TRANSFORMS:
                        ok = True
                        for inp, out in exs:
                            grids = [_transform_grid(inp, f) for f in [f1, f2, f3, f4]]
                            for r in range(Ho):
                                for c in range(Wo):
                                    br = 0 if r < Hi else 1
                                    bc = 0 if c < Wi else 1
                                    bi = br * 2 + bc
                                    if out[r, c] != grids[bi][r % Hi, c % Wi]:
                                        ok = False
                                        break
                                if not ok:
                                    break
                            if not ok:
                                break
                        if ok:
                            return _build_concat_model_2x2(Hi, Wi, [f1, f2, f3, f4])
    return None


def _flip_idx_h(H, W):
    """Horizontal flip index map for grid HxW in 30x30."""
    idx = np.zeros([GH, GW], dtype=np.int64)
    for r in range(GH):
        for c in range(GW):
            if r < H and c < W:
                idx[r, c] = r * GW + (W - 1 - c)
            else:
                idx[r, c] = r * GW + c
    return idx


def _flip_idx_v(H, W):
    idx = np.zeros([GH, GW], dtype=np.int64)
    for r in range(GH):
        for c in range(GW):
            if r < H and c < W:
                idx[r, c] = (H - 1 - r) * GW + c
            else:
                idx[r, c] = r * GW + c
    return idx


def _rot180_idx(H, W):
    idx = np.zeros([GH, GW], dtype=np.int64)
    for r in range(GH):
        for c in range(GW):
            if r < H and c < W:
                idx[r, c] = (H - 1 - r) * GW + (W - 1 - c)
            else:
                idx[r, c] = r * GW + c
    return idx


def _identity_idx(H, W):
    idx = np.arange(900, dtype=np.int64).reshape(GH, GW)
    return idx


_TRANSFORM_IDX = {
    "identity": lambda H, W: _identity_idx(H, W),
    "fliplr": _flip_idx_h,
    "flipud": _flip_idx_v,
    "rot180": _rot180_idx,
}


def _build_concat_model_h2(H, W, f1, f2):
    """Horizontal concat of 2 transformed blocks."""
    idx = np.zeros([GH, GW], dtype=np.int64)
    mask = np.zeros([GH, GW], dtype=np.float32)
    idx1 = _TRANSFORM_IDX[f1](H, W)
    idx2 = _TRANSFORM_IDX[f2](H, W)
    for r in range(GH):
        for c in range(GW):
            if r < H:
                if c < W:
                    idx[r, c] = idx1[r, c]
                    mask[r, c] = 1.0
                elif c < 2 * W:
                    idx[r, c] = idx2[r, c - W]
                    mask[r, c] = 1.0
                else:
                    idx[r, c] = r * GW + c
            else:
                idx[r, c] = r * GW + c
    return _build_gather(idx, mask)


def _build_concat_model_v2(H, W, f1, f2):
    """Vertical concat of 2 transformed blocks."""
    idx = np.zeros([GH, GW], dtype=np.int64)
    mask = np.zeros([GH, GW], dtype=np.float32)
    idx1 = _TRANSFORM_IDX[f1](H, W)
    idx2 = _TRANSFORM_IDX[f2](H, W)
    for r in range(GH):
        for c in range(GW):
            if c < W:
                if r < H:
                    idx[r, c] = idx1[r, c]
                    mask[r, c] = 1.0
                elif r < 2 * H:
                    idx[r, c] = idx2[r - H, c]
                    mask[r, c] = 1.0
                else:
                    idx[r, c] = r * GW + c
            else:
                idx[r, c] = r * GW + c
    return _build_gather(idx, mask)


def _build_concat_model_2x2(H, W, transforms):
    """2x2 grid of transformed blocks."""
    idx = np.zeros([GH, GW], dtype=np.int64)
    mask = np.zeros([GH, GW], dtype=np.float32)
    idxs = [_TRANSFORM_IDX[f](H, W) for f in transforms]
    for r in range(GH):
        for c in range(GW):
            br = 0 if r < H else 1
            bc = 0 if c < W else 1
            bi = br * 2 + bc
            lr, lc = r % H, c % W
            if r < 2 * H and c < 2 * W:
                idx[r, c] = idxs[bi][lr, lc]
                mask[r, c] = 1.0
            else:
                idx[r, c] = r * GW + c
    return _build_gather(idx, mask)


def s_spatial_gather(td):
    exs = get_exs(td)
    out_shapes = {e[1].shape for e in exs}
    if len(out_shapes) != 1:
        return None
    Ho, Wo = list(out_shapes)[0]
    # For each output position, find source (r_in, c_in) or constant value
    src_map = {}
    is_const = {}
    for r in range(Ho):
        for c in range(Wo):
            vals = [int(ex[1][r, c]) for ex in exs]
            # Try to find consistent input source
            found_src = None
            for sr in range(GH):
                for sc in range(GW):
                    ok = True
                    for ei, (inp, _) in enumerate(exs):
                        Hi, Wi = inp.shape
                        if sr >= Hi or sc >= Wi:
                            ok = False
                            break
                        if int(inp[sr, sc]) != vals[ei]:
                            ok = False
                            break
                    if ok:
                        found_src = (sr, sc)
                        break
                if found_src is not None:
                    break
            if found_src is not None:
                src_map[(r, c)] = found_src
                is_const[(r, c)] = False
            else:
                # Check if all examples have same constant value
                if len(set(vals)) != 1:
                    return None
                src_map[(r, c)] = vals[0]
                is_const[(r, c)] = True

    idx = np.zeros([GH, GW], dtype=np.int64)
    mask = np.zeros([GH, GW], dtype=np.float32)
    const_oh = np.zeros([1, CH, GH, GW], dtype=np.float32)
    for r in range(GH):
        for c in range(GW):
            if r < Ho and c < Wo and not is_const.get((r, c), True):
                sr, sc = src_map[(r, c)]
                idx[r, c] = sr * GW + sc
                mask[r, c] = 1.0
            elif r < Ho and c < Wo and is_const.get((r, c), True):
                idx[r, c] = 0
                mask[r, c] = 0.0
                v = src_map[(r, c)]
                if isinstance(v, int):
                    const_oh[0, v, r, c] = 1.0
            else:
                idx[r, c] = 0
                mask[r, c] = 0.0
    has_const = np.any(const_oh > 0)
    return _build_gather(idx, mask, const_oh if has_const else None)


def s_crop(td):
    exs = get_exs(td)
    in_shapes = {e[0].shape for e in exs}
    out_shapes = {e[1].shape for e in exs}
    if len(in_shapes) != 1 or len(out_shapes) != 1:
        return None
    Hi, Wi = list(in_shapes)[0]
    Ho, Wo = list(out_shapes)[0]
    if Ho > Hi or Wo > Wi:
        return None
    dr = (Hi - Ho) // 2
    dc = (Wi - Wo) // 2
    for inp, out in exs:
        for r in range(Ho):
            for c in range(Wo):
                if out[r, c] != inp[r + dr, c + dc]:
                    return None
    # Build Slice → Pad
    inits = [
        numpy_helper.from_array(
            np.array([0, 0, dr, dc], dtype=np.int64), name="cr_starts"
        ),
        numpy_helper.from_array(
            np.array([1, CH, dr + Ho, dc + Wo], dtype=np.int64), name="cr_ends"
        ),
        numpy_helper.from_array(
            np.array([0, 0, 0, 0, 0, 0, GH - Ho, GW - Wo], dtype=np.int64),
            name="cr_pads",
        ),
    ]
    nodes = [
        helper.make_node("Slice", ["input", "cr_starts", "cr_ends"], ["cr_sl"]),
        helper.make_node("Pad", ["cr_sl", "cr_pads"], ["output"], mode="constant"),
    ]
    return mk(nodes, inits)


# ── Convolution Solver ───────────────────────────────────────────────


def _extract_patches(oh_input, ks, H, W):
    """Extract ks x ks one-hot patches from input, centered with zero-pad.

    oh_input : [1, CH, H, W] one-hot float32
    ks       : kernel size (odd)
    Returns  : [n_patches, CH * ks * ks] float32
    """
    pad = ks // 2
    padded = np.pad(
        oh_input,
        ((0, 0), (0, 0), (pad, pad), (pad, pad)),
        mode="constant",
        constant_values=0,
    )
    patches = []
    for r in range(H):
        for c in range(W):
            patch = padded[0, :, r : r + ks, c : c + ks]  # [CH, ks, ks]
            patches.append(patch.flatten())
    return np.array(patches, dtype=np.float32)


def _extract_patches_full30(oh_input, ks, out_h, out_w):
    """Extract patches from full 30x30 one-hot for variable-size conv."""
    pad = ks // 2
    padded = np.pad(
        oh_input,
        ((0, 0), (0, 0), (pad, pad), (pad, pad)),
        mode="constant",
        constant_values=0,
    )
    patches = []
    for r in range(out_h):
        for c in range(out_w):
            patch = padded[0, :, r : r + ks, c : c + ks]
            patches.append(patch.flatten())
    return np.array(patches, dtype=np.float32)


def _lstsq_conv(exs_raw, ks, use_bias, use_full_30):
    """Fit conv weights via least-squares on one-hot patches.

    Returns (Wconv [CH,CH,ks,ks], bias [CH] or None) or None on failure.
    """
    n_features = CH * ks * ks
    all_P = []
    all_T = []

    for inp, out in exs_raw:
        Hi, Wi = inp.shape
        Ho, Wo = out.shape
        oh_in = to_onehot(inp)  # [1, CH, 30, 30]

        if use_full_30:
            patches = _extract_patches_full30(oh_in, ks, GH, GW)
            # Build target for all 900 positions
            t_oh = np.zeros([GH * GW, CH], dtype=np.float32)
            for r in range(Ho):
                for c in range(Wo):
                    v = int(out[r, c])
                    if 0 <= v < CH:
                        t_oh[r * GW + c, v] = 1.0
        else:
            # Extract only within grid bounds
            # Slice one-hot to [1, CH, Hi, Wi]
            oh_crop = oh_in[0, :, :Hi, :Wi].reshape(1, CH, Hi, Wi)
            patches = _extract_patches(oh_crop, ks, Hi, Wi)
            t_oh = np.zeros([Hi * Wi, CH], dtype=np.float32)
            for r in range(Ho):
                for c in range(Wo):
                    v = int(out[r, c])
                    if 0 <= v < CH:
                        t_oh[r * Wi + c, v] = 1.0

        all_P.append(patches)
        all_T.append(t_oh)

    P = np.concatenate(all_P, axis=0)
    T_oh = np.concatenate(all_T, axis=0)
    n_patches = P.shape[0]
    features = n_features + (1 if use_bias else 0)

    if features > 20000 or (features > 5000 and n_patches > 2000):
        return None

    if use_bias:
        P = np.concatenate([P, np.ones((n_patches, 1), dtype=np.float32)], axis=1)

    try:
        WT, _, _, _ = np.linalg.lstsq(P, T_oh, rcond=None)
    except np.linalg.LinAlgError:
        return None

    if use_bias:
        Wconv = WT[:-1].T.reshape(CH, CH, ks, ks).astype(np.float32)
        bias = WT[-1].T.astype(np.float32)
    else:
        Wconv = WT.T.reshape(CH, CH, ks, ks).astype(np.float32)
        bias = None

    # Verify 100% accuracy on training data
    for inp, out in exs_raw:
        Hi, Wi = inp.shape
        Ho, Wo = out.shape
        oh_in = to_onehot(inp)
        if use_full_30:
            patches = _extract_patches_full30(oh_in, ks, GH, GW)
        else:
            oh_crop = oh_in[0, :, :Hi, :Wi].reshape(1, CH, Hi, Wi)
            patches = _extract_patches(oh_crop, ks, Hi, Wi)

        if use_bias:
            P_verify = np.concatenate(
                [patches, np.ones((patches.shape[0], 1), dtype=np.float32)], axis=1
            )
            pred = P_verify @ WT  # [n_pos, CH]
        else:
            pred = patches @ WT  # [n_pos, CH]
        pred_cls = np.argmax(pred, axis=1)

        if use_full_30:
            for r in range(Ho):
                for c in range(Wo):
                    if pred_cls[r * GW + c] != int(out[r, c]):
                        return None
        else:
            for r in range(Ho):
                for c in range(Wo):
                    if pred_cls[r * Wi + c] != int(out[r, c]):
                        return None
    return (Wconv, bias)


def _save_conv_model(Wconv, bias, H_in, W_in, H_out, W_out, path, use_full_30):
    """Build and save the ONNX conv pipeline model."""
    ks = Wconv.shape[2]
    pad = ks // 2

    inits = [numpy_helper.from_array(Wconv, name="cv_W")]

    if use_full_30:
        # Pipeline: Conv(30x30) → ArgMax → OneHot → Mul(mask)
        # mask = ReduceSum(input, axis=1, keepdims=1) gives [1,1,30,30]
        nodes = [
            helper.make_node(
                "Conv",
                ["input", "cv_W"],
                ["cv_c"],
                kernel_shape=[ks, ks],
                pads=[pad, pad, pad, pad],
            ),
        ]
        if bias is not None:
            inits.append(numpy_helper.from_array(bias, name="cv_B"))
            nodes.append(helper.make_node("Add", ["cv_c", "cv_B"], ["cv_cb"]))
            last_conv = "cv_cb"
        else:
            last_conv = "cv_c"

        nodes += [
            helper.make_node("ArgMax", [last_conv], ["cv_am"], axis=1, keepdims=0),
            # OneHot inputs: indices, depth, values
            helper.make_node(
                "OneHot", ["cv_am", "cv_depth", "cv_vals"], ["cv_oh"], axis=1
            ),
            # mask from input
            helper.make_node(
                "ReduceSum", ["input", "cv_rsum_ax"], ["cv_mask"], keepdims=1
            ),
            helper.make_node("Mul", ["cv_oh", "cv_mask"], ["output"]),
        ]
        inits += [
            numpy_helper.from_array(np.array(10, dtype=np.int64), name="cv_depth"),
            numpy_helper.from_array(
                np.array([0.0, 1.0], dtype=np.float32), name="cv_vals"
            ),
            numpy_helper.from_array(np.array([1], dtype=np.int64), name="cv_rsum_ax"),
        ]
    else:
        # Pipeline: Slice → Conv → ArgMax → OneHot → Pad
        inits += [
            numpy_helper.from_array(
                np.array([0, 0, 0, 0], dtype=np.int64), name="cv_starts"
            ),
            numpy_helper.from_array(
                np.array([1, CH, H_in, W_in], dtype=np.int64), name="cv_ends"
            ),
            numpy_helper.from_array(np.array(10, dtype=np.int64), name="cv_depth"),
            numpy_helper.from_array(
                np.array([0.0, 1.0], dtype=np.float32), name="cv_vals"
            ),
            numpy_helper.from_array(
                np.array([0, 0, 0, 0, 0, 0, GH - H_out, GW - W_out], dtype=np.int64),
                name="cv_pads",
            ),
        ]
        nodes = [
            helper.make_node("Slice", ["input", "cv_starts", "cv_ends"], ["cv_sl"]),
            helper.make_node(
                "Conv",
                ["cv_sl", "cv_W"],
                ["cv_c"],
                kernel_shape=[ks, ks],
                pads=[pad, pad, pad, pad],
            ),
        ]
        if bias is not None:
            inits.append(numpy_helper.from_array(bias, name="cv_B"))
            nodes.append(helper.make_node("Add", ["cv_c", "cv_B"], ["cv_cb"]))
            last_conv = "cv_cb"
        else:
            last_conv = "cv_c"

        nodes += [
            helper.make_node("ArgMax", [last_conv], ["cv_am"], axis=1, keepdims=0),
            helper.make_node(
                "OneHot", ["cv_am", "cv_depth", "cv_vals"], ["cv_oh"], axis=1
            ),
            helper.make_node("Pad", ["cv_oh", "cv_pads"], ["output"], mode="constant"),
        ]

    model = mk(nodes, inits)
    onnx.save(model, path)


def _save_conv_diffshape_model(
    Wconv, bias, H_in, W_in, H_out, W_out, crop_dr, crop_dc, ks, path
):
    """Slice → Conv → Slice(crop) → ArgMax → OneHot → Pad."""
    pad = ks // 2
    conv_h = H_in
    conv_w = W_in

    inits = [
        numpy_helper.from_array(Wconv, name="dv_W"),
        numpy_helper.from_array(
            np.array([0, 0, 0, 0], dtype=np.int64), name="dv_sl1_starts"
        ),
        numpy_helper.from_array(
            np.array([1, CH, H_in, W_in], dtype=np.int64), name="dv_sl1_ends"
        ),
        numpy_helper.from_array(
            np.array([0, 0, crop_dr, crop_dc], dtype=np.int64), name="dv_sl2_starts"
        ),
        numpy_helper.from_array(
            np.array([1, CH, crop_dr + H_out, crop_dc + W_out], dtype=np.int64),
            name="dv_sl2_ends",
        ),
        numpy_helper.from_array(np.array(10, dtype=np.int64), name="dv_depth"),
        numpy_helper.from_array(np.array([0.0, 1.0], dtype=np.float32), name="dv_vals"),
        numpy_helper.from_array(
            np.array([0, 0, 0, 0, 0, 0, GH - H_out, GW - W_out], dtype=np.int64),
            name="dv_pads",
        ),
    ]
    if bias is not None:
        inits.append(numpy_helper.from_array(bias, name="dv_B"))

    nodes = [
        helper.make_node("Slice", ["input", "dv_sl1_starts", "dv_sl1_ends"], ["dv_sl"]),
        helper.make_node(
            "Conv",
            ["dv_sl", "dv_W"],
            ["dv_c"],
            kernel_shape=[ks, ks],
            pads=[pad, pad, pad, pad],
        ),
    ]
    if bias is not None:
        nodes.append(helper.make_node("Add", ["dv_c", "dv_B"], ["dv_cb"]))
        last_conv = "dv_cb"
    else:
        last_conv = "dv_c"

    nodes += [
        helper.make_node(
            "Slice", [last_conv, "dv_sl2_starts", "dv_sl2_ends"], ["dv_cr"]
        ),
        helper.make_node("ArgMax", ["dv_cr"], ["dv_am"], axis=1, keepdims=0),
        helper.make_node("OneHot", ["dv_am", "dv_depth", "dv_vals"], ["dv_oh"], axis=1),
        helper.make_node("Pad", ["dv_oh", "dv_pads"], ["output"], mode="constant"),
    ]
    model = mk(nodes, inits)
    onnx.save(model, path)


def solve_conv_fixed(td, path, time_budget):
    """Conv solver for same-shape tasks with fixed input size."""
    exs = get_exs(td)
    in_shapes = {e[0].shape for e in exs}
    out_shapes = {e[1].shape for e in exs}
    if len(in_shapes) != 1 or len(out_shapes) != 1:
        return False
    Hi, Wi = list(in_shapes)[0]
    Ho, Wo = list(out_shapes)[0]
    if Hi != Ho or Wi != Wo:
        return False

    deadline = time.time() + time_budget
    for ks in range(1, 30, 2):
        if time.time() > deadline:
            return False
        for use_bias in [False, True]:
            result = _lstsq_conv(exs, ks, use_bias, False)
            if result is None:
                continue
            Wconv, bias = result
            _save_conv_model(Wconv, bias, Hi, Wi, Ho, Wo, path, False)
            if validate(path, td):
                return True
            # Remove invalid file
            if os.path.exists(path):
                os.remove(path)
    return False


def solve_conv_variable(td, path, time_budget):
    """Conv solver for same-shape tasks with variable input sizes."""
    exs = get_exs(td)
    # Check that output shapes are the same
    out_shapes = {e[1].shape for e in exs}
    if len(out_shapes) != 1:
        return False
    Ho, Wo = list(out_shapes)[0]

    deadline = time.time() + time_budget
    for ks in range(1, 30, 2):
        if time.time() > deadline:
            return False
        for use_bias in [False, True]:
            result = _lstsq_conv(exs, ks, use_bias, True)
            if result is None:
                continue
            Wconv, bias = result
            _save_conv_model(Wconv, bias, 0, 0, Ho, Wo, path, True)
            if validate(path, td):
                return True
            if os.path.exists(path):
                os.remove(path)
    return False


def solve_conv_diffshape(td, path, time_budget):
    """Conv solver for tasks where output is smaller than input."""
    exs = get_exs(td)
    in_shapes = {e[0].shape for e in exs}
    out_shapes = {e[1].shape for e in exs}
    if len(in_shapes) != 1 or len(out_shapes) != 1:
        return False
    Hi, Wi = list(in_shapes)[0]
    Ho, Wo = list(out_shapes)[0]
    if Ho >= Hi or Wo >= Wi:
        return False

    deadline = time.time() + time_budget
    for ks in range(1, min(Hi, Wi) + 1, 2):
        if time.time() > deadline:
            return False
        # Conv with 'same' padding on input gives [1, CH, Hi, Wi]
        # Then crop to [1, CH, Ho, Wo] with offset
        max_dr = Hi - Ho
        max_dc = Wi - Wo
        for use_bias in [False, True]:
            # Build patches for the crop region
            all_P = []
            all_T = []
            n_features = CH * ks * ks
            skip = False
            for inp, out in exs:
                hi, wi = inp.shape
                ho, wo = out.shape
                oh_in = to_onehot(inp)
                oh_crop = oh_in[0, :, :hi, :wi].reshape(1, CH, hi, wi)
                padded = np.pad(
                    oh_crop,
                    ((0, 0), (0, 0), (ks // 2, ks // 2), (ks // 2, ks // 2)),
                    mode="constant",
                    constant_values=0,
                )
                for r in range(ho):
                    for c in range(wo):
                        dr = (hi - ho) // 2
                        dc = (wi - wo) // 2
                        patch = padded[0, :, r + dr : r + dr + ks, c + dc : c + dc + ks]
                        all_P.append(patch.flatten())
                        v = int(out[r, c])
                        t = np.zeros(CH, dtype=np.float32)
                        if 0 <= v < CH:
                            t[v] = 1.0
                        all_T.append(t)
            P = np.array(all_P, dtype=np.float32)
            T_oh = np.array(all_T, dtype=np.float32)
            n_patches = P.shape[0]
            features = n_features + (1 if use_bias else 0)
            if features > 20000 or (features > 5000 and n_patches > 2000):
                continue
            if use_bias:
                P_aug = np.concatenate(
                    [P, np.ones((n_patches, 1), dtype=np.float32)], axis=1
                )
            else:
                P_aug = P
            try:
                WT, _, _, _ = np.linalg.lstsq(P_aug, T_oh, rcond=None)
            except np.linalg.LinAlgError:
                continue

            if use_bias:
                Wconv = WT[:-1].T.reshape(CH, CH, ks, ks).astype(np.float32)
                bias = WT[-1].T.astype(np.float32)
            else:
                Wconv = WT.T.reshape(CH, CH, ks, ks).astype(np.float32)
                bias = None

            # Verify
            ok = True
            for inp, out in exs:
                hi, wi = inp.shape
                ho, wo = out.shape
                oh_in = to_onehot(inp)
                oh_crop_in = oh_in[0, :, :hi, :wi].reshape(1, CH, hi, wi)
                padded = np.pad(
                    oh_crop_in,
                    ((0, 0), (0, 0), (ks // 2, ks // 2), (ks // 2, ks // 2)),
                    mode="constant",
                    constant_values=0,
                )
                for r in range(ho):
                    for c in range(wo):
                        dr = (hi - ho) // 2
                        dc = (wi - wo) // 2
                        patch = padded[0, :, r + dr : r + dr + ks, c + dc : c + dc + ks]
                        pred = patch.flatten() @ WT
                        if np.argmax(pred) != int(out[r, c]):
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                crop_dr = (Hi - Ho) // 2
                crop_dc = (Wi - Wo) // 2
                _save_conv_diffshape_model(
                    Wconv, bias, Hi, Wi, Ho, Wo, crop_dr, crop_dc, ks, path
                )
                if validate(path, td):
                    return True
                if os.path.exists(path):
                    os.remove(path)
    return False


# ── Orchestrator ─────────────────────────────────────────────────────


_SOLVER_FUNCS = {
    "identity": s_identity,
    "constant": s_constant,
    "color_map": s_color_map,
    "transpose": s_transpose,
    "flip": s_flip,
    "rotate": s_rotate,
    "tile": s_tile,
    "upscale": s_upscale,
    "concat": s_concat,
    "spatial_gather": s_spatial_gather,
    "crop": s_crop,
}


def solve_task(task_num, task_data, output_dir, conv_budget=30.0):
    """Try analytical solvers then conv solvers. Returns (solved, name)."""
    os.makedirs(output_dir, exist_ok=True)
    fname = f"task_{task_num:03d}.onnx"
    fpath = os.path.join(output_dir, fname)

    # Phase 1: analytical solvers
    for name in ANALYTICAL_SOLVERS:
        func = _SOLVER_FUNCS[name]
        try:
            model = func(task_data)
        except Exception:
            continue
        if model is not None:
            onnx.save(model, fpath)
            if validate(fpath, task_data):
                return True, name
            if os.path.exists(fpath):
                os.remove(fpath)

    # Phase 2: convolution solvers
    # Determine which conv variant to try
    shapes = fixed_shapes(task_data)
    exs = get_exs(task_data)
    in_shapes = {e[0].shape for e in exs}
    out_shapes = {e[1].shape for e in exs}

    # Fixed-shape conv
    if shapes is not None:
        Hi, Wi = shapes[0]
        Ho, Wo = shapes[1]
        if Hi == Wi and Ho == Wo or True:
            if Hi >= Ho and Wi >= Wo:
                try:
                    if solve_conv_fixed(task_data, fpath, conv_budget):
                        return True, "conv_fixed"
                except Exception:
                    pass
                if os.path.exists(fpath):
                    os.remove(fpath)

    # Variable-shape conv (output shapes same, input shapes may differ)
    if len(out_shapes) == 1:
        try:
            if solve_conv_variable(task_data, fpath, conv_budget):
                return True, "conv_variable"
        except Exception:
            pass
        if os.path.exists(fpath):
            os.remove(fpath)

    # Diff-shape conv (output smaller than input)
    if shapes is not None:
        Hi, Wi = shapes[0]
        Ho, Wo = shapes[1]
        if Ho < Hi and Wo < Wi:
            try:
                if solve_conv_diffshape(task_data, fpath, conv_budget):
                    return True, "conv_diffshape"
            except Exception:
                pass
            if os.path.exists(fpath):
                os.remove(fpath)

    return False, ""


def main():
    parser = argparse.ArgumentParser(description="NeuroGolf 2026 ARC-AGI Solver")
    parser.add_argument("--data_file", required=True, help="Path to all_tasks.json")
    parser.add_argument(
        "--output_dir", default="submission", help="Output directory for ONNX files"
    )
    parser.add_argument(
        "--conv_budget",
        type=float,
        default=30.0,
        help="Time budget per task for conv solver (seconds)",
    )
    parser.add_argument(
        "--tasks", type=str, default=None, help="Comma-separated task numbers to solve"
    )
    args = parser.parse_args()

    tasks = load_tasks_json(args.data_file)
    total = len(tasks)

    if args.tasks is not None:
        selected = [int(t.strip()) for t in args.tasks.split(",")]
    else:
        selected = sorted(tasks.keys())

    solved_count = 0
    solver_counts = {}
    t0 = time.time()

    for idx, task_num in enumerate(selected):
        if task_num not in tasks:
            print(
                f"[{idx + 1}/{len(selected)}] task_{task_num:03d} not found, skipping"
            )
            continue
        td = tasks[task_num]["data"]
        print(
            f"[{idx + 1}/{len(selected)}] Solving task_{task_num:03d} ...",
            end=" ",
            flush=True,
        )
        solved, solver_name = solve_task(
            task_num, td, args.output_dir, args.conv_budget
        )
        if solved:
            solved_count += 1
            solver_counts[solver_name] = solver_counts.get(solver_name, 0) + 1
            print(f"OK ({solver_name})")
        else:
            print("FAILED")

    elapsed = time.time() - t0

    # Summary
    print()
    print("=" * 50)
    print(f"Solved: {solved_count}/{len(selected)}")
    print(f"Time:   {elapsed:.1f}s")
    if solver_counts:
        print("Solver distribution:")
        for name in sorted(solver_counts.keys()):
            print(f"  {name}: {solver_counts[name]}")

    # Create submission.zip
    output_dir = args.output_dir
    zip_path = os.path.join(os.path.dirname(output_dir) or ".", "submission.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        if os.path.isdir(output_dir):
            for fn in sorted(os.listdir(output_dir)):
                if fn.endswith(".onnx"):
                    zf.write(os.path.join(output_dir, fn), fn)
    print(f"Created {zip_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
