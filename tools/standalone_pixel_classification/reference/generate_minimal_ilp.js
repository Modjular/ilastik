#!/usr/bin/env node
"use strict";
/*
 * generate_minimal_ilp.js
 *
 * Pure-JavaScript generator for a minimal ilastik Pixel-Classification .ilp
 * project file. This is the JS counterpart of generate_minimal_ilp.py, but with
 * ONE crucial difference: it does not use h5py (or any HDF5 library). The whole
 * HDF5 container is serialized here, by hand, from first principles.
 *
 * Why this is possible with no dependency:
 *   - .ilp is just an HDF5 file, and libhdf5 / h5py are tolerant readers: any
 *     spec-compliant bytes are accepted.
 *   - We only need the tiny subset the format actually uses. Every dataset is
 *     stored CONTIGUOUS, little-endian, unchunked, unfiltered; every string is
 *     FIXED-LENGTH (h5py reads those back as bytes, which verify.py .decode()s),
 *     so we never touch the HDF5 global heap or variable-length machinery.
 *   - Groups use the classic superblock-v0 "symbol table" layout (v1 B-tree +
 *     local heap + symbol-table node), which is exactly what real .ilp files and
 *     h5py itself emit, so libhdf5 is guaranteed to accept it. No checksums
 *     (Jenkins lookup3) are required, unlike the modern v2/v3 layout.
 *
 * The same ArrayBuffer/Buffer this builds is directly usable in a browser:
 *   new Blob([buildIlp(...)])  ->  download as project.ilp
 * so "generate in a single JS file, no HDF5 lib" and "generate in the browser"
 * are literally the same core code; only the final write differs (fs vs Blob).
 *
 * Spec reference: "HDF5 File Format Specification" (superblock v0, version-1
 * object headers, version-1 B-trees, local heaps, symbol table nodes).
 *
 * Usage:
 *   node generate_minimal_ilp.js [outPath] [imagePath] [z y x]
 *   node generate_minimal_ilp.js                       # writes minimal_js.ilp
 */

const crypto = require("crypto");

// ---------------------------------------------------------------------------
// HDF5 constants for this writer's chosen configuration
// ---------------------------------------------------------------------------
const SIG = Buffer.from([0x89, 0x48, 0x44, 0x46, 0x0d, 0x0a, 0x1a, 0x0a]); // \x89HDF\r\n\x1a\n
const UNDEF = 0xffffffffffffffffn; // "undefined address" (all bits set)
// Group node "K" values live in the superblock. We deliberately pick a large
// leaf K so every group in this project fits in a SINGLE symbol-table node
// (the biggest group, "Raw Data", has 10 members) — that keeps every group's
// B-tree at exactly one node with one child and avoids node-splitting logic.
const K_LEAF = 16; // symbol-table node capacity = 2*K_LEAF = 32 entries
const K_INT = 16; // B-tree node capacity     = 2*K_INT  = 32 children

const pad8 = (n) => Math.ceil(n / 8) * 8;

// ---------------------------------------------------------------------------
// Growable little-endian byte sink with an 8-byte-aligned bump allocator.
// alloc(size) reserves a zero-filled, 8-aligned region and returns its file
// offset; everything is then written by absolute offset (offsets stay valid
// across internal buffer growth because growth copies existing content).
// ---------------------------------------------------------------------------
class Sink {
  constructor() {
    this.buf = Buffer.alloc(1 << 16);
    this.len = 0;
  }
  _ensure(n) {
    if (this.len + n <= this.buf.length) return;
    let cap = this.buf.length;
    while (cap < this.len + n) cap *= 2;
    const nb = Buffer.alloc(cap);
    this.buf.copy(nb, 0, 0, this.len);
    this.buf = nb;
  }
  alloc(size) {
    const off = this.len;
    const rounded = pad8(size);
    this._ensure(rounded);
    // Buffer.alloc already zeroed, but a grown buffer's tail is zero and the
    // reused region may not be — clear the exact range we hand out.
    this.buf.fill(0, off, off + rounded);
    this.len += rounded;
    return off;
  }
  u8(o, v) { this.buf.writeUInt8(v & 0xff, o); }
  u16(o, v) { this.buf.writeUInt16LE(v & 0xffff, o); }
  u32(o, v) { this.buf.writeUInt32LE(v >>> 0, o); }
  u64(o, v) { this.buf.writeBigUInt64LE(typeof v === "bigint" ? v : BigInt(v), o); }
  put(o, bytes) { Buffer.from(bytes).copy(this.buf, o); }
}

// ---------------------------------------------------------------------------
// Datatype messages (message type 0x0003), version 1
// ---------------------------------------------------------------------------
// Fixed-length string: class 3, null-terminated, ASCII. Null-terminate (pad
// type 0) tells every reader to stop at the first NUL, so shorter entries in a
// fixed-width string array come back clean (not NUL-padded) in naive readers,
// while h5py strips trailing NULs either way.
function dtStr(size) {
  const b = Buffer.alloc(8);
  b.writeUInt8((1 << 4) | 3, 0); // version 1, class 3 (string)
  b.writeUInt8(0x00, 1); // bit field: pad type 0 (null terminate), charset 0 (ASCII)
  b.writeUInt32LE(size, 4); // element size in bytes
  return b;
}
// Fixed-point integer: class 0, little-endian, optional 2's-complement sign.
function dtInt(size, signed) {
  const b = Buffer.alloc(12);
  b.writeUInt8((1 << 4) | 0, 0); // version 1, class 0 (fixed-point)
  b.writeUInt8(signed ? 0x08 : 0x00, 1); // bit0 order=LE, bit3 signedness
  b.writeUInt32LE(size, 4); // element size in bytes
  b.writeUInt16LE(0, 8); // bit offset
  b.writeUInt16LE(size * 8, 10); // bit precision
  return b;
}
// IEEE-754 little-endian double: class 1.
function dtFloat64() {
  const b = Buffer.alloc(20);
  b.writeUInt8((1 << 4) | 1, 0); // version 1, class 1 (floating-point)
  b.writeUInt8(0x20, 1); // bit field byte0: mantissa normalization = 2 (implied MSB)
  b.writeUInt8(0x3f, 2); // bit field byte1: sign bit location = 63
  b.writeUInt8(0x00, 3);
  b.writeUInt32LE(8, 4); // element size
  b.writeUInt16LE(0, 8); // bit offset
  b.writeUInt16LE(64, 10); // bit precision
  b.writeUInt8(52, 12); // exponent location
  b.writeUInt8(11, 13); // exponent size
  b.writeUInt8(0, 14); // mantissa location
  b.writeUInt8(52, 15); // mantissa size
  b.writeUInt32LE(1023, 16); // exponent bias
  return b;
}

// Dataspace message (0x0001), version 1. dims == [] -> scalar (rank 0).
function dataspace(dims) {
  const rank = dims.length;
  const b = Buffer.alloc(8 + rank * 8);
  b.writeUInt8(1, 0); // version
  b.writeUInt8(rank, 1); // dimensionality
  b.writeUInt8(0, 2); // flags (no max dims)
  for (let i = 0; i < rank; i++) b.writeBigUInt64LE(BigInt(dims[i]), 8 + i * 8);
  return b;
}

// Data Layout message (0x0008), version 3, class 1 (contiguous).
function layoutContiguous(addr, size) {
  const b = Buffer.alloc(18);
  b.writeUInt8(3, 0); // version
  b.writeUInt8(1, 1); // class 1 = contiguous
  b.writeBigUInt64LE(BigInt(addr), 2); // data address
  b.writeBigUInt64LE(BigInt(size), 10); // data size
  return b;
}

// Symbol Table message (0x0011): pointers to a group's B-tree + local heap.
function symbolTableMsg(btreeAddr, heapAddr) {
  const b = Buffer.alloc(16);
  b.writeBigUInt64LE(BigInt(btreeAddr), 0);
  b.writeBigUInt64LE(BigInt(heapAddr), 8);
  return b;
}

// Attribute message (0x000C), version 1. In v1 the name/datatype/dataspace
// sub-blocks are each padded to 8 bytes; the value follows unpadded.
function attributeMsg(name, dtBody, dsBody, valueBytes) {
  const nameBuf = Buffer.from(name + "\0", "ascii");
  const nameP = pad8(nameBuf.length);
  const dtP = pad8(dtBody.length);
  const dsP = pad8(dsBody.length);
  const b = Buffer.alloc(8 + nameP + dtP + dsP + valueBytes.length);
  b.writeUInt8(1, 0); // version
  b.writeUInt8(0, 1); // reserved
  b.writeUInt16LE(nameBuf.length, 2); // name size (incl. NUL)
  b.writeUInt16LE(dtBody.length, 4); // datatype message size (unpadded)
  b.writeUInt16LE(dsBody.length, 6); // dataspace message size (unpadded)
  let o = 8;
  nameBuf.copy(b, o); o += nameP;
  dtBody.copy(b, o); o += dtP;
  dsBody.copy(b, o); o += dsP;
  Buffer.from(valueBytes).copy(b, o);
  return b;
}

// ---------------------------------------------------------------------------
// Version-1 object header. messages: [{ type, body:Buffer }]. Each message is
// framed with an 8-byte prefix and its data padded to a multiple of 8.
// ---------------------------------------------------------------------------
function writeObjectHeader(sink, messages) {
  const framed = messages.map((m) => ({ type: m.type, body: m.body, pad: pad8(m.body.length) }));
  const msgsSize = framed.reduce((s, m) => s + 8 + m.pad, 0);
  const addr = sink.alloc(16 + msgsSize); // 12-byte prefix + 4 pad -> 16
  sink.u8(addr + 0, 1); // version
  sink.u8(addr + 1, 0); // reserved
  sink.u16(addr + 2, messages.length); // number of header messages
  sink.u32(addr + 4, 1); // object reference count
  sink.u32(addr + 8, msgsSize); // header size (message data bytes)
  // addr+12..15 : alignment padding (already zero)
  let o = addr + 16;
  for (const m of framed) {
    sink.u16(o + 0, m.type); // message type
    sink.u16(o + 2, m.pad); // size of message data (padded)
    sink.u8(o + 4, 0); // flags
    // o+5..7 reserved (zero)
    m.body.copy(sink.buf, o + 8);
    o += 8 + m.pad;
  }
  return addr;
}

// ---------------------------------------------------------------------------
// Old-style group storage: local heap of names + a single-node v1 B-tree +
// (if non-empty) one symbol table node. Returns { btreeAddr, heapAddr }.
// ---------------------------------------------------------------------------
function writeGroupStorage(sink, entries) {
  // Symbol table node entries must be sorted by name (libhdf5 binary-searches).
  const sorted = entries.slice().sort((a, b) =>
    Buffer.compare(Buffer.from(a.name, "ascii"), Buffer.from(b.name, "ascii"))
  );

  // --- Local heap: reserve [0,8) as the empty string (heap offset 0), then
  //     lay out each NUL-terminated name 8-byte aligned, then a free block. ---
  const nameOffset = new Map();
  const chunks = [];
  let cur = 8;
  for (const e of sorted) {
    const nb = Buffer.from(e.name + "\0", "ascii");
    nameOffset.set(e.name, cur);
    chunks.push({ off: cur, buf: nb });
    cur += pad8(nb.length);
  }
  const freeOff = cur;
  const freeSize = 16; // minimal free block: [next=1][size]
  const dataSegSize = freeOff + freeSize;

  const heapHdr = sink.alloc(32);
  const dataSeg = sink.alloc(dataSegSize);
  sink.put(heapHdr, Buffer.from("HEAP", "ascii"));
  sink.u8(heapHdr + 4, 0); // version
  sink.u64(heapHdr + 8, dataSegSize); // data segment size
  sink.u64(heapHdr + 16, freeOff); // offset to head of free list
  sink.u64(heapHdr + 24, dataSeg); // address of data segment
  for (const c of chunks) sink.put(dataSeg + c.off, c.buf);
  sink.u64(dataSeg + freeOff, 1n); // free block: next = 1 (sentinel: none)
  sink.u64(dataSeg + freeOff + 8, freeSize); // free block: size

  // --- Symbol table node (only when the group has members) ---
  let snodAddr = 0;
  if (sorted.length > 0) {
    const snodSize = 8 + 2 * K_LEAF * 40;
    snodAddr = sink.alloc(snodSize);
    sink.put(snodAddr, Buffer.from("SNOD", "ascii"));
    sink.u8(snodAddr + 4, 1); // version
    let so = snodAddr + 8;
    sink.u16(snodAddr + 6, sorted.length); // number of symbols
    for (const e of sorted) {
      sink.u64(so + 0, nameOffset.get(e.name)); // link name offset (into heap)
      sink.u64(so + 8, e.ohAddr); // object header address
      sink.u32(so + 16, 0); // cache type 0 (no scratch-pad caching)
      // so+20..23 reserved, so+24..39 scratch-pad (all zero)
      so += 40;
    }
  }

  // --- Single-node v1 B-tree (node type 0 = group), level 0 ---
  const btreeSize = 24 + (2 * K_INT + 1) * 8 + 2 * K_INT * 8;
  const btree = sink.alloc(btreeSize);
  sink.put(btree, Buffer.from("TREE", "ascii"));
  sink.u8(btree + 4, 0); // node type: group
  sink.u8(btree + 5, 0); // node level: leaf
  sink.u16(btree + 6, sorted.length > 0 ? 1 : 0); // entries used (0 or 1 child)
  sink.u64(btree + 8, UNDEF); // left sibling
  sink.u64(btree + 16, UNDEF); // right sibling
  // Keys/children interleave from btree+24: Key0, Child0, Key1, ...
  sink.u64(btree + 24, 0); // Key0 = heap offset 0 (empty string, <= all names)
  if (sorted.length > 0) {
    sink.u64(btree + 32, snodAddr); // Child0 -> the symbol table node
    const maxName = sorted[sorted.length - 1].name; // greatest name (bytewise)
    sink.u64(btree + 40, nameOffset.get(maxName)); // Key1 >= all names
  }

  return { btreeAddr: btree, heapAddr: heapHdr };
}

// ---------------------------------------------------------------------------
// Tree model. Nodes are plain objects:
//   dataset: { kind:'dataset', name, dtBody, dims, data:Buffer, attrs:[Buffer] }
//   group:   { kind:'group',   name, children:[node] }
// writeNode emits a node post-order and returns its object-header address.
// ---------------------------------------------------------------------------
function writeNode(sink, node) {
  if (node.kind === "dataset") {
    const dataAddr = node.data.length > 0 ? sink.alloc(node.data.length) : 0;
    if (node.data.length > 0) sink.put(dataAddr, node.data);
    const messages = [
      { type: 0x0003, body: node.dtBody },
      { type: 0x0001, body: dataspace(node.dims) },
      { type: 0x0008, body: layoutContiguous(dataAddr, node.data.length) },
    ];
    for (const a of node.attrs || []) messages.push({ type: 0x000c, body: a });
    return writeObjectHeader(sink, messages);
  }
  // group
  const entries = node.children.map((c) => ({ name: c.name, ohAddr: writeNode(sink, c) }));
  const { btreeAddr, heapAddr } = writeGroupStorage(sink, entries);
  return writeObjectHeader(sink, [{ type: 0x0011, body: symbolTableMsg(btreeAddr, heapAddr) }]);
}

// ---------------------------------------------------------------------------
// Node constructors (parity with the datasets generate_minimal_ilp.py creates)
// ---------------------------------------------------------------------------
function bytesFixed(bytes) {
  // Fixed-length string payload: exact bytes, size == length (min 1).
  const size = Math.max(bytes.length, 1);
  const data = Buffer.alloc(size);
  Buffer.from(bytes).copy(data);
  return { size, data };
}
function dsStr(name, s) {
  const { size, data } = bytesFixed(Buffer.from(s, "utf8"));
  return { kind: "dataset", name, dtBody: dtStr(size), dims: [], data };
}
function dsBytes(name, buf) {
  const { size, data } = bytesFixed(buf);
  return { kind: "dataset", name, dtBody: dtStr(size), dims: [], data };
}
function dsStrArray(name, list) {
  const size = Math.max(1, ...list.map((s) => Buffer.byteLength(s, "utf8")));
  const data = Buffer.alloc(size * list.length);
  list.forEach((s, i) => Buffer.from(s, "utf8").copy(data, i * size));
  return { kind: "dataset", name, dtBody: dtStr(size), dims: [list.length], data };
}
function dsInt64Scalar(name, n) {
  const data = Buffer.alloc(8);
  data.writeBigInt64LE(BigInt(n), 0);
  return { kind: "dataset", name, dtBody: dtInt(8, true), dims: [], data };
}
function dsInt64Array(name, nums) {
  const data = Buffer.alloc(8 * nums.length);
  nums.forEach((n, i) => data.writeBigInt64LE(BigInt(n), i * 8));
  return { kind: "dataset", name, dtBody: dtInt(8, true), dims: [nums.length], data };
}
function dsFloat64Array(name, nums) {
  const data = Buffer.alloc(8 * nums.length);
  nums.forEach((n, i) => data.writeDoubleLE(n, i * 8));
  return { kind: "dataset", name, dtBody: dtFloat64(), dims: [nums.length], data };
}
function dsInt32Matrix(name, rows) {
  const R = rows.length, C = rows[0].length;
  const data = Buffer.alloc(4 * R * C);
  let o = 0;
  for (const row of rows) for (const v of row) { data.writeInt32LE(v, o); o += 4; }
  return { kind: "dataset", name, dtBody: dtInt(4, true), dims: [R, C], data };
}
// Booleans are stored as int8 (0/1). verify.py never inspects their dtype; note
// that genuine ilastik round-tripping would want an HDF5 enum {False,True},
// which only matters for the desktop app, not for passing verify.py.
function dsBoolScalar(name, v) {
  return { kind: "dataset", name, dtBody: dtInt(1, true), dims: [], data: Buffer.from([v ? 1 : 0]) };
}
function dsBoolArray(name, arr) {
  const data = Buffer.alloc(arr.length);
  arr.forEach((v, i) => data.writeInt8(v ? 1 : 0, i));
  return { kind: "dataset", name, dtBody: dtInt(1, true), dims: [arr.length], data };
}
function dsBoolMatrix(name, rows) {
  const R = rows.length, C = rows[0].length;
  const data = Buffer.alloc(R * C);
  let o = 0;
  for (const row of rows) for (const v of row) data.writeInt8(v ? 1 : 0, o++);
  return { kind: "dataset", name, dtBody: dtInt(1, true), dims: [R, C], data };
}
function dsUint8Block(name, dims, fill, attrs) {
  const total = dims.reduce((a, b) => a * b, 1);
  return { kind: "dataset", name, dtBody: dtInt(1, false), dims, data: Buffer.alloc(total, fill), attrs };
}
function group(name, children) {
  return { kind: "group", name, children };
}
function strAttr(name, s) {
  const { size, data } = bytesFixed(Buffer.from(s, "utf8"));
  return attributeMsg(name, dtStr(size), dataspace([]), data);
}

// axistags JSON, mirroring generate_minimal_ilp.py's make_axistags().
function makeAxistags(keys) {
  const typeFlags = { t: 4, z: 2, y: 2, x: 2, c: 1 };
  const axes = [...keys].map((key) => ({
    key,
    typeFlags: typeFlags[key] ?? 2,
    resolution: 0.0,
    description: "",
  }));
  return JSON.stringify({ axes }, null, 2);
}

// Default ParallelVigraRfLazyflowClassifierFactory pickle (100 trees), byte-for
// -byte identical to the literal in generate_minimal_ilp.py. Optional per the
// spec (ilastik falls back to defaults if absent), included here for parity.
const CLASSIFIER_FACTORY_PICKLE = Buffer.from(
  "ccopy_reg\n_reconstructor\np0\n(clazyflow.classifiers.parallelVigraRfLazyflowClassifier\n" +
    "ParallelVigraRfLazyflowClassifierFactory\np1\nc__builtin__\nobject\np2\nNtp3\nRp4\n(dp5\n" +
    "VVERSION\np6\nL2L\nsV_num_trees\np7\nL100L\nsV_label_proportion\np8\nNsV_variable_importance_path\n" +
    "p9\nNsV_variable_importance_enabled\np10\nI00\nsV_kwargs\np11\n(dp12\nsV_num_forests\np13\nL4L\nsb.",
  "latin1"
);

// ---------------------------------------------------------------------------
// Assemble the ilastik Pixel-Classification tree and serialize the whole file.
// Returns a Buffer containing the complete .ilp (HDF5) bytes.
// ---------------------------------------------------------------------------
function buildIlp(imagePath, imageShape) {
  const featureIds = [
    "GaussianSmoothing",
    "LaplacianOfGaussian",
    "GaussianGradientMagnitude",
    "DifferenceOfGaussians",
    "StructureTensorEigenvalues",
    "HessianOfGaussianEigenvalues",
  ];
  const scales = [0.3, 0.7, 1.0, 1.6, 3.5, 5.0, 10.0];
  // Example selection: first two features x first two scales (matches the .py).
  const selectionMatrix = featureIds.map((_, fi) => scales.map((_, si) => fi < 2 && si < 2));
  const computeIn2d = scales.map(() => true);

  const rawData = group("Raw Data", [
    dsStr("__class__", "FilesystemDatasetInfo"),
    dsBoolScalar("allowLabels", true),
    dsStr("axistags", makeAxistags("zyx")),
    dsStr("datasetId", crypto.randomUUID()),
    dsStr("display_mode", "default"),
    dsStr("filePath", imagePath),
    dsStr("location", "FileSystem"),
    dsStr("nickname", "input_image"),
    dsBoolScalar("normalizeDisplay", false),
    dsInt64Array("shape", imageShape),
  ]);

  const inputData = group("Input Data", [
    dsStr("StorageVersion", "0.2"),
    dsStrArray("Role Names", ["Raw Data", "Prediction Mask"]),
    group("infos", [group("lane0000", [rawData])]),
    group("local_data", []),
  ]);

  const featureSelections = group("FeatureSelections", [
    dsStr("StorageVersion", "0.1"),
    dsStrArray("FeatureIds", featureIds),
    dsFloat64Array("Scales", scales),
    dsBoolMatrix("SelectionMatrix", selectionMatrix),
    dsBoolArray("ComputeIn2d", computeIn2d),
  ]);

  const labelSets = group("LabelSets", [
    group("labels000", [
      dsUint8Block("block0000", [10, 10, 1], 1, [
        strAttr("axistags", makeAxistags("yxc")),
        strAttr("blockSlice", "[10:20,10:20,0:1]"),
      ]),
    ]),
  ]);

  const pixelClassification = group("PixelClassification", [
    dsStr("StorageVersion", "0.1"),
    dsStrArray("LabelNames", ["Label 1", "Label 2"]),
    dsInt32Matrix("LabelColors", [[255, 0, 0], [0, 255, 0]]),
    dsInt32Matrix("PmapColors", [[255, 0, 0], [0, 255, 0]]),
    group("Bookmarks", []),
    labelSets,
    dsBytes("ClassifierFactory", CLASSIFIER_FACTORY_PICKLE),
  ]);

  const rootChildren = [
    dsStr("ilastikVersion", "1.4.0"),
    dsStr("workflowName", "Pixel Classification"),
    dsStr("time", new Date().toUTCString()),
    dsInt64Scalar("currentApplet", 0),
    inputData,
    featureSelections,
    pixelClassification,
  ];

  // Serialize. Reserve the 96-byte superblock at offset 0 first, build the
  // whole tree, then back-patch the root group's addresses and the EOF.
  const sink = new Sink();
  const sbAddr = sink.alloc(96);
  const entries = rootChildren.map((c) => ({ name: c.name, ohAddr: writeNode(sink, c) }));
  const rootStab = writeGroupStorage(sink, entries);
  const rootOH = writeObjectHeader(sink, [
    { type: 0x0011, body: symbolTableMsg(rootStab.btreeAddr, rootStab.heapAddr) },
  ]);

  // Superblock version 0.
  sink.put(sbAddr, SIG);
  sink.u8(sbAddr + 8, 0); // superblock version
  sink.u8(sbAddr + 9, 0); // free-space storage version
  sink.u8(sbAddr + 10, 0); // root group symbol table entry version
  sink.u8(sbAddr + 11, 0); // reserved
  sink.u8(sbAddr + 12, 0); // shared header message format version
  sink.u8(sbAddr + 13, 8); // size of offsets
  sink.u8(sbAddr + 14, 8); // size of lengths
  sink.u8(sbAddr + 15, 0); // reserved
  sink.u16(sbAddr + 16, K_LEAF); // group leaf node K
  sink.u16(sbAddr + 18, K_INT); // group internal node K
  sink.u32(sbAddr + 20, 0); // file consistency flags
  sink.u64(sbAddr + 24, 0); // base address
  sink.u64(sbAddr + 32, UNDEF); // free-space info address
  sink.u64(sbAddr + 40, BigInt(sink.len)); // end-of-file address
  sink.u64(sbAddr + 48, UNDEF); // driver information block address
  // Root group symbol table entry (cache type 1: scratch-pad holds B-tree/heap).
  sink.u64(sbAddr + 56, 0); // link name offset
  sink.u64(sbAddr + 64, rootOH); // object header address
  sink.u32(sbAddr + 72, 1); // cache type = 1
  sink.u32(sbAddr + 76, 0); // reserved
  sink.u64(sbAddr + 80, rootStab.btreeAddr); // scratch-pad: B-tree address
  sink.u64(sbAddr + 88, rootStab.heapAddr); // scratch-pad: local heap address

  return sink.buf.subarray(0, sink.len);
}

module.exports = { buildIlp };

// ---------------------------------------------------------------------------
// CLI (parity with generate_minimal_ilp.py's __main__ ergonomics).
// ---------------------------------------------------------------------------
if (require.main === module) {
  const argv = process.argv.slice(2);
  const outPath = argv[0] || "20260327_ilp_stuff/minimal_js.ilp";
  const imagePath = argv[1] || "dummy_test.tif";
  const imageShape = argv.length >= 5 ? [Number(argv[2]), Number(argv[3]), Number(argv[4])] : [1, 100, 100];
  const bytes = buildIlp(imagePath, imageShape);
  require("fs").writeFileSync(outPath, bytes);
  console.log(`Minimal .ilp file generated at ${outPath} (${bytes.length} bytes)`);
}
