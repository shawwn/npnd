import numpy as np
from npnd import errors
from . import one_hot as one_hot_lib
from . import shape as shape_lib
import math
from typing import List, Tuple
import dataclasses

def prn(*args, **kws):
  print(*args, **kws)
  if args:
    return args[-1]

# reference implementation of ONNX GatherElements.
def gather_elements_ref(tensor, indices, axis=0) -> np.ndarray:
  tensor = np.asarray(tensor).astype(float)
  indices = np.asarray(indices).astype(int)
  out = np.zeros_like(tensor, shape=indices.shape)
  for src in shape_lib.ndindex(indices.shape):
    dst = list(src)
    dst[axis] = indices[src]
    dst = tuple(dst)
    out[src] = tensor[dst]
  return out

def gather_elements(tensor, indices, axis=0) -> np.ndarray:
  dtype = np.asarray(tensor).dtype
  tensor = np.asarray(tensor).astype(float)
  indices = np.asarray(indices).astype(int)
  hot = one_hot_lib.one_hot(indices, tensor.shape[axis], dtype=tensor.dtype, axis=axis + 1)
  # insert a dimension on the same axis that one_hot inserted itself.
  out = np.expand_dims(tensor, axis)
  # hot = np.swapaxes(hot, axis, axis + 1)
  print(tensor.shape, hot.shape, out.shape)
  print('tensor\n', tensor.shape, '\n', tensor)
  print('hot\n', hot.shape, '\n', hot)
  print('indices\n', indices.shape, '\n', indices)
  print('out\n', out.shape, '\n', out)
  out = np.broadcast_to(out, hot.shape)
  print('out after broadcasting\n', out.shape, '\n', out)
  # multiply (with broadcasting) the tensor against the one_hot encoded indices, along axis.
  out = np.multiply(out, hot)
  print('out after multiply\n', out.shape, '\n', out)
  # sum along the one-hot axis.
  out = np.sum(out, axis=axis + 1)
  out = out.astype(dtype)
  return out

# Python implementation of ONNX GatherElements
def gather_elements(params, indices, axis=0) -> np.ndarray:
  params = np.asarray(params).astype(float)
  indices = np.asarray(indices).astype(int)
  tensor = params[tuple([slice(0, i) for i in indices.shape])]
  hot = one_hot_lib.one_hot(indices, tensor.shape[axis], dtype=tensor.dtype, axis=axis + 1)
  out = np.expand_dims(tensor, axis)
  out = np.multiply(out, hot)
  out = np.sum(out, axis=axis + 1)
  return out

# Python implementation of ONNX Gather (aka numpy.take)
def gather(params, indices, axis=0, wraparound_negative=True) -> np.ndarray:
  params = np.asarray(params)
  indices = np.asarray(indices).astype(int)
  if axis < 0:
    axis += len(params.shape)
  outer_shape, depth, inner_shape = params.shape[:axis], params.shape[axis], params.shape[axis+1:]
  result_shape = outer_shape + indices.shape + inner_shape
  # data = params.reshape((math.prod(outer_shape),) + (depth,) + (math.prod(inner_shape),))
  # data = data.astype(float)
  if False:
    data = params.reshape((math.prod(outer_shape),) + (1,) + (depth,) + (math.prod(inner_shape),))
    indices_flat = indices.reshape((1, -1, 1,))
    hot = one_hot_lib.one_hot(indices_flat, depth, axis=2, dtype=params.dtype, wraparound_negative=wraparound_negative)
    out = data * hot
    out = out.sum(2)
  elif False:
    indices_flat = indices.reshape((-1,))
    hot = one_hot_lib.one_hot(indices_flat, depth, axis=-1, dtype=params.dtype, wraparound_negative=wraparound_negative)
    data = np.expand_dims(data, 1)
    hot = np.expand_dims(hot, 0)
    hot = np.expand_dims(hot, -1)
    out = data * hot
    out = out.sum(2)
  else:
    data = params.reshape((math.prod(outer_shape),) + (depth,) + (math.prod(inner_shape),))
    indices_flat = indices.reshape((-1,))
    # hot = one_hot_lib.one_hot(indices_flat, depth, axis=-2, dtype=params.dtype, wraparound_negative=wraparound_negative)
    hot = one_hot_lib.one_hot(indices_flat, depth, axis=-1, dtype=params.dtype, wraparound_negative=wraparound_negative)
    out = hot @ data
  out = out.reshape(result_shape)
  # out = out.astype(params.dtype)
  return out

def onnx_gather(params, indices, axis=0) -> np.ndarray:
  params = np.asarray(params).astype(float)
  indices = np.asarray(indices).astype(int)
  return gather_nd(params, np.take(np.stack(shape_lib.ndshape(params.shape), -1), indices, axis=axis))

def contract(A, B, axis=-1) -> np.ndarray:
  A = np.asarray(A)
  B = np.asarray(B)
  assert A.shape[axis] == B.shape[axis]
  C = []
  for i in range(A.shape[axis]):
    C.append(np.take(A, i, axis=axis) @ np.take(B, i, axis=axis))
  return C

def preprocessDataForMXM(data, indices, axis=0, forGather=True):
  data = np.asarray(data).astype(float)
  indices = np.asarray(indices).astype(int)
  dataRank = len(data.shape)
  #   if (axis != dataRank - 1) {
  #     llvm::SmallVector<int64_t, 4> dataPermVec(dataRank, 0);
  #     for (int64_t idx = 0; idx < dataRank - 1; ++idx) {
  #       dataPermVec[idx] = idx < axis ? idx : idx + 1;
  #     }
  #     dataPermVec[dataRank - 1] = axis;
  #     data = rewriter.create<mlir::GroqMemRefTransposeOp>(loc, data, dataPermVec)
  #                .getResult();
  #     dataTiledType = data.getType().cast<groq::TiledMemRefType>();
  #   }
  if axis != dataRank - 1:
    dataPermVec = [0] * dataRank
    for idx in range(dataRank - 1):
      dataPermVec[idx] = idx if idx < axis else idx + 1
    dataPermVec[dataRank - 1] = axis
    data = np.transpose(data, dataPermVec)
  #   if (!forGather) {
  if not forGather:
    #   // We're reshaping for GatherElements, we just need to add a one in
    #   // the second last position
    #   auto dataTensorShape = dataTiledType.getTensorShape();
    dataTensorShape = data.shape
    #   llvm::SmallVector<int64_t, 4> reshapedTensorShape(dataRank + 1, 1);
    reshapedTensorShape = [1] * (dataRank + 1)
    #   for (int64_t i = 0; i < dataRank - 1; ++i) {
    #     reshapedTensorShape[i] = dataTensorShape[i];
    #   }
    for i in range(dataRank - 1):
      reshapedTensorShape[i] = dataTensorShape[i]
    #   reshapedTensorShape[dataRank] = dataTensorShape[dataRank - 1];
    reshapedTensorShape[dataRank] = dataTensorShape[dataRank - 1]
    #   auto reshapeTiledType =
    #       groq::TiledMemRefType::get(reshapedTensorShape, {dataRank},
    #           dataTiledType.getTileSizes(), dataTiledType.getElementType());
    #   return rewriter
    #       .create<mlir::GroqMemRefReshapeOp>(loc, reshapeTiledType, data)
    #       .getResult();
    return np.reshape(data, reshapedTensorShape)
  # }
  # auto lwbWidth = groqChip->archParams_.mxmLwbInputWidth_.at(DType::uint8);
  #
  # auto elType = getBaseType(data);
  # auto elTypeSize = elType.getIntOrFloatBitWidth() / BITS_PER_BYTE;
  #
  # // Reduce the concurrency in half for 32-bit data-types
  # auto weightConcurrency = elTypeSize == 4 ? lwbWidth / 2 : lwbWidth;
  # if (dataRank > 1) {
  #   data = rewriter.create<mlir::GroqRestructureMemRefOp>(
  #       loc, dataTiledType, data, weightConcurrency, dataRank - 2, true);
  # }
  #
  # // Reshape the transposed data s.t. the indices are broadcasted first
  # auto dataTensorShape = dataTiledType.getTensorShape();
  dataTensorShape = data.shape
  # auto reshapedRank = dataRank + indicesTiledType.getTensorShape().size() - 1 +
  #                     (dataRank == 1 ? 1 : 0);
  reshapedRank = dataRank + len(indices.shape) - 1 + (1 if dataRank == 1 else 0)
  # llvm::SmallVector<int64_t, 4> reshapedTensorShape(reshapedRank, 1);
  reshapedTensorShape = [1] * reshapedRank
  # for (int64_t i = 0; i < dataRank - 2; ++i) {
  #   reshapedTensorShape[i] = dataTensorShape[i];
  # }
  for i in range(dataRank - 2):
    reshapedTensorShape[i] = dataTensorShape[i]
  # reshapedTensorShape[reshapedTensorShape.size() - 1] =
  #     dataTensorShape[dataRank - 1];
  reshapedTensorShape[len(reshapedTensorShape) - 1] = dataTensorShape[dataRank - 1]
  # if (dataRank > 1) {
  #   reshapedTensorShape[reshapedTensorShape.size() - 2] =
  #       dataTensorShape[dataRank - 2];
  # }
  if dataRank > 1:
    reshapedTensorShape[len(reshapedTensorShape) - 2] = dataTensorShape[dataRank - 2]
  #
  # auto dataReshapedTiledType = groq::TiledMemRefType::get(reshapedTensorShape,
  #     {static_cast<int64_t>(reshapedTensorShape.size() - 1)},
  #     dataTiledType.getTileSizes(), dataTiledType.getElementType());
  #
  # return rewriter
  #     .create<mlir::GroqMemRefReshapeOp>(loc, dataReshapedTiledType, data)
  #     .getResult();
  return np.reshape(data, reshapedTensorShape)

# /// Takes indices and turns them into a one-hot-encoded version of the indices
# /// compatible with performing an embeddings look-up using the MXM.
# /// This is equivalent to the following:
# /// indices -> MemRefReshape -> MemRefTranspose -> TileTranspose -> Broadcast
# /// -> CmpEq(constant) -> output
# /// Where constant is a constant containing a tiled version of the indices in
# /// ascending order.
# mlir::Value oneHotEncode(const GroqChip *const groqChip,
#     mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
#     mlir::Value indices, groq::TiledMemRefType dataTiledType,
#     const int64_t axis) {
def oneHotEncode(data, indices, axis=0) -> np.ndarray:
  data = np.asarray(data).astype(float)
  indices = np.asarray(indices).astype(int)
  # auto indexElType = getBaseType(indices);
  # auto indicesTiledType = indices.getType().cast<groq::TiledMemRefType>();
  # // Need to reshape the indices to add an additional dimension corresponding
  # // to the one hot encoded dimension
  # llvm::SmallVector<int64_t, 4> reshapeTensorShape(
  #     indicesTiledType.getTensorShape().begin(),
  #     indicesTiledType.getTensorShape().end());
  reshapeTensorShape = list(indices.shape)
  # reshapeTensorShape.push_back(1);
  reshapeTensorShape.append(1)
  #
  # auto reshapeTiledType = groq::TiledMemRefType::get(reshapeTensorShape,
  #     indicesTiledType.getTileAxes(), indicesTiledType.getTileSizes(),
  #     indicesTiledType.getElementType());
  #
  # auto reshapeRes =
  #     rewriter.create<mlir::GroqMemRefReshapeOp>(loc, reshapeTiledType, indices)
  #         .getResult();
  reshapeRes = np.reshape(indices, reshapeTensorShape)
  #
  # // Need to shuffle the tiling (i.e. transpose) such that the indices are
  # // tiled on the newly added "1" dimension
  # // This is a two step process of MemRefTranspose -> Transpose
  # llvm::SmallVector<int64_t, 4> permVec(reshapeTensorShape.size(), 0);
  permVec = [0] * len(reshapeTensorShape)
  # for (uint64_t i = 0; i < reshapeTensorShape.size() - 2; ++i) {
  #   permVec[i] = i;
  # }
  for i in range(len(reshapeTensorShape) - 2):
    permVec[i] = i
  # permVec[reshapeTensorShape.size() - 1] = reshapeTensorShape.size() - 2;
  # permVec[reshapeTensorShape.size() - 2] = reshapeTensorShape.size() - 1;
  permVec[len(reshapeTensorShape) - 1] = len(reshapeTensorShape) - 2
  permVec[len(reshapeTensorShape) - 2] = len(reshapeTensorShape) - 1
  #
  # auto shuffleRes =
  #     rewriter.create<mlir::GroqMemRefTransposeOp>(loc, reshapeRes, permVec)
  #         .getResult();
  shuffleRes = np.transpose(reshapeRes, permVec)
  # auto shuffleTiledType = shuffleRes.getType().cast<groq::TiledMemRefType>();
  # auto shuffleTensorShape = shuffleTiledType.getTensorShape();
  shuffleTensorShape = shuffleRes.shape
  #
  # auto transposeTensorShape = llvm::SmallVector<int64_t, 4>(
  #     shuffleTensorShape.begin(), shuffleTensorShape.end());
  transposeTensorShape = list(shuffleTensorShape)
  # transposeTensorShape[transposeTensorShape.size() - 1] =
  #     shuffleTensorShape[transposeTensorShape.size() - 2];
  transposeTensorShape[len(transposeTensorShape) - 1] = shuffleTensorShape[len(transposeTensorShape) - 2]
  # transposeTensorShape[transposeTensorShape.size() - 2] =
  #     shuffleTensorShape[transposeTensorShape.size() - 1];
  transposeTensorShape[len(transposeTensorShape) - 2] = shuffleTensorShape[len(transposeTensorShape) - 1]
  # auto transposeTiledType = groq::TiledMemRefType::get(transposeTensorShape,
  #     shuffleTiledType.getTileAxes(), shuffleTiledType.getTileSizes(),
  #     shuffleTiledType.getElementType());
  #
  # auto transposeRes =
  #     MXMTranspose(groqChip, shuffleRes, transposeTiledType, rewriter, loc);
  transposeRes = MXMTranspose(shuffleRes, transposeTensorShape)
  assert transposeRes.shape == tuple(transposeTensorShape)
  #
  # // Now that we have the input indices in the right shape, the next step
  # // is to perform a broadcast and then compare with a constant of indices
  # // to get a one-hot-encoding of the indices
  # llvm::SmallVector<int64_t, 4> bcastTensorShape(
  #     transposeTiledType.getTensorShape().begin(),
  #     transposeTiledType.getTensorShape().end());
  bcastTensorShape = list(transposeRes.shape)
  # bcastTensorShape[bcastTensorShape.size() - 1] =
  #     groqChip->archParams_.vectorLength_;
  vectorLength = data.shape[axis]
  bcastTensorShape[len(bcastTensorShape) - 1] = vectorLength
  # auto bcastTiledType = groq::TiledMemRefType::get(bcastTensorShape,
  #     transposeTiledType.getTileAxes(), transposeTiledType.getTileSizes(),
  #     transposeTiledType.getElementType());
  # auto broadcastRes =
  #     rewriter.create<mlir::GroqBroadcastOp>(loc, bcastTiledType, transposeRes)
  #         .getResult();
  #
  broadcastRes = np.broadcast_to(transposeRes, bcastTensorShape)
  # // Create the constant to compare against
  # auto dataTensorShape = dataTiledType.getTensorShape();
  dataTensorShape = list(data.shape)
  # int64_t numAddresses =
  #     std::ceil(dataTensorShape[axis] /
  #               static_cast<float>(groqChip->archParams_.vectorLength_));
  numAddresses = int(math.ceil(dataTensorShape[axis] / vectorLength))
  # llvm::SmallVector<mlir::Attribute, DEFAULT_VECTOR_LENGTH> compValues(
  #     numAddresses * groqChip->archParams_.vectorLength_,
  #     getTypeAttr(rewriter, indexElType, 0));
  #
  # auto numElements = numAddresses * groqChip->archParams_.vectorLength_;
  numElements = numAddresses * vectorLength
  compValues = [-1] * numElements
  # for (int64_t idx = 0; idx < numElements; ++idx) {
  #   if (idx < dataTensorShape[axis]) {
  #     compValues[idx] = rewriter.getIntegerAttr(indexElType, idx);
  #   } else {
  #     // populate the remaining (i.e. pad) values with -1 so that we never
  #     // match against it
  #     compValues[idx] = rewriter.getIntegerAttr(indexElType, -1);
  #   }
  # }
  for idx in range(numElements):
    compValues[idx] = idx
  # auto compDenseVals = mlir::DenseElementsAttr::get(
  #     mlir::VectorType::get(
  #         {numAddresses * groqChip->archParams_.vectorLength_}, indexElType),
  #     llvm::makeArrayRef(compValues));
  compDenseVals = np.asarray(compValues)
  # auto compConstTiledType = groq::TiledMemRefType::get(
  #     {numAddresses}, indicesTiledType.getElementType());
  # auto compConstOp =
  #     createGroqConstantOp(loc, rewriter, compConstTiledType, compDenseVals);
  compConstOp = compDenseVals
  #
  # auto compTensorShape = bcastTensorShape;
  compTensorShape = list(bcastTensorShape)
  # compTensorShape[compTensorShape.size() - 1] = dataTensorShape[axis];
  compTensorShape[len(compTensorShape) - 1] = dataTensorShape[axis]
  # auto compTiledType = groq::TiledMemRefType::get(compTensorShape,
  #     bcastTiledType.getTileAxes(), bcastTiledType.getTileSizes(),
  #     mlir::VectorType::get(
  #         groqChip->archParams_.vectorLength_, rewriter.getIntegerType(8)));
  # return rewriter
  #     .create<mlir::GroqEqOp>(loc, compTiledType, broadcastRes, compConstOp)
  #     .getResult();
  # }
  comp = np.reshape(compDenseVals, [1] * (len(compTensorShape) - 1) + [-1])
  result = (comp == broadcastRes)
  result = result.astype(indices.dtype)
  return result

class MXMAllocResults:
  mxmResults: List[np.ndarray]
  addResults: List[np.ndarray]
  def __init__(self):
    self.mxmResults = []
    self.addResults = []

def tiledShape(shape, tiledAxis = -1):
  shape = list(shape)
  shape[tiledAxis] = 320
  return tuple(shape)

# mlir::Value mxmCompute(const GroqChip *const groqChip,
#     mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
#     mlir::Value streamInput, mlir::Value weightInput) {
def mxmCompute(streamInput, weightInput) -> np.ndarray:
  streamInput = np.asarray(streamInput)
  weightInput = np.asarray(weightInput)
  # auto streamTiledType = streamInput.getType().cast<groq::TiledMemRefType>();
  # auto weightTiledType = weightInput.getType().cast<groq::TiledMemRefType>();
  #
  numLanesPerStreamPerSuperlane_ = 16
  vectorLength_ = 320
  numSuperlane_ = vectorLength_ // numLanesPerStreamPerSuperlane_
  mxmWidthFP16ReductionFactor_ = 2
  mxmLwbInputWidth_ = {
    np.dtype('float16'): vectorLength_ // mxmWidthFP16ReductionFactor_,
    np.dtype('int8'): vectorLength_,
    np.dtype('uint8'): vectorLength_,
  }
  # auto lwbWidth = groqChip->archParams_.mxmLwbInputWidth_.at(DType::uint8);
  lwbWidth = mxmLwbInputWidth_[np.dtype('uint8')]
  #
  # auto weightShape = weightTiledType.getShape();
  # auto streamShape = streamTiledType.getShape();
  # weightShape = tiledShape(weightInput.shape)
  # streamShape = tiledShape(streamInput.shape)
  weightShape = weightInput.shape
  streamShape = streamInput.shape
  # auto weightElType = getBaseType(weightInput);
  # auto streamElType = getBaseType(streamInput);
  weightElType = weightInput.dtype
  streamEltype = streamInput.dtype
  # auto elTypeSize =
  #     std::max(weightElType.getIntOrFloatBitWidth() / BITS_PER_BYTE,
  #         streamElType.getIntOrFloatBitWidth() / BITS_PER_BYTE);
  #
  # int64_t size = weightShape.size();
  size = len(weightShape)
  # llvm::SmallVector<int64_t, 4> resultTensorShape(size, 0);
  resultTensorShape = [0] * len(weightShape)
  #
  # resultTensorShape[resultTensorShape.size() - 1] =
  #     weightShape[weightShape.size() - 2];
  resultTensorShape[len(resultTensorShape) - 1] = weightShape[len(weightShape) - 2]
  # resultTensorShape[resultTensorShape.size() - 2] =
  #     streamShape[streamShape.size() - 2];
  resultTensorShape[len(resultTensorShape) - 2] = streamShape[len(streamShape) - 2]
  #
  # assert(weightShape.size() >= streamShape.size());
  assert len(weightShape) >= len(streamShape)
  #
  # int64_t numBatchDims = 1;
  numBatchDims = 1
  # bool batching = true;
  batching = True
  # for (int64_t idx = size - 3; idx >= 0; --idx) {
  for idx in range(size - 3, -1, -1):
    # int64_t weightIdx = idx;
    weightIdx = idx
    # int64_t streamIdx = idx - (weightShape.size() - streamShape.size());
    streamIdx = idx - (len(weightShape) - len(streamShape))
    # int64_t weightDim = weightIdx >= 0 ? weightShape[weightIdx] : 1;
    weightDim = weightShape[weightIdx] if weightIdx >= 0 else 1
    # int64_t streamDim = streamIdx >= 0 ? streamShape[streamIdx] : 1;
    streamDim = streamShape[streamIdx] if streamIdx >= 0 else 1
    # if (weightDim == 1 && batching) {
    if (weightDim == 1) and batching:
      # numBatchDims++;
      numBatchDims += 1
    elif weightDim != 1:
      # } else if (weightDim != 1) {
      # batching = false;
      batching = False
    # }
    # resultTensorShape[idx] = std::max(weightDim, streamDim);
    resultTensorShape[idx] = max(weightDim, streamDim)
  # }
  #
  # numBatchDims =
  #     std::min(numBatchDims, static_cast<int64_t>(streamShape.size() - 1));
  numBatchDims = min(numBatchDims, len(streamShape) - 1)
  #
  # auto resultVecType = weightElType.getIntOrFloatBitWidth() >
  #                              streamElType.getIntOrFloatBitWidth()
  #                          ? weightTiledType.getElementType()
  #                          : streamTiledType.getElementType();
  #
  # auto resultTiledType = groq::TiledMemRefType::get(resultTensorShape,
  #     {static_cast<int64_t>(resultTensorShape.size() - 1)},
  #     weightTiledType.getTileSizes(), resultVecType);
  #
  # auto resultShape = resultTiledType.getShape();
  resultShape = tuple(resultTensorShape)
  #
  # MXMAllocResults results;
  results = MXMAllocResults()
  #
  # for (int64_t reductionIdx = 0; reductionIdx < weightShape.back();
  #      ++reductionIdx) {
  for reductionIdx in range(weightShape[-1]):
    # if (reductionIdx < weightShape.back() - 1) {
    #   results.mxmResults.push_back(
    #       createGroqAllocOp(loc, resultTiledType, rewriter).getResult());
    # }
    if reductionIdx < weightShape[-1] - 1:
      results.mxmResults.append(np.zeros(resultShape))
    # results.addResults.push_back(
    #     createGroqAllocOp(loc, resultTiledType, rewriter).getResult());
    results.addResults.append(np.zeros(resultShape))
  # }
  #
  # llvm::SmallVector<mlir::Value, 4> resultIndices(resultShape.size());
  # llvm::SmallVector<mlir::Value, 4> weightIndices(weightShape.size());
  # llvm::SmallVector<mlir::Value, 4> streamIndices(streamShape.size());
  resultIndices = [None] * len(resultShape)
  weightIndices = [None] * len(weightShape)
  streamIndices = [None] * len(streamShape)
  #
  # mlir::AffineForOp outerLoop = nullptr;
  outerLoop = None
  #
  # int64_t batchDiff = weightShape.size() - 1 - numBatchDims;
  batchDiff = len(weightShape) - 1 - numBatchDims
  # int64_t rankDiff = weightShape.size() - streamShape.size();
  rankDiff = len(weightShape) - len(streamShape)
  #
  # // Iterate over all of the batches of the weights first
  # for (int64_t idx = 0; idx < batchDiff; ++idx) {
  for idx in range(batchDiff):
    # auto loop = rewriter.create<mlir::AffineForOp>(loc, 0, weightShape[idx]);
    loop = range(0, weightShape[idx])
    # if (idx == 0) {
    if idx == 0:
      # outerLoop = loop;
      outerLoop = loop
    # }
    # rewriter.setInsertionPointToStart(loop.getBody());
    # weightIndices[idx] = loop.getInductionVar();
    weightIndices[idx] = loop
    # resultIndices[idx] = loop.getInductionVar();
    resultIndices[idx] = loop
    # if (idx - rankDiff >= 0) {
    if idx - rankDiff >= 0:
      # streamIndices[idx - rankDiff] = loop.getInductionVar();
      streamIndices[idx - rankDiff] = loop
    # }
  # }
  #
  # for (int64_t idx = batchDiff;
  #      idx < static_cast<int64_t>(weightIndices.size() - 2); ++idx) {
  for idx in range(batchDiff, len(weightIndices) - 2):
    # weightIndices[idx] = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
    weightIndices[idx] = 0
  # }
  #
  # // Next add the loops for the final output
  # auto outputLoop =
  #     rewriter.create<mlir::AffineForOp>(loc, 0, resultShape.back());
  outputLoop = range(0, resultShape[-1])
  # rewriter.setInsertionPointToStart(outputLoop.getBody());
  # resultIndices[resultIndices.size() - 1] = outputLoop.getInductionVar();
  resultIndices[len(resultIndices) - 1] = outputLoop
  # weightIndices[weightIndices.size() - 2] = outputLoop.getInductionVar();
  weightIndices[len(weightIndices) - 2] = outputLoop
  #
  # // Next add the reduction loop
  # // Note: for SSA this needs to be unrolled
  # for (int64_t reductionIdx = 0; reductionIdx < weightShape.back();
  #      ++reductionIdx) {
  for reductionIdx in range(weightShape[-1]):
    # // Insert byte index loop
    # auto reductionConstIdx =
    #     rewriter.create<mlir::arith::ConstantIndexOp>(loc, reductionIdx);
    # weightIndices[weightIndices.size() - 1] = reductionConstIdx;
    weightIndices[-1] = reductionIdx
    # streamIndices[streamIndices.size() - 1] = reductionConstIdx;
    streamIndices[-1] = reductionIdx
    #
    # auto byteLoop = rewriter.create<mlir::AffineForOp>(loc, 0, elTypeSize);
    # rewriter.setInsertionPointToStart(byteLoop.getBody());
    # auto byteIndex = byteLoop.getInductionVar();
    #
    # // If we're on the last iteration of the output then we have a partial read
    # auto d0 = rewriter.getAffineDimExpr(0);
    # mlir::AffineExpr ifExpr{d0 + 1 - resultShape.back()};
    # auto ifSet = mlir::IntegerSet::get(1, 0, ifExpr, true);
    # auto ifOp = rewriter.create<mlir::AffineIfOp>(
    #     loc, ifSet, mlir::ValueRange{resultIndices.back()}, true);
    #
    # auto fullSize = weightShape[weightShape.size() - 2];
    fullSize = weightShape[-2]
    # auto vectorLength = groqChip->archParams_.vectorLength_;
    vectorLength = vectorLength_
    # auto sizeInLastIter = fullSize - (resultShape.back() - 1) * vectorLength;
    sizeInLastIter = fullSize - (resultShape[-1] - 1) * vectorLength
    #
    # rewriter.setInsertionPointToStart(ifOp.getThenBlock());
    # // Load/Install partial weights
    # auto numBlocks = sizeInLastIter / lwbWidth;
    numBlocks = sizeInLastIter // lwbWidth
    # auto partialReads = sizeInLastIter % lwbWidth;
    partialReads = sizeInLastIter % lwbWidth
    #
    # auto partialIW = installWeights(groqChip, rewriter, loc, weightInput,
    #     weightIndices, byteIndex, numBlocks, partialReads);
    partialIW = installWeights(weightInput, weightIndices, numBlocks, partialReads)
    # activationLoop(groqChip, rewriter, loc, streamInput, partialIW,
    #     streamIndices, resultIndices, byteIndex, results, numBatchDims,
    #     reductionIdx);
    activationLoop(streamInput, partialIW, streamIndices, resultIndices, results, numBatchDims, reductionIdx)
    #
    # rewriter.setInsertionPointToStart(ifOp.getElseBlock());
    # // Load full weights
    # numBlocks = vectorLength / lwbWidth;
    # partialReads = 0;
    #
    # auto fullIW = installWeights(groqChip, rewriter, loc, weightInput,
    #     weightIndices, byteIndex, numBlocks, partialReads);
    #
    # activationLoop(groqChip, rewriter, loc, streamInput, fullIW, streamIndices,
    #     resultIndices, byteIndex, results, numBatchDims, reductionIdx);
    # rewriter.setInsertionPointAfter(byteLoop);
    breakpoint()
    x = 42
  # }
  # if (outerLoop != nullptr) {
  #   rewriter.setInsertionPointAfter(outerLoop);
  # } else {
  #   rewriter.setInsertionPointAfter(outputLoop);
  # }
  # return results.addResults.back();
  # }

def installWeights(weights, weightIndices, numBlocks, partialReads):
  weights = np.asarray(weights)
  # auto vectorLength = groqChip->archParams_.vectorLength_;
  # auto lwbWidth = groqChip->archParams_.mxmLwbInputWidth_.at(DType::uint8);
  # auto weightShape = weights.getType().cast<groq::TiledMemRefType>().getShape();
  weightShape = weights.shape
  #
  # // collect shape information for the submatrix
  # const int32_t numWeights = numBlocks * lwbWidth + partialReads;
  # auto bufferMemrefType = groq::TiledMemRefType::get(
  #     {numWeights}, mlir::VectorType::get(groqChip->archParams_.vectorLength_,
  #                       rewriter.getIntegerType(8)));
  # auto bufferMemRef =
  #     createGroqAllocOp(loc, bufferMemrefType, rewriter, lwbWidth).getResult();
  # llvm::SmallVector<bool, DEFAULT_LWB_INPUT_WIDTH_I8> bitmask(lwbWidth, true);
  #
  # auto tempIndices = weightIndices;
  # auto flatReadIdxMap = getAffineRavelMap(rewriter, weightShape);
  #
  # auto readVecType =
  #     mlir::VectorType::get(vectorLength, rewriter.getIntegerType(8));
  #
  # if (getBaseType(weights).getIntOrFloatBitWidth() == 8) {
  #   byteIndex = nullptr;
  # }
  #
  # if (numBlocks != 0) {
  #   auto lwbLoop = rewriter.create<mlir::AffineForOp>(loc, 0, numBlocks, 1);
  #   rewriter.setInsertionPointToStart(lwbLoop.getBody());
  #   llvm::SmallVector<mlir::Value, DEFAULT_LWB_INPUT_WIDTH_I8> lwbReads(
  #       lwbWidth);
  #   for (uint64_t readIdx = 0; readIdx < lwbWidth; ++readIdx) {
  #     mlir::AffineExpr readIdxExpr =
  #         rewriter.getAffineDimExpr(0) * vectorLength +
  #         rewriter.getAffineDimExpr(1) * lwbWidth + readIdx;
  #     auto readIdxMap =
  #         mlir::AffineMap::get(2, 0, readIdxExpr, rewriter.getContext());
  #     tempIndices[tempIndices.size() - 2] =
  #         rewriter.create<mlir::AffineApplyOp>(loc, readIdxMap,
  #             mlir::ValueRange{weightIndices[weightIndices.size() - 2],
  #                 lwbLoop.getInductionVar()});
  #
  #     auto flatReadIdx =
  #         rewriter.create<mlir::AffineApplyOp>(loc, flatReadIdxMap, tempIndices)
  #             .getResult();
  #
  #     lwbReads[readIdx] = rewriter
  #                             .create<mlir::GroqReadOp>(loc, readVecType,
  #                                 weights, flatReadIdx, byteIndex)
  #                             .getResult();
  #   }
  #   rewriter.create<mlir::GroqLoadWeightBufferOp>(loc, bufferMemRef,
  #       mlir::ValueRange{lwbReads}, rewriter.getBoolArrayAttr(bitmask));
  #   rewriter.setInsertionPointAfter(lwbLoop);
  # }
  #
  # if (partialReads != 0) {
  #   llvm::SmallVector<mlir::Value, DEFAULT_LWB_INPUT_WIDTH_I8> lwbReads(
  #       partialReads);
  #   for (int64_t readIdx = 0; readIdx < partialReads; ++readIdx) {
  #     mlir::AffineExpr readIdxExpr =
  #         rewriter.getAffineDimExpr(0) * vectorLength + numBlocks * lwbWidth +
  #         readIdx;
  #     auto readIdxMap =
  #         mlir::AffineMap::get(1, 0, readIdxExpr, rewriter.getContext());
  #     tempIndices[tempIndices.size() - 2] =
  #         rewriter.create<mlir::AffineApplyOp>(loc, readIdxMap,
  #             mlir::ValueRange{weightIndices[weightIndices.size() - 2]});
  #     auto flatReadIdx =
  #         rewriter.create<mlir::AffineApplyOp>(loc, flatReadIdxMap, tempIndices)
  #             .getResult();
  #
  #     lwbReads[readIdx] = rewriter
  #                             .create<mlir::GroqReadOp>(loc, readVecType,
  #                                 weights, flatReadIdx, byteIndex)
  #                             .getResult();
  #   }
  #   rewriter.create<mlir::GroqLoadWeightBufferOp>(loc, bufferMemRef,
  #       mlir::ValueRange{lwbReads}, rewriter.getBoolArrayAttr(bitmask));
  # }
  #
  # llvm::SmallVector<int32_t, 2> sama = {0, 1};
  #
  # return rewriter
  #     .create<mlir::GroqInstallWeightsOp>(loc, rewriter.getNoneType(),
  #         bufferMemRef, nullptr,                        /* planes */
  #         mlir::BoolAttr::get(loc.getContext(), false), /* continuous */
  #         mlir::BoolAttr::get(loc.getContext(), false), /* slt enable */
  #         nullptr,                                      /* slt_size */
  #         nullptr,                                      /* float */
  #         nullptr,                                      /* weights signed */
  #         nullptr,                                      /* activations signed */
  #         nullptr,                                      /* wint4 */
  #         rewriter.getI32ArrayAttr(sama),     /* mlir::arithmetic mode */
  #         nullptr,                            /* sa1 */
  #         nullptr,                            /* if */
  #         rewriter.getBoolArrayAttr(bitmask), /* bitmask*/
  #         nullptr,                            /* sg16 */
  #         nullptr,                            /* time */
  #         nullptr)                            /* is_i8_to_fp16_transition */
  #     .getResult();
  return weights

# void activationLoop(const GroqChip *const groqChip,
#     mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
#     mlir::Value streamInput, mlir::Value iwResult,
#     llvm::SmallVector<mlir::Value, 4> streamIndices,
#     llvm::SmallVector<mlir::Value, 4> resultIndices, mlir::Value byteIndex,
#     MXMAllocResults results, int64_t numBatchDims, int64_t reductionIdx) {
def activationLoop(streamInput, iwResult, streamIndices, resultIndices, results: MXMAllocResults, numBatchDims: int, reductionIdx: int):
  streamInput = np.asarray(streamInput)
  # auto inputByteIndex = getBaseType(streamInput).getIntOrFloatBitWidth() == 8
  #                           ? nullptr
  #                           : byteIndex;
  # auto streamTiledType = streamInput.getType().cast<groq::TiledMemRefType>();
  # auto streamShape = streamTiledType.getShape();
  streamShape = streamInput.shape
  # auto resultTiledType =
  #     results.addResults.back().getType().cast<groq::TiledMemRefType>();
  # auto resultShape = resultTiledType.getShape();
  resultShape = results.addResults[-1].shape
  #
  # auto rwType = mlir::VectorType::get(
  #     groqChip->archParams_.vectorLength_, rewriter.getIntegerType(8));
  #
  # for (uint64_t idx = streamShape.size() - 1 - numBatchDims;
  #      idx < streamShape.size() - 1; ++idx) {
  for idx in range(len(streamShape) - 1 - numBatchDims, len(streamShape) - 1):
    # auto loop = rewriter.create<mlir::AffineForOp>(loc, 0, streamShape[idx], 1);
    loop = range(0, streamShape[idx])
    # rewriter.setInsertionPointToStart(loop.getBody());
    #
    # streamIndices[idx] = loop.getInductionVar();
    streamIndices[idx] = loop
    # resultIndices[resultIndices.size() - 1 - (streamShape.size() - 1) + idx] =
    #     loop.getInductionVar();
    resultIndices[len(resultIndices) - 1 - (len(streamShape) - 1) + idx] = loop
  # }
  #
  # auto streamAffineMap = getAffineRavelMap(rewriter, streamShape);
  # auto resultAffineMap = getAffineRavelMap(rewriter, resultShape);
  #
  # auto mxmInputIdx =
  #     rewriter.create<mlir::AffineApplyOp>(loc, streamAffineMap, streamIndices)
  #         .getResult();
  # auto mxmInputRead = rewriter
  #                         .create<mlir::GroqReadOp>(loc, rwType, streamInput,
  #                             mxmInputIdx, inputByteIndex)
  #                         .getResult();
  # auto mxmResult = rewriter
  #                      .create<mlir::GroqMXMOp>(loc, rwType, iwResult,
  #                          mxmInputRead, nullptr, nullptr)
  #                      .getResult();
  #
  # auto resultBuffer = reductionIdx == 0 ? results.addResults[0]
  #                                       : results.mxmResults[reductionIdx - 1];
  #
  # auto resultIdx =
  #     rewriter.create<mlir::AffineApplyOp>(loc, resultAffineMap, resultIndices)
  #         .getResult();
  # auto accumNull = rewriter
  #                      .create<mlir::GroqAccumulateOp>(loc, rwType, mxmResult,
  #                          nullptr, nullptr, true, nullptr)
  #                      .getResult();
  # rewriter.create<mlir::GroqWriteOp>(
  #     loc, resultBuffer, resultIdx, byteIndex, accumNull);
  #
  # if (reductionIdx != 0) {
  #   createVXMBinaryOpWithWrapper<mlir::GroqAddSatOp>(rewriter, loc,
  #       results.addResults[reductionIdx - 1], resultIdx, rwType,
  #       results.mxmResults[reductionIdx - 1], resultIdx, rwType,
  #       results.addResults[reductionIdx], resultIdx, rwType, byteIndex,
  #       byteIndex, byteIndex);
  # }
  # }

# mlir::AffineExpr getAffineRavelIndexExpr(mlir::OpBuilder &rewriter,
#     llvm::ArrayRef<int64_t> shape, llvm::ArrayRef<mlir::AffineExpr> exprs) {
def getAffineRavelIndexExpr(shape, exprs):
  # uint64_t prod = 1;
  prod = 1
  # mlir::AffineExpr linearIndex = rewriter.getAffineConstantExpr(0);
  linearIndex = 0
  # for (unsigned i = shape.size(); i-- > 0;) {
  for i in range(len(shape) - 1, -1, -1):
    # if (i + 1 < shape.size()) {
    #   prod *= shape[i + 1];
    # }
    if i + 1 < len(shape):
      prod *= shape[i + 1]
    # linearIndex = exprs[i] * prod + linearIndex;
    linearIndex = exprs[i] * prod + linearIndex
  # }
  #
  # return linearIndex;
  return linearIndex
  # }

# // MXM Transpose always transposes the last 2 axes
def MXMTranspose(operand, outputShape) -> np.ndarray:
  operand = np.asarray(operand)
  return np.swapaxes(operand, -1, -2)

# >>> axis3 = 0
# >>> hot3 = npnd.one_hot(indices3, data3.shape[axis3], dtype=data.dtype, axis=axis3)
# >>> mask3 = (shape_lib.expand(hot3.sum(axis3 + 1, keepdims=True) > 1, hot3.shape) == 0)
# >>> result3 = shape_lib.expand( (mask3 * hot3 * np.expand_dims(updates3, axis3)).sum(axis3 + 1), data3.shape )
# >>> final3 = (shape_lib.expand( (mask3 * hot3).sum(axis3 + 1), data3.shape) == 0) * data3 + result3
# array([[[106., 107.,   0.],
#         [  0., 110., 111.],
#         [  0.,   0.,   0.],
#         [  0.,   0.,   0.]],
#
#        [[  0.,   0., 108.],
#         [109.,   0.,   0.],
#         [  0.,   0.,   0.],
#         [  0.,   0.,   0.]]])
# >>> data3
# array([[[ 0.1,  1.1,  2.1],
#         [ 3.1,  4.1,  5.1],
#         [ 6.1,  7.1,  8.1],
#         [ 9.1, 10.1, 11.1]],
#
#        [[12.1, 13.1, 14.1],
#         [15.1, 16.1, 17.1],
#         [18.1, 19.1, 20.1],
#         [21.1, 22.1, 23.1]]])
# >>> updates3
# array([[[100, 101, 102],
#         [103, 104, 105]],
#
#        [[106, 107, 108],
#         [109, 110, 111]]])
# >>> indices3
# array([[[0, 0, 1],
#         [1, 0, 0]],
#
#        [[0, 0, 1],
#         [1, 0, 0]]])

# this matches torch.scatter! yay!
def scatter(tensor, indices, updates, axis=0) -> np.ndarray:
  tensor = np.asarray(tensor).astype(float)
  indices = np.asarray(indices).astype(int)
  updates = np.asarray(updates).astype(float)
  hot = one_hot_lib.one_hot(indices, tensor.shape[axis], dtype=tensor.dtype, axis=axis)
  # zero out duplicates
  dedup = shape_lib.expand(hot.sum(axis + 1, keepdims=True) > 1, hot.shape)
  hot = dedup * hot
  # scatter updates
  updates = np.expand_dims(updates, axis)
  updates = (hot * updates).sum(axis + 1)
  updates = shape_lib.expand(updates, tensor.shape)
  # determine tensor mask
  mask = shape_lib.expand(hot.sum(axis + 1), tensor.shape) == 0
  return mask * tensor + updates

# reference implementation of gather_nd.
def gather_nd_ref(params, indices, batch_dims=0) -> np.ndarray:
  assert batch_dims == 0
  params = np.asarray(params)
  indices = np.asarray(indices)
  return params[tuple(indices.T)]

def gather_nd_info(params, indices, batch_dims=0):
  params = np.asarray(params)
  indices = np.asarray(indices)
  # Calculate the number of dimensions in indices
  slice_dim = indices.shape[-1]
  if slice_dim > np.ndim(params):
    return errors.invalid_argument(
        "index innermost dimension length must be <= params rank; saw: ",
        slice_dim, " vs. ", np.ndim(params))
  outer_shape = indices.shape[:-1]
  # prn("outer_shape == ", outer_shape)
  inner_shape = params.shape[slice_dim:]
  # prn("inner_shape == ", inner_shape)
  result_shape = outer_shape + inner_shape

  # Calculate the number of elements that make up each slice of the
  # tensor.
  slice_size = int(np.prod(inner_shape)) # 1 if inner_shape is empty
  # prn("slice_size == ", slice_size)
  # Calculate the number of slices we'll be selecting.
  num_slices = int(np.prod(params.shape)) // slice_size
  # prn("num_slices == ", num_slices)
  # Reshape the incoming tensor into (num_slices, slice_size).
  params_shape = (num_slices, slice_size)

  # Calculate the 1-dimensional indices necessary to select
  # the correct slices.
  strides_shape = params_shape[:slice_dim]
  # indices_mat = shape_lib.flat_nd_indices(indices, strides_shape)
  #assert tuple(strides_shape) == tuple(indices.shape[:-1])

  return GatherNdInfo(slice_dim,
                      outer_shape,
                      inner_shape,
                      result_shape,
                      slice_size,
                      num_slices,
                      params_shape,
                      strides_shape)

@dataclasses.dataclass
class GatherNdInfo:
  slice_dim: int
  outer_shape: Tuple[int, ...]
  inner_shape: Tuple[int, ...]
  result_shape: Tuple[int, ...]
  slice_size: int
  num_slices: int
  params_shape: Tuple[int, ...]
  strides_shape: Tuple[int, ...]

def gather_nd(params, indices, batch_dims=0) -> np.ndarray:
  params = np.asarray(params)
  indices = np.asarray(indices)
  if not np.issubdtype(params.dtype, np.number):
    return gather_nd_generic(params, indices, batch_dims=batch_dims)
  if batch_dims > 0:
    return gather_nd_batched(params, indices, batch_dims=batch_dims)
  if np.ndim(params) < 1:
    return errors.invalid_argument("params must be at least a vector")
  if np.ndim(indices) < 1:
    return errors.invalid_argument("indices must be at least a vector")
  # Calculate the number of dimensions in indices
  slice_dim = indices.shape[-1]
  if slice_dim > np.ndim(params):
    return errors.invalid_argument(
        "index innermost dimension length must be <= params rank; saw: ",
        slice_dim, " vs. ", np.ndim(params))
  outer_shape = indices.shape[:-1]
  # prn("outer_shape == ", outer_shape)
  inner_shape = params.shape[slice_dim:]
  # prn("inner_shape == ", inner_shape)
  result_shape = outer_shape + inner_shape
  # prn("result_shape == ", result_shape)
  # Calculate the number of elements that make up each slice of the
  # tensor.
  slice_size = int(np.prod(inner_shape)) # 1 if inner_shape is empty
  # prn("slice_size == ", slice_size)
  # Calculate the number of slices we'll be selecting.
  num_slices = int(np.prod(params.shape)) // slice_size
  # prn("num_slices == ", num_slices)
  # Reshape the incoming tensor into (num_slices, slice_size).
  params_mat = params.reshape((num_slices, slice_size))
  # prn("params_mat.shape == ", params_mat.shape)
  # Calculate the 1-dimensional indices necessary to select
  # the correct slices.
  strides_shape = params.shape[:slice_dim]
  # indices_mat = shape_lib.flat_nd_indices(indices, strides_shape)
  #assert tuple(strides_shape) == tuple(indices.shape[:-1])
  indices_mat = shape_lib.flat_nd_indices(indices, strides_shape)
  # prn("indices_mat.shape == ", indices_mat.shape)
  # Select the slices we want, via onehot-matmul.
  hot = one_hot_lib.one_hot(indices_mat, num_slices)
  # prn("hot.shape == ", hot.shape)
  result = hot @ params_mat
  # prn("result.shape == ", result.shape)
  # Reshape the result back to the expected shape.
  out = result.reshape(result_shape)
  # prn("out.shape == ", out.shape)
  # return out.astype(params.dtype)
  return out

def gather_nd_generic(params, indices, batch_dims) -> np.ndarray:
  # if params contains non-numbers, handle it specially, since it can't be multiplied
  # against onehot matrices.
  items = params.flat[:].tolist()
  ids = np.arange(np.prod(params.shape)).reshape(params.shape)
  out = gather_nd(ids, indices, batch_dims=batch_dims)
  final = np.asarray([items[i] for i in out.flat[:].tolist()]).reshape(out.shape)
  return final

def gather_nd_batched(params, indices, batch_dims) -> np.ndarray:
  # TODO: Clean this up. I have a feeling it can be unified with the
  # logic in gather_nd.
  #
  # These shapes came from
  # https://www.tensorflow.org/api_docs/python/tf/gather_nd
  index_depth = indices.shape[-1]
  batch_shape = indices.shape[:batch_dims]
  assert params.shape[:batch_dims] == batch_shape
  outer_shape = indices.shape[batch_dims:-1]
  assert index_depth <= np.ndim(params)
  inner_shape = params.shape[batch_dims + index_depth:]
  result_shape = batch_shape + outer_shape + inner_shape
  batched_indices = add_batch_indices(params, indices, batch_dims)
  result = gather_nd(params, batched_indices, batch_dims=0)
  return result.reshape(result_shape)

def add_batch_indices_old(params, indices, batch_dims) -> np.ndarray:
  # TODO: I'm only confident that the batch_dims==1 case works.
  # assert batch_dims == 1, "batch_dims > 1 not yet implemented"
  indices = np.asarray(indices)
  ind = np.arange(indices.shape[0])
  #ind = ind.reshape(indices.shape)
  # shape = (-1,) + tuple(1 for i in range(len(indices.shape) - 1))
  # ind = ind.reshape(shape)
  ind = np.expand_dims(ind, list(range(batch_dims, len(indices.shape))))
  ind = np.concatenate([ind, indices], -1)
  #ind = np.concatenate([indices[...,:-1], ind, indices[...,-1:]], -1)
  # ind = np.expand_dims(ind, -2)
  return ind

def batch_gather_nd_indices(indices, batch_dims=1) -> np.ndarray:
  indices = np.asarray(indices)
  indices_nd = list(shape_lib.ndshape(indices.shape, dtype=indices.dtype))[0:batch_dims]
  indices_nd.append(indices)
  return np.stack(indices_nd, axis=-1)

def add_batch_indices(params, indices, batch_dims) -> np.ndarray:
  shape = list(np.shape(indices))
  shape[-1] = 1
  shapes = shape_lib.ndshape(shape)
  for i in range(batch_dims):
    indices = np.concatenate([shapes[i], indices], -1)
  return indices

#add_batch_indices = add_batch_indices_old

def take_along_axis(params, indices, axis) -> np.ndarray:
  params = np.asanyarray(params)
  indices = np.asanyarray(indices)
  indices = shape_lib.ndindices(indices, axis=axis)
  return gather_nd(params, indices)

take = gather