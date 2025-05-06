#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>

namespace needle {
namespace cuda {

#define BASE_THREAD_NUM 256

#define TILE 4
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

struct CudaArray {
  CudaArray(const size_t size) {
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    this->size = size;
  }
  ~CudaArray() { cudaFree(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }
  
  scalar_t* ptr;
  size_t size;
};

struct CudaDims {
  dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
  /**
   * Utility function to get cuda dimensions for 1D call
   */
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  int32_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<int32_t>& x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}

////////////////////////////////////////////////////////////////////////////////
// Fill call
////////////////////////////////////////////////////////////////////////////////

__global__ void FillKernel(scalar_t* out, scalar_t val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = val;
}

void Fill(CudaArray* out, scalar_t val) {
  CudaDims dim = CudaOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

// Untility function to convert contiguous index i to memory location from strides



__global__ void CompactKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the 
   * non-compact input a, to the corresponding item (at location gid) in the compact array out.
   * 
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of out array
   *   offset: offset of out array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN SOLUTION
  int indices[MAX_VEC_SIZE];
  size_t idx = gid;

  for (int i = shape.size - 1; i >= 0; --i) {
    indices[i] = idx % shape.data[i];
    idx /= shape.size > 0 ? shape.data[i] : 1;
  }

  size_t a_index = offset;
  for (int i = 0; i < shape.size; ++i) {
    a_index += indices[i] * strides.data[i];
  }

  out[gid] = a[a_index];
  /// END SOLUTION
}

void Compact(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory.  Unlike the C++ version, in CUDA this will primarily call the 
   * relevant CUDA kernel.  In this case, we illustrate how you should set this up (i.e., we give 
   * you the code for this fuction, and also the prototype for the CompactKernel() function).  For
   * the functions after this, however, you'll need to define these kernels as you see fit to 
   * execute the underlying function.
   * 
   * Args:
   *   a: non-compact represntation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   */

  // Nothing needs to be added here
  CudaDims dim = CudaOneDim(out->size);
  CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
}

__global__ void EwiseSetitemKernel(const scalar_t* a, scalar_t* out, CudaVec shape,
                                   CudaVec strides, size_t offset) {

  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  int indices[MAX_VEC_SIZE];
  size_t idx = gid;

  for (int i = shape.size - 1; i >= 0; --i) {
    indices[i] = idx % shape.data[i];
    idx /= shape.data[i];
  }

  size_t out_index = offset;
  for (int i = 0; i < shape.size; ++i) {
    out_index += indices[i] * strides.data[i];
  }
  
  out[out_index] = a[gid];

}

void EwiseSetitem(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array using CUDA.  Yyou will most likely want to implement a
   * EwiseSetitemKernel() function, similar to those above, that will do the actual work.
   * 
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  EwiseSetitemKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, VecToCuda(shape),
                                         VecToCuda(strides), offset);
  /// END SOLUTION
}

__global__ void ScalarSetitemKernel(size_t size, scalar_t val, scalar_t* out, CudaVec shape,
                                    CudaVec strides, size_t offset) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid >= size) return;
  
  int indices[MAX_VEC_SIZE];
  size_t idx = gid;

  for (int i = shape.size - 1; i >= 0; --i) {
    indices[i] = idx % shape.data[i];
    idx /= shape.size > 0 ? shape.data[i] : 1;
  }

  size_t out_index = offset;
  for (int i = 0; i < shape.size; ++i) {
    out_index += indices[i] * strides.data[i];
  }

  out[out_index] = val;
}

void ScalarSetitem(size_t size, scalar_t val, CudaArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   * 
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the 
   *         product of items in shape, but covenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  ScalarSetitemKernel<<<dim.grid, dim.block>>>(size, val , out->ptr, VecToCuda(shape),
                                         VecToCuda(strides), offset);
  /// END SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////


__global__ void EwiseAddKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + b[gid];
}

void EwiseAdd(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Add together two CUDA arrays.
   * Args:
   *   a: Input array 'a' to be added
   *   b: Input array 'b' to be added
   *   out: Output array to store the result of 'a + b'
   */
  CudaDims dim = CudaOneDim(out->size);

  // Kernel will execute on 'dim.grid' blocks, each containing 'dim.block' threads.
  EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarAddKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + val;
}

void ScalarAdd(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Add a scalar value to every element of a CUDA array.
   * Args:
   *   a: Input array 'a'
   *   val: Scalar value to be added
   *   out: Output array to store the result of 'a + val'
   */
  CudaDims dim = CudaOneDim(out->size);

  // Launch the ScalarAddKernel that will add the scalar 'val' to each element of array 'a', 
  // and store the result in array 'out'.
  ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

/**
 * In the code the follows, use the above template to create analogous elementise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */


////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

template<typename T, typename BinaryOp>
__global__ void EwiseOpKernel(const T* a, const T* b, T* out, size_t size, BinaryOp op) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    out[idx] = op(a[idx], b[idx]);
  }
}

template<typename T, typename BinaryOp>
__global__ void ScalarOpKernel(const T* a, T scalar, T* out, size_t size, BinaryOp op) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    out[idx] = op(a[idx], scalar);
  }
}

template<typename T, typename UnaryOp>
__global__ void UnaryOpKernel(const T* a, T* out, size_t size, UnaryOp op) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    out[idx] = op(a[idx]);
  }
}

struct MulOp {
  template<typename T>
  __device__ T operator()(const T& a, const T& b) const {
    return a * b;
  }
};

struct DivOp {
  template<typename T>
  __device__ T operator()(const T& a, const T& b) const {
    return a / b;
  }
};

struct PowerOp {
  template<typename T>
  __device__ T operator()(const T& a, const T& b) const {
    if constexpr (std::is_same_v<T, float>) {
      return powf(a, b);
    } else {
      return pow(a, b);
    }
  }
};

struct MaximumOp {
  template<typename T>
  __device__ T operator()(const T& a, const T& b) const {
    return a >= b ? a : b;
  }
};

struct EqOp {
  template<typename T>
  __device__ T operator()(const T& a, const T& b) const {
    return a == b;
  }
};

struct GeOp {
  template<typename T>
  __device__ T operator()(const T& a, const T& b) const {
    return a >= b;
  }
};

struct LogOp {
  template<typename T>
  __device__ T operator()(const T& a) const {
    if constexpr (std::is_same_v<T, float>) {
      return logf(a);
    } else {
      return log(a);
    }
  }
};

struct ExpOp {
  template<typename T>
  __device__ T operator()(const T& a) const {
    if constexpr (std::is_same_v<T, float>) {
      return expf(a);
    } else {
      return exp(a);
    }
  }
};

struct TanhOp {
  template<typename T>
  __device__ T operator()(const T& a) const {
    if constexpr (std::is_same_v<T, float>) {
      return tanhf(a);
    } else {
      return tanh(a);
    }
  }
};

void EwiseMul(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseOpKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size, MulOp());
}

void ScalarMul(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarOpKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size, MulOp());
}

void EwiseDiv(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseOpKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size, DivOp());
}

void ScalarDiv(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarOpKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size, DivOp());
}

void ScalarPower(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarOpKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size, PowerOp());
}

void EwiseMaximum(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseOpKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size, MaximumOp());
}

void ScalarMaximum(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarOpKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size, MaximumOp());
}

void EwiseEq(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseOpKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size, EqOp());
}

void ScalarEq(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarOpKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size, EqOp());
}

void EwiseGe(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseOpKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size, GeOp());
}

void ScalarGe(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarOpKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size, GeOp());
}

void EwiseLog(const CudaArray& a, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  UnaryOpKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, LogOp());
}

void EwiseExp(const CudaArray& a, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  UnaryOpKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, ExpOp());
}

void EwiseTanh(const CudaArray& a, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  UnaryOpKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, TanhOp());
}

// __global__ void MatmulKernel(
//   const scalar_t* __restrict__ A,
//   const scalar_t* __restrict__ B,
//   scalar_t* __restrict__ C,
//   uint32_t M, uint32_t N, uint32_t P)
// {
//   // 当前线程的行列索引
//   uint32_t bx = blockIdx.x;
//   uint32_t by = blockIdx.y;
//   uint32_t tx = threadIdx.x;
//   uint32_t ty = threadIdx.y;

//   // 共享内存缓存 A 和 B 的 tile
//   __shared__ scalar_t As[TILE_SIZE][TILE_SIZE];
//   __shared__ scalar_t Bs[TILE_SIZE][TILE_SIZE];

//   // 输出寄存器初始化
//   scalar_t acc[TILE_SIZE / 4][TILE_SIZE / 4] = {};

//   // 循环遍历所有 K 分块
//   for (uint32_t k = 0; k < N; k += TILE_SIZE) {
//     // 加载 A[k:k+TILE_SIZE] 到共享内存
//     uint32_t row_a = by * TILE_SIZE + ty;
//     uint32_t col_a = k + tx;
//     if (row_a < M && col_a < N)
//       As[ty][tx] = A[row_a * N + col_a];
//     else
//       As[ty][tx] = 0;

//     // 加载 B[k:k+TILE_SIZE] 到共享内存
//     uint32_t row_b = k + ty;
//     uint32_t col_b = bx * TILE_SIZE + tx;
//     if (row_b < N && col_b < P)
//       Bs[ty][tx] = B[row_b * P + col_b];
//     else
//       Bs[ty][tx] = 0;

//     __syncthreads();

//     //寄存器分块计算
//     #pragma unroll
//     for (uint32_t i = 0; i < TILE_SIZE / 4; ++i) {
//       #pragma unroll
//       for (uint32_t j = 0; j < TILE_SIZE / 4; ++j) {
//         #pragma unroll
//         for (uint32_t l = 0; l < TILE_SIZE; ++l) {
//           acc[i][j] += As[ty][l] * Bs[l][tx];
//         }
//       }
//     }

//     __syncthreads();
//   }

//   // 写回结果
//   uint32_t row_c = by * TILE_SIZE + ty;
//   uint32_t col_c = bx * TILE_SIZE + tx;

//   #pragma unroll
//   for (uint32_t i = 0; i < TILE_SIZE / 4; ++i) {
//     #pragma unroll
//     for (uint32_t j = 0; j < TILE_SIZE / 4; ++j) {
//       uint32_t r = row_c + i;
//       uint32_t c = col_c + j;
//       if (r < M && c < P) {
//         C[r * P + c] = acc[i][j];
//       }
//     }
//   }
// }

__global__ void MatmulKernel(
  const scalar_t* __restrict__ a,
  const scalar_t* __restrict__ b,
  scalar_t* __restrict__ out,
  int M, int N, int P
)
{
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < M && j < P) {
    scalar_t sum = 0;
    for (int k = 0; k < N; ++k) {
      sum += a[i * N + k] * b[k * P + j];
    }
    out[i * P + j] = sum;
  }
}

void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
            uint32_t P) {
  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix.  You will want to look
   * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
   * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
   * over (i,j) entries in the output array.  However, to really get the full benefit of this
   * problem, we would encourage you to use cooperative fetching, shared memory register tiling, 
   * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
   * the CPU backend, here you should implement a single function that works across all size
   * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
   * implementations, this function here will largely just set up the kernel call, and you should
   * implement the logic in a separate MatmulKernel() call.
   * 
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: comapct 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   M: rows of a / out
   *   N: columns of a / rows of b
   *   P: columns of b / out
   */

  /// BEGIN SOLUTION
  // 设置每个 block 为 16x16 线程
  dim3 threads(16, 16);
  // 计算需要多少个 block（向上取整）
  dim3 blocks((P + 15) / 16, (M + 15) / 16);
  MatmulKernel<<<blocks, threads>>>(a.ptr, b.ptr, out->ptr, M, N, P);
  cudaDeviceSynchronize();
  /// END SOLUTION
}


// void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
//             uint32_t P) {
//   /**
//    * Multiply two (compact) matrices into an output (also comapct) matrix.  You will want to look
//    * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
//    * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
//    * over (i,j) entries in the output array.  However, to really get the full benefit of this
//    * problem, we would encourage you to use cooperative fetching, shared memory register tiling, 
//    * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
//    * the CPU backend, here you should implement a single function that works across all size
//    * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
//    * implementations, this function here will largely just set up the kernel call, and you should
//    * implement the logic in a separate MatmulKernel() call.
//    * 
//    *
//    * Args:
//    *   a: compact 2D array of size m x n
//    *   b: comapct 2D array of size n x p
//    *   out: compact 2D array of size m x p to write the output to
//    *   M: rows of a / out
//    *   N: columns of a / rows of b
//    *   P: columns of b / out
//    */

//   /// BEGIN SOLUTION
//   dim3 threads(TILE_SIZE, TILE_SIZE);
//   dim3 blocks((P + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

//   MatmulKernel<<<blocks, threads>>>(a.ptr, b.ptr, out->ptr, M, N, P);

//   cudaDeviceSynchronize();
//   /// END SOLUTION
// }

////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////
__global__ void ReduceMaxKernel(const scalar_t* a, scalar_t* out, size_t size, size_t reduce_size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    scalar_t max = a[idx * reduce_size];
    for (size_t j = 0; j < reduce_size; ++j) {
      scalar_t val = a[idx * reduce_size + j];
      if (val > max) {
        max = val;
      }
    }
    out[idx] = max;
  }
}

void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even though it is inefficient,
   * for simplicity you can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  ReduceMaxKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, reduce_size);
  /// END SOLUTION
}

__global__ void ReduceSumKernel(const scalar_t* a, scalar_t* out, size_t size, size_t reduce_size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    scalar_t sum = 0;
    for (size_t j = 0; j < reduce_size; ++j) {
      sum += a[idx * reduce_size + j];
    }
    out[idx] = sum;
  }
}

void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking summation over `reduce_size` contiguous blocks.  Again, for simplicity you 
   * can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  ReduceSumKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, reduce_size);
  /// END SOLUTION
}

}  // namespace cuda
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cuda, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray& a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });

    // copy memory to host
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
  });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray* out) {
    cudaError_t err =
        cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}
