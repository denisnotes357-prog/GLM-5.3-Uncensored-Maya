// tools/glm53/allreduce_probe.cu - how fast can N GPUs without peer access all-reduce a row of n floats through pinned
// host memory?  (docs: P8, tensor parallelism of glm-dsa's dense weights: two all-reduces a layer, 156 a token.)
//
// Each GPU, per round: a kernel writes its partial row into a host-mapped slot (zero-copy over PCIe), fences, raises
// its flag to the round number; then spins until every GPU's flag reached the round and sums the N rows it reads from
// host memory into its own device row.  Rounds alternate between two slot sets so a fast GPU never overwrites a row a
// slow one still reads.  Reports the mean time a round on every GPU and checks the sum.
//
//   nvcc -O3 -arch=sm_80 -gencode arch=compute_89,code=sm_89 -o allreduce_probe allreduce_probe.cu
//   CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5,0,2,3,4,7,6 ./allreduce_probe [n=6144] [rounds=2000] [blocks=0]
// blocks > 0: the sliced variant - each of `blocks` blocks owns a slice of the row end to end (writes it, raises its own
// flag, waits for that slice's flags from every GPU, sums it): no grid-wide step, so the rounds run back to back in the
// stream with one host sync at the end (what a layer loop would do).
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("%s: %s\n", #x, cudaGetErrorString(e_)); std::exit(1); } } while (0)

__global__ void round_kernel(const float* __restrict__ mine, float* __restrict__ out, volatile float* slots,
                             volatile unsigned* flags, int rank, int n_gpu, int n, unsigned round) {
    const int set = round & 1;
    volatile float* my = slots + ((size_t) set * n_gpu + rank) * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) my[i] = mine[i] + (float) round;
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) flags[set * n_gpu + rank] = round;
    __threadfence_system();
    if (threadIdx.x < n_gpu)
        while (flags[set * n_gpu + threadIdx.x] < round) {}
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float acc = 0.0f;
        for (int g = 0; g < n_gpu; ++g) acc += slots[((size_t) set * n_gpu + g) * n + i];
        out[i] = acc;
    }
}

__global__ void round_sliced_kernel(const float* __restrict__ mine, float* __restrict__ out, volatile float* slots,
                                    volatile unsigned* flags, int rank, int n_gpu, int n, unsigned round) {
    const int set = round & 1, nb = gridDim.x, b = blockIdx.x;
    const int per = (n + nb - 1) / nb, lo = b * per, hi = min(n, lo + per);
    volatile float* my = slots + ((size_t) set * n_gpu + rank) * n;
    for (int i = lo + threadIdx.x; i < hi; i += blockDim.x) my[i] = mine[i] + (float) round;
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) flags[((size_t) set * n_gpu + rank) * nb + b] = round;
    __threadfence_system();
    if (threadIdx.x < n_gpu)
        while (flags[((size_t) set * n_gpu + threadIdx.x) * nb + b] < round) {}
    __syncthreads();
    for (int i = lo + threadIdx.x; i < hi; i += blockDim.x) {
        float acc = 0.0f;
        for (int g = 0; g < n_gpu; ++g) acc += slots[((size_t) set * n_gpu + g) * n + i];
        out[i] = acc;
    }
}

int main(int argc, char** argv) {
    const int n = argc > 1 ? std::atoi(argv[1]) : 6144;
    const int rounds = argc > 2 ? std::atoi(argv[2]) : 2000;
    const int blocks = argc > 3 ? std::atoi(argv[3]) : 0;
    int n_gpu = 0;
    CK(cudaGetDeviceCount(&n_gpu));
    float* slots = nullptr;
    unsigned* flags = nullptr;
    CK(cudaHostAlloc(&slots, sizeof(float) * 2 * n_gpu * n, cudaHostAllocMapped | cudaHostAllocPortable));
    const int nflags = 2 * n_gpu * (blocks > 0 ? blocks : 1);
    CK(cudaHostAlloc(&flags, sizeof(unsigned) * nflags, cudaHostAllocMapped | cudaHostAllocPortable));
    for (int i = 0; i < nflags; ++i) flags[i] = 0;
    std::vector<float*> mine(n_gpu), out(n_gpu);
    std::vector<double> ms(n_gpu);
    std::vector<int> bad(n_gpu, 0);
    std::vector<std::thread> th;
    for (int g = 0; g < n_gpu; ++g) {
        CK(cudaSetDevice(g));
        CK(cudaMalloc(&mine[g], sizeof(float) * n));
        CK(cudaMalloc(&out[g], sizeof(float) * n));
        std::vector<float> h(n, (float) (g + 1));
        CK(cudaMemcpy(mine[g], h.data(), sizeof(float) * n, cudaMemcpyHostToDevice));
    }
    for (int g = 0; g < n_gpu; ++g)
        th.emplace_back([&, g] {
            cudaSetDevice(g);
            cudaStream_t s;
            cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
            float *ds = nullptr;
            unsigned* df = nullptr;
            cudaHostGetDevicePointer((void**) &ds, slots, 0);
            cudaHostGetDevicePointer((void**) &df, flags, 0);
            const auto t0 = std::chrono::steady_clock::now();
            for (int r = 1; r <= rounds; ++r) {
                if (blocks > 0) {
                    round_sliced_kernel<<<blocks, 256, 0, s>>>(mine[g], out[g], ds, df, g, n_gpu, n, (unsigned) r);
                } else {
                    round_kernel<<<1, 1024, 0, s>>>(mine[g], out[g], ds, df, g, n_gpu, n, (unsigned) r);
                    cudaStreamSynchronize(s);   // a layer's next step needs the sum: one round trip a round
                }
            }
            cudaStreamSynchronize(s);
            ms[g] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / rounds;
            std::vector<float> h(n);
            cudaMemcpy(h.data(), out[g], sizeof(float) * n, cudaMemcpyDeviceToHost);
            const float want = (float) (n_gpu * (n_gpu + 1) / 2) + (float) n_gpu * rounds;
            for (int i = 0; i < n; ++i)
                if (h[i] != want) { bad[g] = 1; break; }
        });
    for (auto& t : th) t.join();
    for (int g = 0; g < n_gpu; ++g)
        std::printf("GPU %d: %.1f us a round (%s)%s\n", g, ms[g] * 1000.0,
                    blocks > 0 ? "sliced, in the stream" : "launch + all-reduce + host sync", bad[g] ? "  WRONG SUM" : "");
    return 0;
}
