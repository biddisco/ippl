#include "LandauDiagnostics.hpp"

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/tuple.h>

namespace ippl::nbody {

namespace {

template <class T>
struct SquareOp {
    __device__ T operator()(T x) const { return x * x; }
};

template <class T>
struct AbsOp {
    __device__ T operator()(T x) const { return x < T(0) ? -x : x; }
};

template <class T>
struct MaxOp {
    __device__ T operator()(T a, T b) const { return a < b ? b : a; }
};

}  // namespace

template <class T>
AxisStats<T> reduceExStats(SphexaParticleContainer<T, 3>& pc) {
    const unsigned start = pc.startIndex();
    const unsigned end   = pc.endIndex();

    AxisStats<T> out{T(0), T(0)};
    if (end <= start) { return out; }

    thrust::device_ptr<const T> Ex(pc.getExRaw());
    auto first = Ex + start;
    auto last  = Ex + end;

    out.sumSq = thrust::transform_reduce(
        first, last, SquareOp<T>(), T(0), thrust::plus<T>());

    out.maxAbs = thrust::transform_reduce(
        first, last, AbsOp<T>(), T(0), MaxOp<T>());

    return out;
}

template AxisStats<float>  reduceExStats<float>(SphexaParticleContainer<float, 3>&);
template AxisStats<double> reduceExStats<double>(SphexaParticleContainer<double, 3>&);

namespace {

template <class T>
struct CosSinPlus {
    __host__ __device__
    thrust::tuple<T, T> operator()(const thrust::tuple<T, T>& a,
                                    const thrust::tuple<T, T>& b) const {
        return thrust::make_tuple(thrust::get<0>(a) + thrust::get<0>(b),
                                  thrust::get<1>(a) + thrust::get<1>(b));
    }
};

template <class T>
struct CosSinFunctor {
    const T* __restrict__ Ex;
    const T* __restrict__ Rx;
    T kx;
    __device__ thrust::tuple<T, T> operator()(unsigned i) const {
        T phase = kx * Rx[i];
        return thrust::make_tuple(Ex[i] * cos(phase), Ex[i] * sin(phase));
    }
};

}  // namespace

template <class T>
CosSinAmp<T> reduceExCosineMode(SphexaParticleContainer<T, 3>& pc, T kx) {
    const unsigned start = pc.startIndex();
    const unsigned end   = pc.endIndex();

    CosSinAmp<T> out{T(0), T(0)};
    if (end <= start) { return out; }

    CosSinFunctor<T> functor{pc.getExRaw(), pc.getRxRaw(), kx};
    auto first = thrust::counting_iterator<unsigned>(start);
    auto last  = thrust::counting_iterator<unsigned>(end);

    auto sum = thrust::transform_reduce(
        first, last, functor,
        thrust::make_tuple(T(0), T(0)),
        CosSinPlus<T>());

    out.cosAmp = thrust::get<0>(sum);
    out.sinAmp = thrust::get<1>(sum);
    return out;
}

template CosSinAmp<float>  reduceExCosineMode<float>(SphexaParticleContainer<float, 3>&, float);
template CosSinAmp<double> reduceExCosineMode<double>(SphexaParticleContainer<double, 3>&, double);

namespace {

// Scatter Eₓ_p onto a periodic G³ grid with CIC weights. Each particle
// contributes to its surrounding 8 cells; gExSum accumulates Σ w·Eₓ and gW
// accumulates Σ w. After scattering, Eₓ_avg(cell) = gExSum/gW where gW > 0.
template <class T>
__global__ void cicScatterExKernel(unsigned numLocal, unsigned start, int G, T invH,
                                    const T* __restrict__ Rx,
                                    const T* __restrict__ Ry,
                                    const T* __restrict__ Rz,
                                    const T* __restrict__ Ex,
                                    T* gExSum, T* gW) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numLocal) { return; }
    const unsigned p = start + tid;

    T xn = Rx[p] * invH;
    T yn = Ry[p] * invH;
    T zn = Rz[p] * invH;
    int ix = static_cast<int>(floor(xn));
    int iy = static_cast<int>(floor(yn));
    int iz = static_cast<int>(floor(zn));
    T fx = xn - ix;
    T fy = yn - iy;
    T fz = zn - iz;
    T e  = Ex[p];

    #pragma unroll
    for (int oz = 0; oz < 2; ++oz) {
        T   wz = oz ? fz : (T(1) - fz);
        int gz = ((iz + oz) % G + G) % G;
        #pragma unroll
        for (int oy = 0; oy < 2; ++oy) {
            T   wy = oy ? fy : (T(1) - fy);
            int gy = ((iy + oy) % G + G) % G;
            #pragma unroll
            for (int ox = 0; ox < 2; ++ox) {
                T   wx = ox ? fx : (T(1) - fx);
                int gx  = ((ix + ox) % G + G) % G;
                int idx = gx + G * (gy + G * gz);
                T   w   = wx * wy * wz;
                atomicAdd(&gExSum[idx], w * e);
                atomicAdd(&gW[idx],     w);
            }
        }
    }
}

template <class T>
struct CellEnergyOp {
    const T* gExSum;
    const T* gW;
    __device__ T operator()(int i) const {
        T w = gW[i];
        if (w <= T(0)) { return T(0); }
        T e = gExSum[i] / w;
        return e * e;
    }
};

}  // namespace

template <class T>
T reduceExGridEnergyCIC(SphexaParticleContainer<T, 3>& pc, T L, int G) {
    const unsigned start    = pc.startIndex();
    const unsigned end      = pc.endIndex();
    const unsigned numLocal = (end > start) ? (end - start) : 0u;
    if (numLocal == 0 || G <= 0) { return T(0); }

    const int total = G * G * G;
    thrust::device_vector<T> gExSum(total, T(0));
    thrust::device_vector<T> gW(total,     T(0));

    const int block = 256;
    const int grid  = static_cast<int>((numLocal + block - 1) / block);
    const T   invH  = static_cast<T>(G) / L;

    cicScatterExKernel<T><<<grid, block>>>(
        numLocal, start, G, invH,
        pc.getRxRaw(), pc.getRyRaw(), pc.getRzRaw(), pc.getExRaw(),
        thrust::raw_pointer_cast(gExSum.data()),
        thrust::raw_pointer_cast(gW.data()));

    CellEnergyOp<T> op{thrust::raw_pointer_cast(gExSum.data()),
                       thrust::raw_pointer_cast(gW.data())};
    auto first = thrust::counting_iterator<int>(0);
    T sumSq = thrust::transform_reduce(first, first + total, op,
                                       T(0), thrust::plus<T>());

    const T h = L / static_cast<T>(G);
    return sumSq * h * h * h;
}

template float  reduceExGridEnergyCIC<float >(SphexaParticleContainer<float,  3>&, float,  int);
template double reduceExGridEnergyCIC<double>(SphexaParticleContainer<double, 3>&, double, int);

}  // namespace ippl::nbody
