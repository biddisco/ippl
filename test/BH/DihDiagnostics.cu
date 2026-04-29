#include "DihDiagnostics.hpp"

#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>

namespace ippl::nbody {

namespace {

template <class T>
struct BeamStatsRaw {
    T vals[12];
    __host__ __device__ BeamStatsRaw() {
        for (int i = 0; i < 12; ++i) { vals[i] = T(0); }
    }
};

template <class T>
struct BeamStatsPlus {
    __host__ __device__ BeamStatsRaw<T> operator()(const BeamStatsRaw<T>& a,
                                                   const BeamStatsRaw<T>& b) const {
        BeamStatsRaw<T> r;
        for (int i = 0; i < 12; ++i) { r.vals[i] = a.vals[i] + b.vals[i]; }
        return r;
    }
};

template <class T>
struct BeamStatsFunctor {
    const T* __restrict__ Rx;
    const T* __restrict__ Ry;
    const T* __restrict__ Rz;
    const T* __restrict__ Px;
    const T* __restrict__ Py;
    const T* __restrict__ Pz;

    __device__ BeamStatsRaw<T> operator()(unsigned i) const {
        BeamStatsRaw<T> s;
        s.vals[ 0] = Rx[i] * Rx[i];   s.vals[ 1] = Px[i] * Px[i];   s.vals[ 2] = Rx[i] * Px[i];
        s.vals[ 3] = Ry[i] * Ry[i];   s.vals[ 4] = Py[i] * Py[i];   s.vals[ 5] = Ry[i] * Py[i];
        s.vals[ 6] = Rz[i] * Rz[i];   s.vals[ 7] = Pz[i] * Pz[i];   s.vals[ 8] = Rz[i] * Pz[i];
        s.vals[ 9] = Rx[i];           s.vals[10] = Ry[i];           s.vals[11] = Rz[i];
        return s;
    }
};

template <class T>
struct AbsTripleFunctor {
    const T* __restrict__ Ex;
    const T* __restrict__ Ey;
    const T* __restrict__ Ez;
    __device__ thrust::tuple<T, T, T> operator()(unsigned i) const {
        return thrust::make_tuple(Ex[i] < T(0) ? -Ex[i] : Ex[i],
                                  Ey[i] < T(0) ? -Ey[i] : Ey[i],
                                  Ez[i] < T(0) ? -Ez[i] : Ez[i]);
    }
};

template <class T>
struct TuplePlus3 {
    __host__ __device__ thrust::tuple<T, T, T>
    operator()(const thrust::tuple<T, T, T>& a, const thrust::tuple<T, T, T>& b) const {
        return thrust::make_tuple(thrust::get<0>(a) + thrust::get<0>(b),
                                  thrust::get<1>(a) + thrust::get<1>(b),
                                  thrust::get<2>(a) + thrust::get<2>(b));
    }
};

}  // namespace

template <class T>
BeamStats12<T> reduceBeamStats(SphexaParticleContainer<T, 3>& pc) {
    const unsigned start = pc.startIndex();
    const unsigned end   = pc.endIndex();
    BeamStats12<T> out{};
    for (int i = 0; i < 12; ++i) { out.vals[i] = T(0); }
    if (end <= start) { return out; }

    BeamStatsFunctor<T> functor{
        pc.getRxRaw(), pc.getRyRaw(), pc.getRzRaw(),
        pc.getPxRaw(), pc.getPyRaw(), pc.getPzRaw()};

    auto first = thrust::counting_iterator<unsigned>(start);
    auto last  = thrust::counting_iterator<unsigned>(end);

    BeamStatsRaw<T> sum = thrust::transform_reduce(
        first, last, functor, BeamStatsRaw<T>(), BeamStatsPlus<T>());

    for (int i = 0; i < 12; ++i) { out.vals[i] = sum.vals[i]; }
    return out;
}

template <class T>
Triple<T> reduceMeanAbsAccel(SphexaParticleContainer<T, 3>& pc) {
    const unsigned start = pc.startIndex();
    const unsigned end   = pc.endIndex();
    Triple<T> out{T(0), T(0), T(0)};
    if (end <= start) { return out; }
    const unsigned n = end - start;
    AbsTripleFunctor<T> functor{pc.getExRaw(), pc.getEyRaw(), pc.getEzRaw()};
    auto first = thrust::counting_iterator<unsigned>(start);
    auto last  = thrust::counting_iterator<unsigned>(end);
    auto sum = thrust::transform_reduce(
        first, last, functor,
        thrust::make_tuple(T(0), T(0), T(0)),
        TuplePlus3<T>());
    out.x = thrust::get<0>(sum) / T(n);
    out.y = thrust::get<1>(sum) / T(n);
    out.z = thrust::get<2>(sum) / T(n);
    return out;
}

template BeamStats12<float>  reduceBeamStats<float>(SphexaParticleContainer<float, 3>&);
template BeamStats12<double> reduceBeamStats<double>(SphexaParticleContainer<double, 3>&);
template Triple<float>  reduceMeanAbsAccel<float>(SphexaParticleContainer<float, 3>&);
template Triple<double> reduceMeanAbsAccel<double>(SphexaParticleContainer<double, 3>&);

}  // namespace ippl::nbody
