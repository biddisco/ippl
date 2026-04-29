#include "NBody/BeamDiagnostics.hpp"

#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>

namespace ippl::nbody {

namespace {

template <class T>
struct VelMinusMeanSqFunctor {
    const T* __restrict__ Px;
    const T* __restrict__ Py;
    const T* __restrict__ Pz;
    T meanVx, meanVy, meanVz;
    __device__ thrust::tuple<T, T, T> operator()(unsigned i) const {
        T dx = Px[i] - meanVx;
        T dy = Py[i] - meanVy;
        T dz = Pz[i] - meanVz;
        return thrust::make_tuple(dx * dx, dy * dy, dz * dz);
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
Triple<T> reduceMeanVelocity(SphexaParticleContainer<T, 3>& pc) {
    const unsigned start = pc.startIndex();
    const unsigned end   = pc.endIndex();
    Triple<T> out{T(0), T(0), T(0)};
    if (end <= start) { return out; }
    const unsigned n = end - start;
    thrust::device_ptr<const T> dPx(pc.getPxRaw());
    thrust::device_ptr<const T> dPy(pc.getPyRaw());
    thrust::device_ptr<const T> dPz(pc.getPzRaw());
    out.x = thrust::reduce(dPx + start, dPx + end, T(0), thrust::plus<T>()) / T(n);
    out.y = thrust::reduce(dPy + start, dPy + end, T(0), thrust::plus<T>()) / T(n);
    out.z = thrust::reduce(dPz + start, dPz + end, T(0), thrust::plus<T>()) / T(n);
    return out;
}

template <class T>
Triple<T> reduceTemperature(SphexaParticleContainer<T, 3>& pc, Triple<T> avgV) {
    const unsigned start = pc.startIndex();
    const unsigned end   = pc.endIndex();
    Triple<T> out{T(0), T(0), T(0)};
    if (end <= start) { return out; }
    const unsigned n = end - start;
    VelMinusMeanSqFunctor<T> functor{
        pc.getPxRaw(), pc.getPyRaw(), pc.getPzRaw(),
        avgV.x, avgV.y, avgV.z};
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

template Triple<float>  reduceMeanVelocity<float>(SphexaParticleContainer<float, 3>&);
template Triple<double> reduceMeanVelocity<double>(SphexaParticleContainer<double, 3>&);
template Triple<float>  reduceTemperature<float>(SphexaParticleContainer<float, 3>&, Triple<float>);
template Triple<double> reduceTemperature<double>(SphexaParticleContainer<double, 3>&, Triple<double>);

}  // namespace ippl::nbody
