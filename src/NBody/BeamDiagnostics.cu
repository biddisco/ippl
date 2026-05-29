#include "NBody/BeamDiagnostics.hpp"

#include <mpi.h>

#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>

namespace ippl::nbody {

namespace {

template <class T> struct MpiTypeOf;
template <> struct MpiTypeOf<float>  { static MPI_Datatype value() { return MPI_FLOAT;  } };
template <> struct MpiTypeOf<double> { static MPI_Datatype value() { return MPI_DOUBLE; } };

template <class T>
Triple<T> finalizeMeanTriple(Triple<T> localSum, unsigned long localN) {
    T inout[4] = {localSum.x, localSum.y, localSum.z, static_cast<T>(localN)};
    MPI_Allreduce(MPI_IN_PLACE, inout, 4, MpiTypeOf<T>::value(), MPI_SUM, MPI_COMM_WORLD);
    const T invN = (inout[3] > T(0)) ? T(1) / inout[3] : T(0);
    return Triple<T>{inout[0] * invN, inout[1] * invN, inout[2] * invN};
}

}  // namespace

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

template <class P>
Triple<typename P::Tc> reduceMeanVelocity(SphexaParticleContainer<P, 3>& pc) {
    using Tc = typename P::Tc;
    const unsigned start = pc.startIndex();
    const unsigned end   = pc.endIndex();
    Triple<Tc> localSum{Tc(0), Tc(0), Tc(0)};
    if (end > start) {
        thrust::device_ptr<const Tc> dPx(getRaw<"Px">(pc));
        thrust::device_ptr<const Tc> dPy(getRaw<"Py">(pc));
        thrust::device_ptr<const Tc> dPz(getRaw<"Pz">(pc));
        localSum.x = thrust::reduce(dPx + start, dPx + end, Tc(0), thrust::plus<Tc>());
        localSum.y = thrust::reduce(dPy + start, dPy + end, Tc(0), thrust::plus<Tc>());
        localSum.z = thrust::reduce(dPz + start, dPz + end, Tc(0), thrust::plus<Tc>());
    }
    return finalizeMeanTriple<Tc>(localSum,
                                  static_cast<unsigned long>(end > start ? end - start : 0u));
}

template <class P>
Triple<typename P::Tc> reduceTemperature(SphexaParticleContainer<P, 3>& pc,
                                         Triple<typename P::Tc> avgV) {
    using Tc = typename P::Tc;
    const unsigned start = pc.startIndex();
    const unsigned end   = pc.endIndex();
    Triple<Tc> localSum{Tc(0), Tc(0), Tc(0)};
    if (end > start) {
        VelMinusMeanSqFunctor<Tc> functor{
            getRaw<"Px">(pc), getRaw<"Py">(pc), getRaw<"Pz">(pc),
            avgV.x, avgV.y, avgV.z};
        auto first = thrust::counting_iterator<unsigned>(start);
        auto last  = thrust::counting_iterator<unsigned>(end);
        auto sum = thrust::transform_reduce(
            first, last, functor,
            thrust::make_tuple(Tc(0), Tc(0), Tc(0)),
            TuplePlus3<Tc>());
        localSum.x = thrust::get<0>(sum);
        localSum.y = thrust::get<1>(sum);
        localSum.z = thrust::get<2>(sum);
    }
    return finalizeMeanTriple<Tc>(localSum,
                                  static_cast<unsigned long>(end > start ? end - start : 0u));
}

template Triple<double> reduceMeanVelocity<DoublePrecision>(SphexaParticleContainer<DoublePrecision, 3>&);
template Triple<double> reduceMeanVelocity<MixedPrecision> (SphexaParticleContainer<MixedPrecision,  3>&);
template Triple<float>  reduceMeanVelocity<FloatPrecision> (SphexaParticleContainer<FloatPrecision,  3>&);
template Triple<double> reduceTemperature<DoublePrecision>(SphexaParticleContainer<DoublePrecision, 3>&, Triple<double>);
template Triple<double> reduceTemperature<MixedPrecision> (SphexaParticleContainer<MixedPrecision,  3>&, Triple<double>);
template Triple<float>  reduceTemperature<FloatPrecision> (SphexaParticleContainer<FloatPrecision,  3>&, Triple<float>);

}  // namespace ippl::nbody
