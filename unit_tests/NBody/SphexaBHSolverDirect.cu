// Sanity test: SphexaBHSolver should match direct N² to within the
// theta-determined error bound on a small, well-conditioned system.
//
// We run the BH pipeline first (which SFC-sorts particles via updateGrav), then
// run ryoanji::directSum on the post-sync positions/charges to obtain reference
// accelerations indexed in the same SFC slot. Comparison: per-particle L2 norm
// of the acceleration error relative to the L2 norm of the reference.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "Ippl.h"

#include "NBody/SphexaParticleContainer.hpp"
#include "NBody/SphexaBHSolver.hpp"

#include "cstone/sfc/box.hpp"
// cstone/cuda/cuda_utils.cuh must precede direct.cuh — direct.cuh references
// kernelSuccess() but does not include its declaration (upstream omission).
// RyoanjiDirect.cu uses the same workaround.
#include "cstone/cuda/cuda_utils.cuh"
#include "ryoanji/nbody/direct.cuh"
#include "ryoanji/nbody/types.h"

using ippl::nbody::DoublePrecision;
using ippl::nbody::SphexaBHSolver;
using ippl::nbody::SphexaParticleContainer;

namespace {

using C = SphexaParticleContainer<DoublePrecision, 3>;
constexpr auto kIdxCharge = C::idxOf<"charge">;
constexpr auto kIdxID     = C::idxOf<"ID">;
constexpr auto kIdxPx     = C::idxOf<"Px">;
constexpr auto kIdxPy     = C::idxOf<"Py">;
constexpr auto kIdxPz     = C::idxOf<"Pz">;

constexpr unsigned kN             = 4096;
constexpr unsigned kBucketSize    = 64;
constexpr unsigned kBucketSizeFoc = 64;
constexpr float    kTheta         = 0.5f;

template <class T>
void downloadDevice(const T* dPtr, std::size_t n, std::vector<T>& host) {
    host.resize(n);
    if (n == 0) { return; }
    cudaError_t err = cudaMemcpy(host.data(), dPtr, n * sizeof(T), cudaMemcpyDeviceToHost);
    ASSERT_EQ(err, cudaSuccess) << "cudaMemcpy D2H failed: " << cudaGetErrorString(err);
}

template <class T>
void uploadHost(const std::vector<T>& host, T* dPtr) {
    if (host.empty()) { return; }
    cudaError_t err = cudaMemcpy(dPtr, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice);
    ASSERT_EQ(err, cudaSuccess) << "cudaMemcpy H2D failed: " << cudaGetErrorString(err);
}

// Allocate a device buffer of n elements of type T. Returns the device pointer.
// Free with cudaFree.
template <class T>
T* deviceAlloc(std::size_t n) {
    T* p = nullptr;
    cudaError_t err = cudaMalloc(&p, n * sizeof(T));
    EXPECT_EQ(err, cudaSuccess) << "cudaMalloc failed: " << cudaGetErrorString(err);
    return p;
}

} // namespace

TEST(SphexaBHSolver, MatchesDirectSumOpenBC) {
    using T = double;
    using P = DoublePrecision;
    using cstone::BoundaryType;

    SphexaParticleContainer<P, 3> pc(
        /*rank=*/0, /*nRanks=*/1,
        kBucketSize, kBucketSizeFoc, kTheta,
        std::array<T, 6>{0.0, 1.0, 0.0, 1.0, 0.0, 1.0},
        std::array<BoundaryType, 3>{
            BoundaryType::open, BoundaryType::open, BoundaryType::open});

    pc.create(kN);

    std::vector<T> xPre(kN), yPre(kN), zPre(kN), hPre(kN, 1.0e-2), qPre(kN, 1.0);
    std::vector<typename SphexaParticleContainer<P, 3>::IdType> idPre(kN);
    ::srand48(/*seed=*/424242);
    for (unsigned i = 0; i < kN; ++i) {
        xPre[i]  = drand48();
        yPre[i]  = drand48();
        zPre[i]  = drand48();
        idPre[i] = i;
    }

    uploadHost(xPre,  getRaw<"Rx">(pc));
    uploadHost(yPre,  getRaw<"Ry">(pc));
    uploadHost(zPre,  getRaw<"Rz">(pc));
    uploadHost(hPre,  getRaw<"h">(pc));
    uploadHost(qPre,  getRaw<"charge">(pc));
    uploadHost(idPre, getRaw<"ID">(pc));

    typename SphexaBHSolver<P, 3>::Params params;
    params.G         = T(1);
    params.numShells = 0;

    SphexaBHSolver<P, 3> solver(pc, params);
    pc.setUniformH(0.01);
    pc.updateGrav<kIdxCharge, kIdxID, kIdxPx, kIdxPy, kIdxPz>();
    solver.runSolver();

    const unsigned start      = pc.startIndex();
    const unsigned end        = pc.endIndex();
    const unsigned nWithHalos = pc.nWithHalos();
    ASSERT_EQ(end - start, kN) << "Single-rank: every particle should be locally owned.";

    // Reference: direct N² on the post-sync (SFC-sorted) positions/charges/h.
    // Allocate output buffers on device for the directSum reference. directKernel
    // does `+=` into these buffers (direct.cuh:78-83), so they must be zeroed —
    // cudaMalloc does not initialize memory.
    T* refPx = deviceAlloc<T>(nWithHalos);
    T* refAx = deviceAlloc<T>(nWithHalos);
    T* refAy = deviceAlloc<T>(nWithHalos);
    T* refAz = deviceAlloc<T>(nWithHalos);
    {
        const std::size_t bytes = static_cast<std::size_t>(nWithHalos) * sizeof(T);
        cudaMemset(refPx, 0, bytes);
        cudaMemset(refAx, 0, bytes);
        cudaMemset(refAy, 0, bytes);
        cudaMemset(refAz, 0, bytes);
    }

    // Open BCs: numShells=0; box vector unused for the gravity sum but required
    // by the API. Pass the same box dimensions the container was constructed with.
    const auto box = pc.box();
    ryoanji::Vec3<T> boxL{box.lx(), box.ly(), box.lz()};

    ryoanji::directSum(
        /*first=*/start, /*last=*/end, /*numBodies=*/end,
        boxL, /*numShells=*/0,
        getRaw<"Rx">(pc), getRaw<"Ry">(pc), getRaw<"Rz">(pc),
        getRaw<"charge">(pc), getRaw<"h">(pc),
        refPx, refAx, refAy, refAz);

    cudaDeviceSynchronize();

    std::vector<T> bhAx, bhAy, bhAz, dirAx, dirAy, dirAz;
    downloadDevice(getRaw<"Ex">(pc), nWithHalos, bhAx);
    downloadDevice(getRaw<"Ey">(pc), nWithHalos, bhAy);
    downloadDevice(getRaw<"Ez">(pc), nWithHalos, bhAz);
    downloadDevice(refAx, nWithHalos, dirAx);
    downloadDevice(refAy, nWithHalos, dirAy);
    downloadDevice(refAz, nWithHalos, dirAz);

    cudaFree(refPx);
    cudaFree(refAx);
    cudaFree(refAy);
    cudaFree(refAz);

    // Mean-relative L2 error over the locally-owned range.
    long double sqErr = 0.0L;
    long double sqRef = 0.0L;
    for (unsigned j = start; j < end; ++j) {
        const long double ex = static_cast<long double>(bhAx[j]) - dirAx[j];
        const long double ey = static_cast<long double>(bhAy[j]) - dirAy[j];
        const long double ez = static_cast<long double>(bhAz[j]) - dirAz[j];
        sqErr += ex * ex + ey * ey + ez * ez;

        const long double rx = dirAx[j];
        const long double ry = dirAy[j];
        const long double rz = dirAz[j];
        sqRef += rx * rx + ry * ry + rz * rz;
    }
    ASSERT_GT(sqRef, 0.0L) << "Direct sum produced zero-norm reference — invalid input.";
    const long double relL2 = std::sqrt(sqErr / sqRef);

    // theta=0.5 with Cartesian quadrupole multipoles: typical mean-relative L2 is
    // around 1e-3 to 1e-2 on uniform random distributions. Lock at 1e-2.
    EXPECT_LT(relL2, 1e-2L)
        << "BH-vs-direct relative L2 error " << static_cast<double>(relL2)
        << " exceeds 1e-2 for theta=" << kTheta;
}

int main(int argc, char* argv[]) {
    // ippl::initialize wraps MPI_Init + Kokkos::initialize; cstone's Domain calls
    // MPI collectives unconditionally even on nRanks=1.
    ippl::initialize(argc, argv);
    int success = 1;
    {
        ::testing::InitGoogleTest(&argc, argv);
        success = RUN_ALL_TESTS();
    }
    ippl::finalize();
    return success;
}
