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

#include <gtest/gtest.h>

#include "Ippl.h"

#include "NBody/SphexaParticleContainer.hpp"
#include "NBody/SphexaBHSolver.hpp"
#include "NBodyTestUtil.hpp"

#include "cstone/sfc/box.hpp"
// cstone/cuda/cuda_utils.cuh must precede direct.cuh — direct.cuh references
// kernelSuccess() but does not include its declaration (upstream omission).
// RyoanjiDirect.cu uses the same workaround. It also provides the portable
// memcpy/syncGpu wrappers (CUDA or HIP).
#include "cstone/cuda/cuda_utils.cuh"
#include "ryoanji/nbody/direct.cuh"
#include "ryoanji/nbody/types.h"

using ippl::nbody::DoublePrecision;
using ippl::nbody::FieldVector;
using ippl::nbody::SphexaBHSolver;
using ippl::nbody::SphexaParticleContainer;
using ippl::nbody::test::downloadDevice;
using ippl::nbody::test::uploadHost;

namespace {

using C = SphexaParticleContainer<DoublePrecision, 3>;
constexpr auto kIdxCharge = C::idxOf<"charge">;

constexpr unsigned kN             = 4096;
constexpr unsigned kBucketSize    = 64;
constexpr unsigned kBucketSizeFoc = 64;
constexpr float    kTheta         = 0.5f;

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

    FieldVector<typename C::IdType> deviceID;
    FieldVector<T>                  dPx, dPy, dPz;
    deviceID.resize(kN);
    dPx.resize(kN);
    dPy.resize(kN);
    dPz.resize(kN);

    std::vector<T> xPre(kN), yPre(kN), zPre(kN), hPre(kN, 1.0e-2), qPre(kN, 1.0);
    std::vector<T> pxPre(kN, 0.0), pyPre(kN, 0.0), pzPre(kN, 0.0);
    std::vector<typename C::IdType> idPre(kN);
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
    uploadHost(idPre, deviceID.data());
    uploadHost(pxPre, dPx.data());
    uploadHost(pyPre, dPy.data());
    uploadHost(pzPre, dPz.data());

    typename SphexaBHSolver<P, 3>::Params params;
    params.G         = T(1);
    params.numShells = 0;

    SphexaBHSolver<P, 3> solver(pc, params);
    pc.setUniformH(0.01);
    pc.updateGrav<kIdxCharge>(deviceID, dPx, dPy, dPz);
    solver.runSolver();

    const unsigned start      = pc.startIndex();
    const unsigned end        = pc.endIndex();
    const unsigned nWithHalos = pc.nWithHalos();
    ASSERT_EQ(end - start, kN) << "Single-rank: every particle should be locally owned.";

    // Reference: direct N² on the post-sync (SFC-sorted) positions/charges/h.
    // directKernel does `+=` into these buffers (direct.cuh:78-83), so they must
    // be zero-initialized — the DeviceVector(size, init) ctor does that.
    cstone::DeviceVector<T> refPx(nWithHalos, T(0));
    cstone::DeviceVector<T> refAx(nWithHalos, T(0));
    cstone::DeviceVector<T> refAy(nWithHalos, T(0));
    cstone::DeviceVector<T> refAz(nWithHalos, T(0));

    // Open BCs: numShells=0; box vector unused for the gravity sum but required
    // by the API. Pass the same box dimensions the container was constructed with.
    const auto box = pc.box();
    ryoanji::Vec3<T> boxL{box.lx(), box.ly(), box.lz()};

    ryoanji::directSum(
        /*first=*/start, /*last=*/end, /*numBodies=*/end,
        boxL, /*numShells=*/0,
        getRaw<"Rx">(pc), getRaw<"Ry">(pc), getRaw<"Rz">(pc),
        getRaw<"charge">(pc), getRaw<"h">(pc),
        refPx.data(), refAx.data(), refAy.data(), refAz.data());

    syncGpu();

    std::vector<T> bhAx, bhAy, bhAz, dirAx, dirAy, dirAz;
    downloadDevice(getRaw<"Ex">(pc), nWithHalos, bhAx);
    downloadDevice(getRaw<"Ey">(pc), nWithHalos, bhAy);
    downloadDevice(getRaw<"Ez">(pc), nWithHalos, bhAz);
    downloadDevice(refAx.data(), nWithHalos, dirAx);
    downloadDevice(refAy.data(), nWithHalos, dirAy);
    downloadDevice(refAz.data(), nWithHalos, dirAz);

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
