// Sanity test for ippl::nbody::SphexaParticleContainer<DoublePrecision, 3>.
//
// Verifies that update() (which wraps cstone::Domain::sync) correctly:
//   1. preserves the particle count on a single rank,
//   2. produces a permutation of the input on the locally-owned range,
//   3. permutes the user ID attribute in lockstep with positions —
//      i.e. for every j in [startIndex(), endIndex()), R_post[j] == R_pre[ID_post[j]].
// Repeated update() calls without movement must keep the same invariants.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <unordered_set>
#include <vector>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "Ippl.h"   // ippl::initialize / ippl::finalize bring up MPI + Kokkos
#include "NBody/SphexaParticleContainer.hpp"
#include "cstone/sfc/box.hpp"

using ippl::nbody::DoublePrecision;
using ippl::nbody::SphexaParticleContainer;

namespace {

constexpr unsigned kN              = 8192;
constexpr unsigned kBucketSize     = 64;
constexpr unsigned kBucketSizeFoc  = 64;
constexpr float    kTheta          = 0.5f;

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

using IdType = SphexaParticleContainer<DoublePrecision, 3>::IdType;

void verifyLockstep(SphexaParticleContainer<DoublePrecision, 3>& pc,
                    const std::vector<double>& xPre,
                    const std::vector<double>& yPre,
                    const std::vector<double>& zPre) {
    const unsigned start = pc.startIndex();
    const unsigned end   = pc.endIndex();
    const unsigned nWithHalos = pc.nWithHalos();

    ASSERT_LE(start, end);
    ASSERT_LE(end, nWithHalos);
    EXPECT_EQ(end - start, kN) << "Single-rank: every input particle should be locally owned.";

    std::vector<double> xPost, yPost, zPost;
    std::vector<IdType> idPost;
    downloadDevice(getRaw<"Rx">(pc), nWithHalos, xPost);
    downloadDevice(getRaw<"Ry">(pc), nWithHalos, yPost);
    downloadDevice(getRaw<"Rz">(pc), nWithHalos, zPost);
    downloadDevice(getRaw<"ID">(pc), nWithHalos, idPost);

    std::unordered_set<IdType> seen;
    seen.reserve(end - start);

    for (unsigned j = start; j < end; ++j) {
        const IdType id = idPost[j];
        ASSERT_LT(id, kN) << "ID out of range at slot " << j;
        ASSERT_TRUE(seen.insert(id).second) << "Duplicate ID " << id << " at slot " << j;

        EXPECT_DOUBLE_EQ(xPost[j], xPre[id])
            << "Rx mismatch at j=" << j << " ID=" << id;
        EXPECT_DOUBLE_EQ(yPost[j], yPre[id])
            << "Ry mismatch at j=" << j << " ID=" << id;
        EXPECT_DOUBLE_EQ(zPost[j], zPre[id])
            << "Rz mismatch at j=" << j << " ID=" << id;
    }
    EXPECT_EQ(seen.size(), end - start) << "ID set must be a permutation of [0, N).";
}

} // namespace

TEST(SphexaParticleContainer, SyncPermutesAttributesInLockstep) {
    using cstone::BoundaryType;

    SphexaParticleContainer<DoublePrecision, 3> pc(
        /*rank=*/0, /*nRanks=*/1,
        kBucketSize, kBucketSizeFoc, kTheta,
        /*boxLoHi=*/std::array<double, 6>{0.0, 1.0, 0.0, 1.0, 0.0, 1.0},
        /*boundaries=*/std::array<BoundaryType, 3>{
            BoundaryType::open, BoundaryType::open, BoundaryType::open});

    pc.create(kN);

    std::vector<double> xPre(kN), yPre(kN), zPre(kN), hPre(kN, 1.0e-2);
    std::vector<IdType> idPre(kN);
    ::srand48(/*seed=*/424242);
    for (unsigned i = 0; i < kN; ++i) {
        xPre[i]  = drand48();
        yPre[i]  = drand48();
        zPre[i]  = drand48();
        idPre[i] = static_cast<IdType>(i);
    }

    uploadHost(xPre,  getRaw<"Rx">(pc));
    uploadHost(yPre,  getRaw<"Ry">(pc));
    uploadHost(zPre,  getRaw<"Rz">(pc));
    uploadHost(hPre,  getRaw<"h">(pc));
    uploadHost(idPre, getRaw<"ID">(pc));

    pc.update();
    verifyLockstep(pc, xPre, yPre, zPre);

    // A second sync without any change in positions must preserve the invariant.
    pc.update();
    verifyLockstep(pc, xPre, yPre, zPre);
}

int main(int argc, char* argv[]) {
    // cstone's Domain calls MPI collectives unconditionally (even on nRanks=1),
    // so we cannot use the stock GoogleTest main. ippl::initialize wraps
    // MPI_Init + Kokkos::initialize.
    ippl::initialize(argc, argv);
    int success = 1;
    {
        ::testing::InitGoogleTest(&argc, argv);
        success = RUN_ALL_TESTS();
    }
    ippl::finalize();
    return success;
}
