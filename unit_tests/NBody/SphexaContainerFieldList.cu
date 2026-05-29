// Tests for the templated update / updateGrav / exchangeHalos path on
// SphexaParticleContainer, exercised via the idxOf<"Name"> compile-time helper.

#include <cstdint>
#include <unordered_set>
#include <vector>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "Ippl.h"
#include "NBody/SphexaParticleContainer.hpp"
#include "cstone/sfc/box.hpp"

using ippl::nbody::DoublePrecision;
using ippl::nbody::SphexaParticleContainer;

namespace {

using C      = SphexaParticleContainer<DoublePrecision, 3>;
using IdType = C::IdType;

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

} // namespace

TEST(SphexaContainerFieldList, IdxOfResolvesNamesAtCompileTime) {
    static_assert(C::idxOf<"Rx">     == 0);
    static_assert(C::idxOf<"Ry">     == 1);
    static_assert(C::idxOf<"Rz">     == 2);
    static_assert(C::idxOf<"h">      == 3);
    static_assert(C::idxOf<"ID">     == 4);
    static_assert(C::idxOf<"charge"> == 5);
    static_assert(C::idxOf<"Px">     == 6);
    static_assert(C::idxOf<"Py">     == 7);
    static_assert(C::idxOf<"Pz">     == 8);
    static_assert(C::idxOf<"Ex">     == 9);
    static_assert(C::idxOf<"Ey">     == 10);
    static_assert(C::idxOf<"Ez">     == 11);

    // Unknown name → past-the-end (sentinel from cstone::getFieldIndex).
    static_assert(C::idxOf<"unknownField"> == C::fieldNames.size());

    EXPECT_EQ(C::idxOf<"Rx">,     0u);
    EXPECT_EQ(C::idxOf<"charge">, 5u);
}

TEST(SphexaContainerFieldList, TemplatedUpdateGravPermutesIDInLockstep) {
    using cstone::BoundaryType;

    C pc(/*rank=*/0, /*nRanks=*/1,
         kBucketSize, kBucketSizeFoc, kTheta,
         std::array<double, 6>{0.0, 1.0, 0.0, 1.0, 0.0, 1.0},
         std::array<BoundaryType, 3>{
             BoundaryType::open, BoundaryType::open, BoundaryType::open});

    pc.create(kN);
    pc.setUniformH(0.01);

    std::vector<double> xPre(kN), yPre(kN), zPre(kN), hPre(kN, 1.0e-2), qPre(kN, 1.0);
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
    uploadHost(qPre,  getRaw<"charge">(pc));
    uploadHost(idPre, getRaw<"ID">(pc));

    pc.updateGrav<C::idxOf<"charge">,
                  C::idxOf<"ID">,
                  C::idxOf<"Px">,
                  C::idxOf<"Py">,
                  C::idxOf<"Pz">>();

    const unsigned start      = pc.startIndex();
    const unsigned end        = pc.endIndex();
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
        EXPECT_DOUBLE_EQ(xPost[j], xPre[id]) << "Rx mismatch at j=" << j << " ID=" << id;
        EXPECT_DOUBLE_EQ(yPost[j], yPre[id]) << "Ry mismatch at j=" << j << " ID=" << id;
        EXPECT_DOUBLE_EQ(zPost[j], zPre[id]) << "Rz mismatch at j=" << j << " ID=" << id;
    }
    EXPECT_EQ(seen.size(), end - start) << "ID set must be a permutation of [0, N).";
}

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    int success = 1;
    {
        ::testing::InitGoogleTest(&argc, argv);
        success = RUN_ALL_TESTS();
    }
    ippl::finalize();
    return success;
}
