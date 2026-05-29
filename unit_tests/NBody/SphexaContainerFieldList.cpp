// Tests for the templated update / updateGrav / exchangeHalos path on
// SphexaParticleContainer, exercised via the idxOf<"Name"> compile-time helper.

#include <cstdint>
#include <unordered_set>
#include <vector>

#include <gtest/gtest.h>

#include "Ippl.h"
#include "cstone/sfc/box.hpp"

#include "NBody/SphexaParticleContainer.hpp"
#include "NBodyTestUtil.hpp"

using ippl::nbody::DoublePrecision;
using ippl::nbody::FieldVector;
using ippl::nbody::SphexaParticleContainer;
using ippl::nbody::test::downloadDevice;
using ippl::nbody::test::uploadHost;

namespace {

using C      = SphexaParticleContainer<DoublePrecision, 3>;
using IdType = C::IdType;

constexpr unsigned kN              = 8192;
constexpr unsigned kBucketSize     = 64;
constexpr unsigned kBucketSizeFoc  = 64;
constexpr float    kTheta          = 0.5f;

constexpr auto kIdxCharge = C::idxOf<"charge">;

} // namespace

TEST(SphexaContainerFieldList, IdxOfResolvesNamesAtCompileTime) {
    static_assert(C::idxOf<"Rx">     == 0);
    static_assert(C::idxOf<"Ry">     == 1);
    static_assert(C::idxOf<"Rz">     == 2);
    static_assert(C::idxOf<"h">      == 3);
    static_assert(C::idxOf<"charge"> == 4);
    static_assert(C::idxOf<"Ex">     == 5);
    static_assert(C::idxOf<"Ey">     == 6);
    static_assert(C::idxOf<"Ez">     == 7);

    // Unknown name → past-the-end (sentinel from cstone::getFieldIndex).
    static_assert(C::idxOf<"unknownField"> == C::fieldNames.size());

    EXPECT_EQ(C::idxOf<"Rx">,     0u);
    EXPECT_EQ(C::idxOf<"charge">, 4u);
}

TEST(SphexaContainerFieldList, TemplatedUpdateGravPermutesDriverFieldsInLockstep) {
    using cstone::BoundaryType;

    C pc(/*rank=*/0, /*nRanks=*/1,
         kBucketSize, kBucketSizeFoc, kTheta,
         std::array<double, 6>{0.0, 1.0, 0.0, 1.0, 0.0, 1.0},
         std::array<BoundaryType, 3>{
             BoundaryType::open, BoundaryType::open, BoundaryType::open});

    pc.create(kN);
    pc.setUniformH(0.01);

    FieldVector<IdType> id;
    FieldVector<double> px, py, pz;
    id.resize(kN);
    px.resize(kN);
    py.resize(kN);
    pz.resize(kN);

    std::vector<double> xPre(kN), yPre(kN), zPre(kN), hPre(kN, 1.0e-2), qPre(kN);
    std::vector<double> pxPre(kN), pyPre(kN), pzPre(kN);
    std::vector<IdType> idPre(kN);
    ::srand48(/*seed=*/424242);
    for (unsigned i = 0; i < kN; ++i) {
        xPre[i]  = drand48();
        yPre[i]  = drand48();
        zPre[i]  = drand48();
        pxPre[i] = 2.0 * static_cast<double>(i);
        pyPre[i] = 3.0 * static_cast<double>(i);
        pzPre[i] = 5.0 * static_cast<double>(i);
        qPre[i]  = 0.1 + 0.001 * static_cast<double>(i);
        idPre[i] = static_cast<IdType>(i);
    }

    uploadHost(xPre,  ippl::nbody::getRaw<"Rx">(pc));
    uploadHost(yPre,  ippl::nbody::getRaw<"Ry">(pc));
    uploadHost(zPre,  ippl::nbody::getRaw<"Rz">(pc));
    uploadHost(hPre,  ippl::nbody::getRaw<"h">(pc));
    uploadHost(qPre,  ippl::nbody::getRaw<"charge">(pc));
    uploadHost(pxPre, px.data());
    uploadHost(pyPre, py.data());
    uploadHost(pzPre, pz.data());
    uploadHost(idPre, id.data());

    pc.updateGrav<kIdxCharge>(id, px, py, pz);

    const unsigned start      = pc.startIndex();
    const unsigned end        = pc.endIndex();
    const unsigned nWithHalos = pc.nWithHalos();
    ASSERT_LE(start, end);
    ASSERT_LE(end, nWithHalos);
    EXPECT_EQ(end - start, kN) << "Single-rank: every input particle should be locally owned.";

    std::vector<double> xPost, yPost, zPost, pxPost, pyPost, pzPost, qPost;
    std::vector<IdType> idPost;
    downloadDevice(ippl::nbody::getRaw<"Rx">(pc),     nWithHalos, xPost);
    downloadDevice(ippl::nbody::getRaw<"Ry">(pc),     nWithHalos, yPost);
    downloadDevice(ippl::nbody::getRaw<"Rz">(pc),     nWithHalos, zPost);
    downloadDevice(ippl::nbody::getRaw<"charge">(pc), nWithHalos, qPost);
    downloadDevice(id.data(), nWithHalos, idPost);
    downloadDevice(px.data(), nWithHalos, pxPost);
    downloadDevice(py.data(), nWithHalos, pyPost);
    downloadDevice(pz.data(), nWithHalos, pzPost);

    std::unordered_set<IdType> seen;
    seen.reserve(end - start);
    for (unsigned j = start; j < end; ++j) {
        const IdType ii = idPost[j];
        ASSERT_LT(ii, kN) << "ID out of range at slot " << j;
        ASSERT_TRUE(seen.insert(ii).second) << "Duplicate ID " << ii << " at slot " << j;
        EXPECT_DOUBLE_EQ(xPost[j],  xPre[ii])  << "Rx mismatch at j=" << j << " ID=" << ii;
        EXPECT_DOUBLE_EQ(yPost[j],  yPre[ii])  << "Ry mismatch at j=" << j << " ID=" << ii;
        EXPECT_DOUBLE_EQ(zPost[j],  zPre[ii])  << "Rz mismatch at j=" << j << " ID=" << ii;
        EXPECT_DOUBLE_EQ(pxPost[j], pxPre[ii]) << "Px mismatch at j=" << j << " ID=" << ii;
        EXPECT_DOUBLE_EQ(pyPost[j], pyPre[ii]) << "Py mismatch at j=" << j << " ID=" << ii;
        EXPECT_DOUBLE_EQ(pzPost[j], pzPre[ii]) << "Pz mismatch at j=" << j << " ID=" << ii;
        EXPECT_DOUBLE_EQ(qPost[j],  qPre[ii])  << "charge mismatch at j=" << j << " ID=" << ii;
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
