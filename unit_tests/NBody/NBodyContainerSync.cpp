// Sanity test for NBodyParticleContainer<DoublePrecision, 3>.
// Verifies that updateBH<ConservedFields>(...) (which wraps cstone::Domain::sync)
// permutes the container-resident ID array in lockstep with positions.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <unordered_set>
#include <vector>

#include <gtest/gtest.h>

#include "Ippl.h"
#include "cstone/sfc/box.hpp"

#include "NBody/NBodyParticleContainer.hpp"
#include "NBodyTestUtil.hpp"

using ippl::nbody::DoublePrecision;
using ippl::nbody::NBodyParticleContainer;
using ippl::nbody::updateBH;
using ippl::nbody::test::downloadDevice;
using ippl::nbody::test::uploadHost;
namespace fields = ippl::nbody::fields;

namespace {

using PC     = NBodyParticleContainer<DoublePrecision, 3>;
using IdType = PC::IdType;

constexpr unsigned kN              = 8192;
constexpr unsigned kBucketSize     = 64;
constexpr unsigned kBucketSizeFoc  = 64;
constexpr float    kTheta          = 0.5f;

void verifyLockstep(PC& pc,
                    const std::vector<double>& xPre,
                    const std::vector<double>& yPre,
                    const std::vector<double>& zPre) {
    const unsigned start      = pc.startIndex();
    const unsigned end        = pc.endIndex();
    const unsigned nWithHalos = pc.nWithHalos();

    ASSERT_LE(start, end);
    ASSERT_LE(end, nWithHalos);
    EXPECT_EQ(end - start, kN) << "Single-rank: every input particle should be locally owned.";

    std::vector<double> xPost, yPost, zPost;
    std::vector<IdType> idPost;
    downloadDevice(ippl::nbody::getRaw<"Rx">(pc), nWithHalos, xPost);
    downloadDevice(ippl::nbody::getRaw<"Ry">(pc), nWithHalos, yPost);
    downloadDevice(ippl::nbody::getRaw<"Rz">(pc), nWithHalos, zPost);
    downloadDevice(ippl::nbody::getRaw<"ID">(pc), nWithHalos, idPost);

    std::unordered_set<IdType> seen;
    seen.reserve(end - start);
    for (unsigned j = start; j < end; ++j) {
        const IdType ii = idPost[j];
        ASSERT_LT(ii, kN) << "ID out of range at slot " << j;
        ASSERT_TRUE(seen.insert(ii).second) << "Duplicate ID " << ii << " at slot " << j;
        EXPECT_DOUBLE_EQ(xPost[j], xPre[ii]) << "Rx mismatch at j=" << j << " ID=" << ii;
        EXPECT_DOUBLE_EQ(yPost[j], yPre[ii]) << "Ry mismatch at j=" << j << " ID=" << ii;
        EXPECT_DOUBLE_EQ(zPost[j], zPre[ii]) << "Rz mismatch at j=" << j << " ID=" << ii;
    }
    EXPECT_EQ(seen.size(), end - start) << "ID set must be a permutation of [0, N).";
}

} // namespace

TEST(NBodyParticleContainer, SyncPermutesAttributesInLockstep) {
    using cstone::BoundaryType;

    PC pc(/*rank=*/0, /*nRanks=*/1,
          kBucketSize, kBucketSizeFoc, kTheta,
          std::array<double, 6>{0.0, 1.0, 0.0, 1.0, 0.0, 1.0},
          std::array<BoundaryType, 3>{
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

    uploadHost(xPre,  ippl::nbody::getRaw<"Rx">(pc));
    uploadHost(yPre,  ippl::nbody::getRaw<"Ry">(pc));
    uploadHost(zPre,  ippl::nbody::getRaw<"Rz">(pc));
    uploadHost(hPre,  ippl::nbody::getRaw<"h">(pc));
    uploadHost(idPre, ippl::nbody::getRaw<"ID">(pc));

    updateBH<DoublePrecision, fields::StdConserved>(pc);
    verifyLockstep(pc, xPre, yPre, zPre);

    updateBH<DoublePrecision, fields::StdConserved>(pc);
    verifyLockstep(pc, xPre, yPre, zPre);
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
