// Sanity test for ippl::nbody::ParticleAttrib + SphexaParticleContainer::addAttribute.
//
// Verifies that user-registered attributes are resized in lockstep with the
// built-in attributes on create(), regardless of registration order, and that
// the device storage is usable end-to-end.

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "Ippl.h"

#include "NBody/BHPrecision.hpp"
#include "NBody/ParticleAttrib.hpp"
#include "NBody/SphexaParticleContainer.hpp"
#include "NBodyTestUtil.hpp"

using ippl::nbody::DoublePrecision;
using ippl::nbody::ParticleAttrib;
using ippl::nbody::SphexaParticleContainer;
using ippl::nbody::test::downloadDevice;
using ippl::nbody::test::uploadHost;

namespace {

constexpr unsigned kN             = 1024;
constexpr unsigned kBucketSize    = 64;
constexpr unsigned kBucketSizeFoc = 64;
constexpr float    kTheta         = 0.5f;

SphexaParticleContainer<DoublePrecision, 3> makeContainer() {
    using cstone::BoundaryType;
    return SphexaParticleContainer<DoublePrecision, 3>(
        /*rank=*/0, /*nRanks=*/1,
        kBucketSize, kBucketSizeFoc, kTheta,
        std::array<double, 6>{0.0, 1.0, 0.0, 1.0, 0.0, 1.0},
        std::array<BoundaryType, 3>{
            BoundaryType::open, BoundaryType::open, BoundaryType::open});
}

} // namespace

TEST(SphexaContainerAttribute, RegisterBeforeCreateSizesAttribute) {
    auto pc = makeContainer();

    ParticleAttrib<float>    tag;
    ParticleAttrib<int>      species;
    ParticleAttrib<uint32_t> color;

    pc.addAttribute(tag);
    pc.addAttribute(species);
    pc.addAttribute(color);

    EXPECT_EQ(tag.size(), 0u);
    EXPECT_EQ(species.size(), 0u);
    EXPECT_EQ(color.size(), 0u);

    pc.create(kN);

    EXPECT_EQ(tag.size(), kN);
    EXPECT_EQ(species.size(), kN);
    EXPECT_EQ(color.size(), kN);
}

TEST(SphexaContainerAttribute, RegisterAfterCreateBackfillsSize) {
    auto pc = makeContainer();
    pc.create(kN);

    ParticleAttrib<float> tag;
    EXPECT_EQ(tag.size(), 0u);

    pc.addAttribute(tag);
    EXPECT_EQ(tag.size(), kN);
}

TEST(SphexaContainerAttribute, DeviceStorageIsUsable) {
    auto pc = makeContainer();

    ParticleAttrib<float> tag;
    pc.addAttribute(tag);
    pc.create(kN);

    std::vector<float> tagPre(kN);
    for (unsigned i = 0; i < kN; ++i) {
        tagPre[i] = static_cast<float>(i) * 0.5f;
    }
    uploadHost(tagPre, tag.data());

    std::vector<float> tagPost;
    downloadDevice(tag.data(), kN, tagPost);

    ASSERT_EQ(tagPost.size(), kN);
    for (unsigned i = 0; i < kN; ++i) {
        EXPECT_FLOAT_EQ(tagPost[i], tagPre[i]) << "at i=" << i;
    }
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
