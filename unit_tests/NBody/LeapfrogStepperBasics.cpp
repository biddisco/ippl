// Basic correctness tests for the bespoke leapfrog kernels.
// Each test runs one kernel in isolation and checks the math holds bit-for-bit
// against a host-side reference. No BH solver, no oscillator dynamics.

#include <cstdint>
#include <cstdlib>
#include <vector>

#include <gtest/gtest.h>

#include "Ippl.h"

#include "NBody/SphexaParticleContainer.hpp"
#include "NBody/LeapfrogStepper.hpp"
#include "NBodyTestUtil.hpp"

using ippl::nbody::DoublePrecision;
using ippl::nbody::FieldVector;
using ippl::nbody::leapfrogDrift;
using ippl::nbody::leapfrogKickHalf;
using ippl::nbody::SphexaParticleContainer;
using ippl::nbody::test::downloadDevice;
using ippl::nbody::test::uploadHost;

namespace {

constexpr unsigned kN             = 256;
constexpr unsigned kBucketSize    = 64;
constexpr unsigned kBucketSizeFoc = 64;
constexpr float    kTheta         = 0.5f;

} // namespace

TEST(LeapfrogStepper, DriftAdvancesPositionLinearly) {
    using T = double;
    using P = DoublePrecision;
    using cstone::BoundaryType;

    SphexaParticleContainer<P, 3> pc(
        /*rank=*/0, /*nRanks=*/1,
        kBucketSize, kBucketSizeFoc, kTheta,
        std::array<T, 6>{0.0, 1.0, 0.0, 1.0, 0.0, 1.0},
        std::array<BoundaryType, 3>{
            BoundaryType::open, BoundaryType::open, BoundaryType::open});

    pc.create(kN); FieldVector<typename SphexaParticleContainer<P,3>::IdType> deviceID; deviceID.resize(kN); FieldVector<T> dPx, dPy, dPz; dPx.resize(kN); dPy.resize(kN); dPz.resize(kN);

    std::vector<T> xPre(kN), yPre(kN), zPre(kN), hPre(kN, 1.0e-2);
    std::vector<T> pxPre(kN), pyPre(kN), pzPre(kN);
    std::vector<typename SphexaParticleContainer<P, 3>::IdType> idPre(kN);
    ::srand48(/*seed=*/131);
    for (unsigned i = 0; i < kN; ++i) {
        xPre[i]  = drand48();
        yPre[i]  = drand48();
        zPre[i]  = drand48();
        pxPre[i] = drand48() - 0.5;
        pyPre[i] = drand48() - 0.5;
        pzPre[i] = drand48() - 0.5;
        idPre[i] = i;
    }

    uploadHost(xPre,  getRaw<"Rx">(pc));
    uploadHost(yPre,  getRaw<"Ry">(pc));
    uploadHost(zPre,  getRaw<"Rz">(pc));
    uploadHost(hPre,  getRaw<"h">(pc));
    uploadHost(pxPre, dPx.data());
    uploadHost(pyPre, dPy.data());
    uploadHost(pzPre, dPz.data());
    uploadHost(idPre, deviceID.data());

    // update() to populate startIndex()/endIndex(); the SFC sort permutes the
    // input arrays, so we re-snapshot R and P after sync to compute the host
    // reference for the *post-sync* state.
    pc.update(deviceID, dPx, dPy, dPz);

    const unsigned start      = pc.startIndex();
    const unsigned end        = pc.endIndex();
    const unsigned nWithHalos = pc.nWithHalos();
    ASSERT_EQ(end - start, kN);

    std::vector<T> xMid, yMid, zMid, pxMid, pyMid, pzMid;
    downloadDevice(getRaw<"Rx">(pc), nWithHalos, xMid);
    downloadDevice(getRaw<"Ry">(pc), nWithHalos, yMid);
    downloadDevice(getRaw<"Rz">(pc), nWithHalos, zMid);
    downloadDevice(dPx.data(), kN, pxMid);
    downloadDevice(dPy.data(), kN, pyMid);
    downloadDevice(dPz.data(), kN, pzMid);

    const T dt = 1.25e-3;
    leapfrogDrift<P>(pc, dPx, dPy, dPz, dt);

    std::vector<T> xPost, yPost, zPost;
    downloadDevice(getRaw<"Rx">(pc), nWithHalos, xPost);
    downloadDevice(getRaw<"Ry">(pc), nWithHalos, yPost);
    downloadDevice(getRaw<"Rz">(pc), nWithHalos, zPost);

    for (unsigned j = start; j < end; ++j) {
        EXPECT_DOUBLE_EQ(xPost[j], xMid[j] + dt * pxMid[j]) << "j=" << j;
        EXPECT_DOUBLE_EQ(yPost[j], yMid[j] + dt * pyMid[j]) << "j=" << j;
        EXPECT_DOUBLE_EQ(zPost[j], zMid[j] + dt * pzMid[j]) << "j=" << j;
    }
}

TEST(LeapfrogStepper, KickHalfReducesVelocityByHalfDtE) {
    using T = double;
    using P = DoublePrecision;
    using cstone::BoundaryType;

    SphexaParticleContainer<P, 3> pc(
        /*rank=*/0, /*nRanks=*/1,
        kBucketSize, kBucketSizeFoc, kTheta,
        std::array<T, 6>{0.0, 1.0, 0.0, 1.0, 0.0, 1.0},
        std::array<BoundaryType, 3>{
            BoundaryType::open, BoundaryType::open, BoundaryType::open});

    pc.create(kN); FieldVector<typename SphexaParticleContainer<P,3>::IdType> deviceID; deviceID.resize(kN); FieldVector<T> dPx, dPy, dPz; dPx.resize(kN); dPy.resize(kN); dPz.resize(kN);

    std::vector<T> xPre(kN), yPre(kN), zPre(kN), hPre(kN, 1.0e-2);
    std::vector<T> pxPre(kN), pyPre(kN), pzPre(kN);
    std::vector<T> exPre(kN), eyPre(kN), ezPre(kN);
    std::vector<typename SphexaParticleContainer<P, 3>::IdType> idPre(kN);
    ::srand48(/*seed=*/132);
    for (unsigned i = 0; i < kN; ++i) {
        xPre[i]  = drand48();
        yPre[i]  = drand48();
        zPre[i]  = drand48();
        pxPre[i] = drand48() - 0.5;
        pyPre[i] = drand48() - 0.5;
        pzPre[i] = drand48() - 0.5;
        exPre[i] = drand48() - 0.5;
        eyPre[i] = drand48() - 0.5;
        ezPre[i] = drand48() - 0.5;
        idPre[i] = i;
    }

    uploadHost(xPre,  getRaw<"Rx">(pc));
    uploadHost(yPre,  getRaw<"Ry">(pc));
    uploadHost(zPre,  getRaw<"Rz">(pc));
    uploadHost(hPre,  getRaw<"h">(pc));
    uploadHost(pxPre, dPx.data());
    uploadHost(pyPre, dPy.data());
    uploadHost(pzPre, dPz.data());
    uploadHost(exPre, getRaw<"Ex">(pc));
    uploadHost(eyPre, getRaw<"Ey">(pc));
    uploadHost(ezPre, getRaw<"Ez">(pc));
    uploadHost(idPre, deviceID.data());

    pc.update(deviceID, dPx, dPy, dPz);

    const unsigned start = pc.startIndex();
    const unsigned end   = pc.endIndex();
    ASSERT_EQ(end - start, kN);

    // After update(), positions are SFC-permuted but P and E (not in any sync
    // tuple) are unchanged byte-for-byte at indices [0, kN). Snapshot them as
    // the host reference, then run kickHalf and verify P_post == P_mid - 0.5*dt*E.
    std::vector<T> pxMid, pyMid, pzMid, exMid, eyMid, ezMid;
    downloadDevice(dPx.data(), kN, pxMid);
    downloadDevice(dPy.data(), kN, pyMid);
    downloadDevice(dPz.data(), kN, pzMid);
    downloadDevice(getRaw<"Ex">(pc), kN, exMid);
    downloadDevice(getRaw<"Ey">(pc), kN, eyMid);
    downloadDevice(getRaw<"Ez">(pc), kN, ezMid);

    const T dt = 2.5e-3;
    leapfrogKickHalf<P>(pc, dPx, dPy, dPz, dt);

    std::vector<T> pxPost, pyPost, pzPost;
    downloadDevice(dPx.data(), kN, pxPost);
    downloadDevice(dPy.data(), kN, pyPost);
    downloadDevice(dPz.data(), kN, pzPost);

    const T halfDt = T(0.5) * dt;
    for (unsigned j = start; j < end; ++j) {
        EXPECT_DOUBLE_EQ(pxPost[j], pxMid[j] - halfDt * exMid[j]) << "j=" << j;
        EXPECT_DOUBLE_EQ(pyPost[j], pyMid[j] - halfDt * eyMid[j]) << "j=" << j;
        EXPECT_DOUBLE_EQ(pzPost[j], pzMid[j] - halfDt * ezMid[j]) << "j=" << j;
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
