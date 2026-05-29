// Periodic-wrap kernel tests. Three sub-tests:
//   1. AllPeriodicWrapsIntoBox  — all-periodic BCs, scattered positions wrap inside.
//   2. OpenBCsLeavePositionsUntouched — open BCs are a no-op (no kernel launch).
//   3. MixedBCsWrapOnlyPeriodicAxes — periodic in x, open in y and z.

#include <cstdint>
#include <cstdlib>
#include <vector>

#include <gtest/gtest.h>

#include "Ippl.h"

#include "NBody/SphexaParticleContainer.hpp"
#include "NBody/PeriodicWrap.hpp"
#include "NBodyTestUtil.hpp"

using ippl::nbody::DoublePrecision;
using ippl::nbody::FieldVector;
using ippl::nbody::SphexaParticleContainer;
using ippl::nbody::wrapToBox;
using ippl::nbody::test::downloadDevice;
using ippl::nbody::test::uploadHost;

namespace {

constexpr unsigned kN             = 256;
constexpr unsigned kBucketSize    = 64;
constexpr unsigned kBucketSizeFoc = 64;
constexpr float    kTheta         = 0.5f;

// Generate kN positions scattered within one box-length outside the unit box,
// i.e. [-0.9, 1.9)^3, so a single-box-length wrap (cstone::putInBox, matching the
// post-drift wrap) brings every periodic axis back into [0, 1). A particle never
// drifts more than one box-length per step, so this is the physical regime.
// Returns by reference. h is uniform; ID is i. BC choice is the caller's job.
template <class P>
void seedScatteredPositions(unsigned long seed,
                            std::vector<typename P::Tc>& x,
                            std::vector<typename P::Tc>& y,
                            std::vector<typename P::Tc>& z,
                            std::vector<typename P::Th>& h,
                            std::vector<typename SphexaParticleContainer<P, 3>::IdType>& id) {
    using Tc = typename P::Tc;
    using Th = typename P::Th;
    x.resize(kN); y.resize(kN); z.resize(kN);
    h.assign(kN, Th(1.0e-2));
    id.resize(kN);
    ::srand48(seed);
    for (unsigned i = 0; i < kN; ++i) {
        x[i]  = Tc(-0.9 + 2.8 * drand48());
        y[i]  = Tc(-0.9 + 2.8 * drand48());
        z[i]  = Tc(-0.9 + 2.8 * drand48());
        id[i] = i;
    }
}

// Single-box-length wrap, matching cstone::putInBox: shift by one L if outside
// [lo, lo+L]. Valid because the scatter above stays within ±1 box-length.
template <class T>
T expectedWrap(T r, T lo, T L) {
    const T hi = lo + L;
    if (r > hi) { return r - L; }
    if (r < lo) { return r + L; }
    return r;
}

} // namespace

TEST(PeriodicWrap, AllPeriodicWrapsIntoBox) {
    using T = double;
    using P = DoublePrecision;
    using cstone::BoundaryType;

    SphexaParticleContainer<P, 3> pc(
        /*rank=*/0, /*nRanks=*/1,
        kBucketSize, kBucketSizeFoc, kTheta,
        std::array<T, 6>{0.0, 1.0, 0.0, 1.0, 0.0, 1.0},
        std::array<BoundaryType, 3>{
            BoundaryType::periodic, BoundaryType::periodic, BoundaryType::periodic});

    pc.create(kN); FieldVector<typename SphexaParticleContainer<P,3>::IdType> deviceID; deviceID.resize(kN);

    std::vector<T> xPre, yPre, zPre, hPre;
    std::vector<typename SphexaParticleContainer<P, 3>::IdType> idPre;
    seedScatteredPositions<P>(/*seed=*/231, xPre, yPre, zPre, hPre, idPre);

    uploadHost(xPre,  getRaw<"Rx">(pc));
    uploadHost(yPre,  getRaw<"Ry">(pc));
    uploadHost(zPre,  getRaw<"Rz">(pc));
    uploadHost(hPre,  getRaw<"h">(pc));
    uploadHost(idPre, deviceID.data());

    // pc.update(deviceID) with out-of-box positions would fault inside cstone's SFC key
    // computation. Sync once with in-box positions to populate startIndex/endIndex,
    // then overwrite with the scattered set and run wrap. This mirrors the real-world
    // flow: drift produces possibly-out-of-box R, wrapToBox brings it back, then
    // updateGrav re-syncs the tree.
    {
        std::vector<T> xClamp(kN), yClamp(kN), zClamp(kN);
        ::srand48(/*seed=*/2310);
        for (unsigned i = 0; i < kN; ++i) {
            xClamp[i] = drand48();
            yClamp[i] = drand48();
            zClamp[i] = drand48();
        }
        uploadHost(xClamp, getRaw<"Rx">(pc));
        uploadHost(yClamp, getRaw<"Ry">(pc));
        uploadHost(zClamp, getRaw<"Rz">(pc));
        pc.update(deviceID);
    }

    uploadHost(xPre, getRaw<"Rx">(pc));
    uploadHost(yPre, getRaw<"Ry">(pc));
    uploadHost(zPre, getRaw<"Rz">(pc));

    wrapToBox<P>(pc);

    const unsigned start = pc.startIndex();
    const unsigned end   = pc.endIndex();
    ASSERT_EQ(end - start, kN);

    std::vector<T> xPost, yPost, zPost;
    downloadDevice(getRaw<"Rx">(pc), end, xPost);
    downloadDevice(getRaw<"Ry">(pc), end, yPost);
    downloadDevice(getRaw<"Rz">(pc), end, zPost);

    for (unsigned j = start; j < end; ++j) {
        EXPECT_GE(xPost[j], 0.0); EXPECT_LT(xPost[j], 1.0) << "x j=" << j;
        EXPECT_GE(yPost[j], 0.0); EXPECT_LT(yPost[j], 1.0) << "y j=" << j;
        EXPECT_GE(zPost[j], 0.0); EXPECT_LT(zPost[j], 1.0) << "z j=" << j;
    }
}

TEST(PeriodicWrap, OpenBCsLeavePositionsUntouched) {
    using T = double;
    using P = DoublePrecision;
    using cstone::BoundaryType;

    SphexaParticleContainer<P, 3> pc(
        /*rank=*/0, /*nRanks=*/1,
        kBucketSize, kBucketSizeFoc, kTheta,
        std::array<T, 6>{0.0, 1.0, 0.0, 1.0, 0.0, 1.0},
        std::array<BoundaryType, 3>{
            BoundaryType::open, BoundaryType::open, BoundaryType::open});

    pc.create(kN); FieldVector<typename SphexaParticleContainer<P,3>::IdType> deviceID; deviceID.resize(kN);

    // Sync with in-box positions so the container has valid start/end indices.
    {
        std::vector<T> x(kN), y(kN), z(kN), h(kN, 1.0e-2);
        std::vector<typename SphexaParticleContainer<P, 3>::IdType> id(kN);
        ::srand48(/*seed=*/2320);
        for (unsigned i = 0; i < kN; ++i) {
            x[i] = drand48(); y[i] = drand48(); z[i] = drand48();
            id[i] = i;
        }
        uploadHost(x,  getRaw<"Rx">(pc));
        uploadHost(y,  getRaw<"Ry">(pc));
        uploadHost(z,  getRaw<"Rz">(pc));
        uploadHost(h,  getRaw<"h">(pc));
        uploadHost(id, deviceID.data());
        pc.update(deviceID);
    }

    // Now overwrite with deliberately-out-of-box positions.
    std::vector<T> xPre, yPre, zPre, hPre;
    std::vector<typename SphexaParticleContainer<P, 3>::IdType> idPre;
    seedScatteredPositions<P>(/*seed=*/233, xPre, yPre, zPre, hPre, idPre);
    uploadHost(xPre, getRaw<"Rx">(pc));
    uploadHost(yPre, getRaw<"Ry">(pc));
    uploadHost(zPre, getRaw<"Rz">(pc));

    wrapToBox<P>(pc);

    const unsigned start = pc.startIndex();
    const unsigned end   = pc.endIndex();

    std::vector<T> xPost, yPost, zPost;
    downloadDevice(getRaw<"Rx">(pc), end, xPost);
    downloadDevice(getRaw<"Ry">(pc), end, yPost);
    downloadDevice(getRaw<"Rz">(pc), end, zPost);

    // Open BCs → wrapToBox is a no-op (no kernel launch). Bytes should match
    // exactly what we uploaded.
    for (unsigned j = start; j < end; ++j) {
        const unsigned src = j - start;
        EXPECT_DOUBLE_EQ(xPost[j], xPre[src]) << "j=" << j;
        EXPECT_DOUBLE_EQ(yPost[j], yPre[src]) << "j=" << j;
        EXPECT_DOUBLE_EQ(zPost[j], zPre[src]) << "j=" << j;
    }
}

TEST(PeriodicWrap, MixedBCsWrapOnlyPeriodicAxes) {
    using T = double;
    using P = DoublePrecision;
    using cstone::BoundaryType;

    SphexaParticleContainer<P, 3> pc(
        /*rank=*/0, /*nRanks=*/1,
        kBucketSize, kBucketSizeFoc, kTheta,
        std::array<T, 6>{0.0, 1.0, 0.0, 1.0, 0.0, 1.0},
        std::array<BoundaryType, 3>{
            BoundaryType::periodic, BoundaryType::open, BoundaryType::open});

    pc.create(kN); FieldVector<typename SphexaParticleContainer<P,3>::IdType> deviceID; deviceID.resize(kN);

    {
        std::vector<T> x(kN), y(kN), z(kN), h(kN, 1.0e-2);
        std::vector<typename SphexaParticleContainer<P, 3>::IdType> id(kN);
        ::srand48(/*seed=*/2330);
        for (unsigned i = 0; i < kN; ++i) {
            x[i] = drand48(); y[i] = drand48(); z[i] = drand48();
            id[i] = i;
        }
        uploadHost(x,  getRaw<"Rx">(pc));
        uploadHost(y,  getRaw<"Ry">(pc));
        uploadHost(z,  getRaw<"Rz">(pc));
        uploadHost(h,  getRaw<"h">(pc));
        uploadHost(id, deviceID.data());
        pc.update(deviceID);
    }

    std::vector<T> xPre, yPre, zPre, hPre;
    std::vector<typename SphexaParticleContainer<P, 3>::IdType> idPre;
    seedScatteredPositions<P>(/*seed=*/235, xPre, yPre, zPre, hPre, idPre);
    uploadHost(xPre, getRaw<"Rx">(pc));
    uploadHost(yPre, getRaw<"Ry">(pc));
    uploadHost(zPre, getRaw<"Rz">(pc));

    wrapToBox<P>(pc);

    const unsigned start = pc.startIndex();
    const unsigned end   = pc.endIndex();

    std::vector<T> xPost, yPost, zPost;
    downloadDevice(getRaw<"Rx">(pc), end, xPost);
    downloadDevice(getRaw<"Ry">(pc), end, yPost);
    downloadDevice(getRaw<"Rz">(pc), end, zPost);

    // x is periodic → wrapped. y and z are open → unchanged.
    for (unsigned j = start; j < end; ++j) {
        const unsigned src = j - start;
        EXPECT_DOUBLE_EQ(xPost[j], expectedWrap<T>(xPre[src], 0.0, 1.0)) << "j=" << j;
        EXPECT_DOUBLE_EQ(yPost[j], yPre[src]) << "j=" << j;
        EXPECT_DOUBLE_EQ(zPost[j], zPre[src]) << "j=" << j;
        EXPECT_GE(xPost[j], 0.0); EXPECT_LT(xPost[j], 1.0) << "x j=" << j;
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
