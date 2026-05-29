// Basic correctness tests for the bespoke leapfrog kernels.
// Each test runs one kernel in isolation and checks the math holds bit-for-bit
// against a host-side reference. No BH solver, no oscillator dynamics.

#include <cstdint>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "Ippl.h"

#include "NBody/SphexaParticleContainer.hpp"
#include "NBody/LeapfrogStepper.hpp"

using ippl::nbody::DoublePrecision;
using ippl::nbody::leapfrogDrift;
using ippl::nbody::leapfrogKickHalf;
using ippl::nbody::SphexaParticleContainer;

namespace {

constexpr unsigned kN             = 256;
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

    pc.create(kN);

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
    uploadHost(pxPre, getRaw<"Px">(pc));
    uploadHost(pyPre, getRaw<"Py">(pc));
    uploadHost(pzPre, getRaw<"Pz">(pc));
    uploadHost(idPre, getRaw<"ID">(pc));

    // update() to populate startIndex()/endIndex(); the SFC sort permutes the
    // input arrays, so we re-snapshot R and P after sync to compute the host
    // reference for the *post-sync* state.
    pc.update();

    const unsigned start      = pc.startIndex();
    const unsigned end        = pc.endIndex();
    const unsigned nWithHalos = pc.nWithHalos();
    ASSERT_EQ(end - start, kN);

    std::vector<T> xMid, yMid, zMid, pxMid, pyMid, pzMid;
    downloadDevice(getRaw<"Rx">(pc), nWithHalos, xMid);
    downloadDevice(getRaw<"Ry">(pc), nWithHalos, yMid);
    downloadDevice(getRaw<"Rz">(pc), nWithHalos, zMid);
    downloadDevice(getRaw<"Px">(pc), kN, pxMid);
    downloadDevice(getRaw<"Py">(pc), kN, pyMid);
    downloadDevice(getRaw<"Pz">(pc), kN, pzMid);

    const T dt = 1.25e-3;
    leapfrogDrift<P>(pc, dt);

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

    pc.create(kN);

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
    uploadHost(pxPre, getRaw<"Px">(pc));
    uploadHost(pyPre, getRaw<"Py">(pc));
    uploadHost(pzPre, getRaw<"Pz">(pc));
    uploadHost(exPre, getRaw<"Ex">(pc));
    uploadHost(eyPre, getRaw<"Ey">(pc));
    uploadHost(ezPre, getRaw<"Ez">(pc));
    uploadHost(idPre, getRaw<"ID">(pc));

    pc.update();

    const unsigned start = pc.startIndex();
    const unsigned end   = pc.endIndex();
    ASSERT_EQ(end - start, kN);

    // After update(), positions are SFC-permuted but P and E (not in any sync
    // tuple) are unchanged byte-for-byte at indices [0, kN). Snapshot them as
    // the host reference, then run kickHalf and verify P_post == P_mid - 0.5*dt*E.
    std::vector<T> pxMid, pyMid, pzMid, exMid, eyMid, ezMid;
    downloadDevice(getRaw<"Px">(pc), kN, pxMid);
    downloadDevice(getRaw<"Py">(pc), kN, pyMid);
    downloadDevice(getRaw<"Pz">(pc), kN, pzMid);
    downloadDevice(getRaw<"Ex">(pc), kN, exMid);
    downloadDevice(getRaw<"Ey">(pc), kN, eyMid);
    downloadDevice(getRaw<"Ez">(pc), kN, ezMid);

    const T dt = 2.5e-3;
    leapfrogKickHalf<P>(pc, dt);

    std::vector<T> pxPost, pyPost, pzPost;
    downloadDevice(getRaw<"Px">(pc), kN, pxPost);
    downloadDevice(getRaw<"Py">(pc), kN, pyPost);
    downloadDevice(getRaw<"Pz">(pc), kN, pzPost);

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
