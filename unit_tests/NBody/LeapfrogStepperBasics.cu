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
    using cstone::BoundaryType;

    SphexaParticleContainer<T, 3> pc(
        /*rank=*/0, /*nRanks=*/1,
        kBucketSize, kBucketSizeFoc, kTheta,
        std::array<T, 6>{0.0, 1.0, 0.0, 1.0, 0.0, 1.0},
        std::array<BoundaryType, 3>{
            BoundaryType::open, BoundaryType::open, BoundaryType::open});

    pc.create(kN);

    std::vector<T> xPre(kN), yPre(kN), zPre(kN), hPre(kN, 1.0e-2);
    std::vector<T> pxPre(kN), pyPre(kN), pzPre(kN);
    std::vector<typename SphexaParticleContainer<T, 3>::IdType> idPre(kN);
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

    uploadHost(xPre,  pc.getRxRaw());
    uploadHost(yPre,  pc.getRyRaw());
    uploadHost(zPre,  pc.getRzRaw());
    uploadHost(hPre,  pc.getHRaw());
    uploadHost(pxPre, pc.getPxRaw());
    uploadHost(pyPre, pc.getPyRaw());
    uploadHost(pzPre, pc.getPzRaw());
    uploadHost(idPre, pc.getIDRaw());

    // update() to populate startIndex()/endIndex(); the SFC sort permutes the
    // input arrays, so we re-snapshot R and P after sync to compute the host
    // reference for the *post-sync* state.
    pc.update();

    const unsigned start      = pc.startIndex();
    const unsigned end        = pc.endIndex();
    const unsigned nWithHalos = pc.nWithHalos();
    ASSERT_EQ(end - start, kN);

    std::vector<T> xMid, yMid, zMid, pxMid, pyMid, pzMid;
    downloadDevice(pc.getRxRaw(), nWithHalos, xMid);
    downloadDevice(pc.getRyRaw(), nWithHalos, yMid);
    downloadDevice(pc.getRzRaw(), nWithHalos, zMid);
    downloadDevice(pc.getPxRaw(), kN, pxMid);
    downloadDevice(pc.getPyRaw(), kN, pyMid);
    downloadDevice(pc.getPzRaw(), kN, pzMid);

    const T dt = 1.25e-3;
    leapfrogDrift<T>(pc, dt);

    std::vector<T> xPost, yPost, zPost;
    downloadDevice(pc.getRxRaw(), nWithHalos, xPost);
    downloadDevice(pc.getRyRaw(), nWithHalos, yPost);
    downloadDevice(pc.getRzRaw(), nWithHalos, zPost);

    for (unsigned j = start; j < end; ++j) {
        EXPECT_DOUBLE_EQ(xPost[j], xMid[j] + dt * pxMid[j]) << "j=" << j;
        EXPECT_DOUBLE_EQ(yPost[j], yMid[j] + dt * pyMid[j]) << "j=" << j;
        EXPECT_DOUBLE_EQ(zPost[j], zMid[j] + dt * pzMid[j]) << "j=" << j;
    }
}

TEST(LeapfrogStepper, KickHalfReducesVelocityByHalfDtE) {
    using T = double;
    using cstone::BoundaryType;

    SphexaParticleContainer<T, 3> pc(
        /*rank=*/0, /*nRanks=*/1,
        kBucketSize, kBucketSizeFoc, kTheta,
        std::array<T, 6>{0.0, 1.0, 0.0, 1.0, 0.0, 1.0},
        std::array<BoundaryType, 3>{
            BoundaryType::open, BoundaryType::open, BoundaryType::open});

    pc.create(kN);

    std::vector<T> xPre(kN), yPre(kN), zPre(kN), hPre(kN, 1.0e-2);
    std::vector<T> pxPre(kN), pyPre(kN), pzPre(kN);
    std::vector<T> exPre(kN), eyPre(kN), ezPre(kN);
    std::vector<typename SphexaParticleContainer<T, 3>::IdType> idPre(kN);
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

    uploadHost(xPre,  pc.getRxRaw());
    uploadHost(yPre,  pc.getRyRaw());
    uploadHost(zPre,  pc.getRzRaw());
    uploadHost(hPre,  pc.getHRaw());
    uploadHost(pxPre, pc.getPxRaw());
    uploadHost(pyPre, pc.getPyRaw());
    uploadHost(pzPre, pc.getPzRaw());
    uploadHost(exPre, pc.getExRaw());
    uploadHost(eyPre, pc.getEyRaw());
    uploadHost(ezPre, pc.getEzRaw());
    uploadHost(idPre, pc.getIDRaw());

    pc.update();

    const unsigned start = pc.startIndex();
    const unsigned end   = pc.endIndex();
    ASSERT_EQ(end - start, kN);

    // After update(), positions are SFC-permuted but P and E (not in any sync
    // tuple) are unchanged byte-for-byte at indices [0, kN). Snapshot them as
    // the host reference, then run kickHalf and verify P_post == P_mid - 0.5*dt*E.
    std::vector<T> pxMid, pyMid, pzMid, exMid, eyMid, ezMid;
    downloadDevice(pc.getPxRaw(), kN, pxMid);
    downloadDevice(pc.getPyRaw(), kN, pyMid);
    downloadDevice(pc.getPzRaw(), kN, pzMid);
    downloadDevice(pc.getExRaw(), kN, exMid);
    downloadDevice(pc.getEyRaw(), kN, eyMid);
    downloadDevice(pc.getEzRaw(), kN, ezMid);

    const T dt = 2.5e-3;
    leapfrogKickHalf<T>(pc, dt);

    std::vector<T> pxPost, pyPost, pzPost;
    downloadDevice(pc.getPxRaw(), kN, pxPost);
    downloadDevice(pc.getPyRaw(), kN, pyPost);
    downloadDevice(pc.getPzRaw(), kN, pzPost);

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
