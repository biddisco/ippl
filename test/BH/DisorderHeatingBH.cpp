// Disorder-Induced Heating — Barnes-Hut backend.
//
// Mirrors the sphexa DIH driver (fmm/test/disorder_heating.cu) but routes
// the BH pipeline through ippl::nbody::NBodySolver. Simulation logic
// lives in DisorderHeatingBHManager (test/BH/DisorderHeatingBHManager.hpp).
//
// Two IC modes:
//   DisorderHeatingBH file <dih_initial_positions.bin> [Nt]
//   DisorderHeatingBH gen  <N>                          [Nt] [seed]
//
// Optional everywhere: [--precision=double|mixed|float]
//   selects the BH precision policy (NBody/BHPrecision.hpp). Default `double`
//   preserves bit-for-bit behaviour for callers that don't pass the flag.
//
// File mode reproduces the sphexa reference run from a pre-baked binary
// (rank 0 reads + MPI_Scatterv; intended for the fixed-N validation against
// the sphexa+IPPL P3M references). Generator mode samples a per-rank uniform-
// in-sphere distribution at arbitrary N (intended for scaling sweeps).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "Ippl.h"
#include "Utility/IpplException.h"
#include "Utility/IpplTimings.h"

#include "DisorderHeatingBHManager.hpp"
#include "IpplArgs.hpp"

const char* TestName = "DisorderHeatingBH";

namespace {

void printUsage(const char* argv0) {
    std::fprintf(stderr,
                 "Usage:\n"
                 "  %s file <dih_initial_positions.bin> [Nt] [--precision=double|mixed|float]\n"
                 "  %s gen  <N>                          [Nt] [seed] [--precision=double|mixed|float]\n"
                 "\n"
                 "file mode: load the sphexa reference .bin (rank-0 read +\n"
                 "           MPI_Scatterv). N comes from the file header.\n"
                 "gen  mode: per-rank uniform-in-sphere generator, no file IO.\n",
                 argv0, argv0);
}

template <class P>
int runFile(std::uint64_t globalN,
            std::vector<double> hx, std::vector<double> hy, std::vector<double> hz,
            int Nt) {
    using Manager = ippl::nbody::DisorderHeatingBHManager<P, 3>;
    typename Manager::Config cfg;
    cfg.mode = Manager::Mode::File;
    cfg.Nt   = Nt;

    Manager manager(globalN, std::move(hx), std::move(hy), std::move(hz), cfg);
    if (ippl::Comm->rank() == 0) {
        std::printf("Pre Run: computing initial forces...\n");
    }
    manager.pre_run();
    if (ippl::Comm->rank() == 0) {
        std::printf("Starting iterations ...\n");
        std::fflush(stdout);
    }
    manager.run(manager.getNt());
    if (ippl::Comm->rank() == 0) {
        std::printf("Simulation finished.\n");
    }
    return 0;
}

template <class P>
int runGen(std::uint64_t globalN, int Nt, unsigned long seed) {
    using Manager = ippl::nbody::DisorderHeatingBHManager<P, 3>;
    typename Manager::Config cfg;
    cfg.mode = Manager::Mode::Generator;
    cfg.Nt   = Nt;
    cfg.seed = seed;

    Manager manager(globalN, cfg);
    if (ippl::Comm->rank() == 0) {
        std::printf("Pre Run: computing initial forces...\n");
    }
    manager.pre_run();
    if (ippl::Comm->rank() == 0) {
        std::printf("Starting iterations ...\n");
        std::fflush(stdout);
    }
    manager.run(manager.getNt());
    if (ippl::Comm->rank() == 0) {
        std::printf("Simulation finished.\n");
    }
    return 0;
}

}  // namespace

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);

    if (ippl::Info->getOutputLevel() < 1) {
        ippl::Info->setOutputLevel(1);
    }
    int returnCode = 0;
    {
        Inform msg(TestName);

        auto pos = ippl::nbody::filterIpplFlags(argc, argv);

        std::string precision;
        try {
            precision = ippl::nbody::extractPrecisionFlag(pos);
        } catch (const std::exception& e) {
            if (ippl::Comm->rank() == 0) { std::fprintf(stderr, "%s\n", e.what()); }
            ippl::finalize();
            return 2;
        }

        if (pos.size() < 3) {
            if (ippl::Comm->rank() == 0) { printUsage(pos[0]); }
            ippl::finalize();
            return 2;
        }
        const std::string mode = pos[1];

        static IpplTimings::TimerRef mainTimer = IpplTimings::getTimer("total");
        IpplTimings::startTimer(mainTimer);

        try {
            if (mode == "file") {
                const std::string posPath = std::string(pos[2]);
                std::vector<double> hx, hy, hz;
                std::uint64_t globalN = 0;
                ippl::nbody::loadDihPositions(posPath, hx, hy, hz, &globalN);
                msg << "[file] precision = " << precision
                    << ", N = " << globalN
                    << ", this rank holds " << hx.size() << endl;
                if (ippl::Comm->rank() == 0) {
                    std::printf("Loaded %llu particles from %s\n",
                                static_cast<unsigned long long>(globalN), posPath.c_str());
                }

                int Nt = 1000;
                if (pos.size() > 3) Nt = std::atoi(pos[3]);
                msg << "Nt = " << Nt << endl;

                if (precision == "double") {
                    returnCode = runFile<ippl::nbody::DoublePrecision>(
                        globalN, std::move(hx), std::move(hy), std::move(hz), Nt);
                } else if (precision == "mixed") {
                    returnCode = runFile<ippl::nbody::MixedPrecision>(
                        globalN, std::move(hx), std::move(hy), std::move(hz), Nt);
                } else /* "float" */ {
                    returnCode = runFile<ippl::nbody::FloatPrecision>(
                        globalN, std::move(hx), std::move(hy), std::move(hz), Nt);
                }
            } else if (mode == "gen") {
                const std::uint64_t globalN =
                    static_cast<std::uint64_t>(std::atoll(pos[2]));
                int           Nt   = 1000;
                unsigned long seed = 42;
                if (pos.size() > 3) Nt   = std::atoi(pos[3]);
                if (pos.size() > 4) seed = std::atoll(pos[4]);
                msg << "[gen] precision = " << precision
                    << ", N = " << globalN
                    << ", Nt = " << Nt
                    << ", seed = " << seed << endl;

                if (precision == "double") {
                    returnCode = runGen<ippl::nbody::DoublePrecision>(globalN, Nt, seed);
                } else if (precision == "mixed") {
                    returnCode = runGen<ippl::nbody::MixedPrecision>(globalN, Nt, seed);
                } else /* "float" */ {
                    returnCode = runGen<ippl::nbody::FloatPrecision>(globalN, Nt, seed);
                }
            } else {
                if (ippl::Comm->rank() == 0) {
                    std::fprintf(stderr, "Unknown mode '%s'\n\n", mode.c_str());
                    printUsage(pos[0]);
                }
                returnCode = 2;
            }
        } catch (const IpplException& e) {
            std::fprintf(stderr, "[rank %d] IpplException in %s: %s\n",
                         ippl::Comm->rank(), e.where().c_str(), e.what());
            returnCode = 1;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[rank %d] std::exception: %s\n",
                         ippl::Comm->rank(), e.what());
            returnCode = 1;
        }

        IpplTimings::stopTimer(mainTimer);
        IpplTimings::print();
        IpplTimings::print(std::string("timing.dat"));
    }
    ippl::finalize();
    return returnCode;
}
