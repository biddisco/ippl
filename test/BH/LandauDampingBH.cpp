// Landau Damping test — Barnes-Hut backend.
//
// Mirrors alpine/LandauDamping.cpp but routes the field solve through
// ippl::nbody::NBodySolver instead of FFT-PIC. Simulation logic lives in
// LandauDampingBHManager (test/BH/LandauDampingBHManager.hpp).
//
//   Usage:
//     LandauDampingBH <N> <Nt> <theta> [seed] [numShells] [smoothH]
//                     [--precision=double|mixed|float] [--leaf-h=on|off]
//
// --precision selects the BH precision policy (NBody/BHPrecision.hpp);
// default is `double`, preserving bit-for-bit behaviour for callers that
// don't pass the flag.
//
// --leaf-h toggles per-particle h = leaf edge length (default on). When off,
// the positional [smoothH] arg is used as a uniform h for the BH P2P
// softening. The first step before any syncGrav always uses [smoothH] since
// the focus tree is not yet populated.
//
// The CIC grid-sampled energy diagnostic is gated by a source-level flag in
// LandauDampingBHManager.hpp (kEnableCicEnergy) and is off in production runs
// — for large N it converges to the particle-sampled estimator.

#include "Ippl.h"
#include "Utility/IpplException.h"
#include "Utility/IpplTimings.h"

#include "IpplArgs.hpp"
#include "LandauDampingBHManager.hpp"

const char* TestName = "LandauDampingBH";

namespace {

// Precision-agnostic CLI inputs. We parse argv once into this struct, then
// dispatch to a templated runner that materialises the Manager's typed Config.
struct CliInputs {
    unsigned long N         = 10000;
    int           Nt        = 25;
    float         theta     = 0.5f;
    unsigned long seed      = 42;
    int           numShells = 1;
    double        smoothH   = 0.05;  // narrowed to P::Th inside runWith<P>
    bool          leafBasedH = true;
};

template <class P>
int runWith(const CliInputs& in, Inform& msg) {
    using Manager = ippl::nbody::LandauDampingBHManager<P, 3>;
    typename Manager::Config cfg;
    cfg.N         = in.N;
    cfg.Nt        = in.Nt;
    cfg.theta     = in.theta;
    cfg.seed      = in.seed;
    cfg.numShells = in.numShells;
    cfg.smoothH    = static_cast<typename P::Th>(in.smoothH);
    cfg.leafBasedH = in.leafBasedH;

    Manager manager(cfg);
    manager.pre_run();

    msg << "Starting iterations ..." << endl;
    manager.run(manager.getNt());
    msg << "End." << endl;
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
        bool        leafBasedH;
        try {
            precision  = ippl::nbody::extractPrecisionFlag(pos);
            leafBasedH = ippl::nbody::extractLeafHFlag(pos, /*defaultOn=*/true);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "%s\n", e.what());
            ippl::finalize();
            return 2;
        }

        CliInputs in;
        in.leafBasedH = leafBasedH;
        if (pos.size() > 1) in.N         = std::atoll(pos[1]);
        if (pos.size() > 2) in.Nt        = std::atoi(pos[2]);
        if (pos.size() > 3) in.theta     = std::atof(pos[3]);
        if (pos.size() > 4) in.seed      = std::atoll(pos[4]);
        if (pos.size() > 5) in.numShells = std::atoi(pos[5]);
        if (pos.size() > 6) in.smoothH   = std::atof(pos[6]);

        msg << "precision = " << precision
            << ", N = "          << in.N
            << ", Nt = "         << in.Nt
            << ", theta = "      << in.theta
            << ", seed = "       << in.seed
            << ", numShells = "  << in.numShells
            << ", smoothH = "    << in.smoothH
            << ", leaf-h = "     << (in.leafBasedH ? "on" : "off") << endl;

        static IpplTimings::TimerRef mainTimer = IpplTimings::getTimer("total");
        IpplTimings::startTimer(mainTimer);

        try {
            if (precision == "double") {
                returnCode = runWith<ippl::nbody::DoublePrecision>(in, msg);
            } else if (precision == "mixed") {
                returnCode = runWith<ippl::nbody::MixedPrecision>(in, msg);
            } else /* "float" */ {
                returnCode = runWith<ippl::nbody::FloatPrecision>(in, msg);
            }
        } catch (const IpplException& e) {
            std::fprintf(stderr, "[rank %d] IpplException in %s: %s\n",
                         ippl::Comm->rank(), e.where().c_str(), e.what());
            IpplTimings::stopTimer(mainTimer);
            ippl::finalize();
            return 1;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[rank %d] std::exception: %s\n",
                         ippl::Comm->rank(), e.what());
            IpplTimings::stopTimer(mainTimer);
            ippl::finalize();
            return 1;
        }

        IpplTimings::stopTimer(mainTimer);
        IpplTimings::print();
        IpplTimings::print(std::string("timing.dat"));
    }
    ippl::finalize();
    return returnCode;
}
