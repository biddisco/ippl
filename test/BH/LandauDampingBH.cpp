// Landau Damping test — Barnes-Hut backend.
//
// Mirrors alpine/LandauDamping.cpp but routes the field solve through
// ippl::nbody::SphexaBHSolver instead of FFT-PIC. Simulation logic lives in
// LandauDampingBHManager (test/BH/LandauDampingBHManager.hpp).
//
//   Usage:
//     LandauDampingBH <N> <Nt> <theta> [seed] [numShells] [smoothH] [cicGrid]

#include "Ippl.h"
#include "Utility/IpplTimings.h"

#include "IpplArgs.hpp"
#include "LandauDampingBHManager.hpp"

const char* TestName = "LandauDampingBH";

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);

    if (ippl::Info->getOutputLevel() < 1) {
        ippl::Info->setOutputLevel(1);
    }
    {
        Inform msg(TestName);

        auto pos = ippl::nbody::filterIpplFlags(argc, argv);

        ippl::nbody::LandauDampingBHManager<double, 3>::Config cfg;
        if (pos.size() > 1) cfg.N         = std::atoll(pos[1]);
        if (pos.size() > 2) cfg.Nt        = std::atoi(pos[2]);
        if (pos.size() > 3) cfg.theta     = std::atof(pos[3]);
        if (pos.size() > 4) cfg.seed      = std::atoll(pos[4]);
        if (pos.size() > 5) cfg.numShells = std::atoi(pos[5]);
        if (pos.size() > 6) cfg.smoothH   = std::atof(pos[6]);
        if (pos.size() > 7) cfg.cicGrid   = std::atoi(pos[7]);

        msg << "N = "          << cfg.N
            << ", Nt = "       << cfg.Nt
            << ", theta = "    << cfg.theta
            << ", seed = "     << cfg.seed
            << ", numShells = "<< cfg.numShells
            << ", smoothH = "  << cfg.smoothH
            << ", cicGrid = "  << cfg.cicGrid << endl;

        static IpplTimings::TimerRef mainTimer = IpplTimings::getTimer("total");
        IpplTimings::startTimer(mainTimer);

        ippl::nbody::LandauDampingBHManager<double, 3> manager(cfg);
        manager.pre_run();

        msg << "Starting iterations ..." << endl;
        manager.run(manager.getNt());
        msg << "End." << endl;

        IpplTimings::stopTimer(mainTimer);
        IpplTimings::print();
        IpplTimings::print(std::string("timing.dat"));
    }
    ippl::finalize();
    return 0;
}
