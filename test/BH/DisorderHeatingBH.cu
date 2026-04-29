// Disorder-Induced Heating — Barnes-Hut backend.
//
// Mirrors /users/tschwab/internship/fmm/sphexa/fmm/test/disorder_heating.cu
// but routes the BH pipeline through ippl::nbody::SphexaBHSolver. Simulation
// logic lives in DisorderHeatingBHManager (test/BH/DisorderHeatingBHManager.hpp).
//
// Usage:
//   DisorderHeatingBH <dih_initial_positions.bin> [Nt]

#include <cstdio>
#include <string>
#include <vector>

#include "Ippl.h"
#include "Utility/IpplTimings.h"

#include "DisorderHeatingBHManager.hpp"
#include "IpplArgs.hpp"

const char* TestName = "DisorderHeatingBH";

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);

    if (ippl::Info->getOutputLevel() < 1) {
        ippl::Info->setOutputLevel(1);
    }
    {
        Inform msg(TestName);

        auto pos = ippl::nbody::filterIpplFlags(argc, argv);

        if (pos.size() < 2) {
            std::fprintf(stderr,
                         "Usage: %s <dih_initial_positions.bin> [Nt]\n"
                         "  Default path tried: "
                         "/users/tschwab/internship/psi/ippl/build/dih_initial_positions.bin\n",
                         pos[0]);
        }
        const std::string posPath =
            (pos.size() > 1) ? std::string(pos[1])
                             : std::string("/users/tschwab/internship/psi/ippl/build/"
                                           "dih_initial_positions.bin");

        std::vector<double> hx, hy, hz;
        ippl::nbody::loadDihPositions(posPath, hx, hy, hz);
        msg << "Loaded N = " << hx.size() << " particles" << endl;
        std::printf("Loaded %llu particles from %s\n",
                    static_cast<unsigned long long>(hx.size()), posPath.c_str());

        ippl::nbody::DisorderHeatingBHManager<double, 3>::Config cfg;
        if (pos.size() > 2) cfg.Nt = std::atoi(pos[2]);

        msg << "Nt = " << cfg.Nt << endl;

        static IpplTimings::TimerRef mainTimer = IpplTimings::getTimer("total");
        IpplTimings::startTimer(mainTimer);

        ippl::nbody::DisorderHeatingBHManager<double, 3> manager(
            std::move(hx), std::move(hy), std::move(hz), cfg);

        std::printf("Pre Run: computing initial forces...\n");
        manager.pre_run();

        std::printf("Starting iterations ...\n");
        std::fflush(stdout);
        manager.run(manager.getNt());
        std::printf("Simulation finished.\n");

        IpplTimings::stopTimer(mainTimer);
        IpplTimings::print();
        IpplTimings::print(std::string("timing.dat"));
    }
    ippl::finalize();
    return 0;
}
