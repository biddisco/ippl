#ifndef IPPL_NBODY_MANAGER_HPP
#define IPPL_NBODY_MANAGER_HPP

#include <memory>

#include "Ippl.h"
#include "Manager/BaseManager.h"
#include "Utility/IpplTimings.h"

#include "NBody/BHPrecision.hpp"
#include "NBody/LeapfrogStepper.hpp"
#include "NBody/PeriodicWrap.hpp"
#include "NBody/SphexaBHSolver.hpp"
#include "NBody/SphexaParticleContainer.hpp"

namespace ippl::nbody {

// RAII wrapper around IpplTimings::startTimer / stopTimer so per-function
// instrumentation in the manager pipeline is exception-safe and clutter-free.
// Lifetime: starts on construction, stops on destruction. Use with a
// static-local TimerRef so the timer is registered once per call site:
//
//     static IpplTimings::TimerRef t = IpplTimings::getTimer("kickHalf");
//     ScopedIpplTimer scope(t);
class ScopedIpplTimer {
public:
    explicit ScopedIpplTimer(IpplTimings::TimerRef ref) : ref_(ref) {
        IpplTimings::startTimer(ref_);
    }
    ~ScopedIpplTimer() { IpplTimings::stopTimer(ref_); }
    ScopedIpplTimer(const ScopedIpplTimer&)            = delete;
    ScopedIpplTimer& operator=(const ScopedIpplTimer&) = delete;

private:
    IpplTimings::TimerRef ref_;
};

// Base class for gridless Barnes-Hut simulations driven by IPPL's BaseManager
// loop (pre_step / advance / post_step). Mirrors the role AlpineManager plays
// for the FFT-PIC stack: owns the particle container and the BH solver, exposes
// the physics-agnostic helpers (kickHalf, drift, wrap, solve), and delegates
// the simulation-specific pieces (IC sampling, solver parameters, per-step
// ordering, diagnostics) to derived managers via pure-virtual hooks.
//
// Templated on a precision policy P (NBody/BHPrecision.hpp). All inner types
// (container, solver, leapfrog, diagnostics) take the same P so a single
// template parameter selects the precision config end-to-end.
//
// Timing: every physics-active helper is wrapped with a ScopedIpplTimer so a
// per-function breakdown lands in IpplTimings::print() at the end of the run.
// The driver still controls the outer "total" timer.
//
// Lifetime / ownership:
//   - The base owns the container and the solver as unique_ptrs. Both are
//     constructed lazily inside pre_run(), because derived classes need to
//     supply construction parameters (box, BCs, theta, G, ...) that depend
//     on simulation-specific state.
//   - run(int nt) is inherited verbatim from ippl::BaseManager.
//
// Required overrides in derived classes:
//   initializeContainer()   - construct pcontainer_m with the right box/BCs
//   initializeParticles()   - fill positions, velocities, charges, h
//   advanceImpl()           - one timestep (KDK, drift-then-kick, etc.)
//
// Optional overrides:
//   initializeSolverParams(Params&)  - tweak G, theta, Ewald shells
//   post_initial_solve()             - hook between t=0 BH solve and t=0 dump
//                                       (e.g. DIH calibrates focusing here)
//   dumpImpl()                       - write CSV row / stdout report
//
// Note the `Impl` suffix on advance / dump: the base class wraps both with a
// ScopedIpplTimer so derived classes don't need to remember to instrument.
template <class P, unsigned Dim>
class NBodyManager : public ippl::BaseManager {
    static_assert(Dim == 3, "NBodyManager requires Dim == 3");

public:
    using Precision    = P;
    using Tc           = typename P::Tc;
    using Container    = SphexaParticleContainer<P, Dim>;
    using Solver       = SphexaBHSolver<P, Dim>;
    using SolverParams = typename Solver::Params;

    NBodyManager(unsigned long N, int Nt, Tc dt)
        : ippl::BaseManager()
        , N_m(N)
        , Nt_m(Nt)
        , dt_m(dt) {}

    ~NBodyManager() override = default;

    NBodyManager(const NBodyManager&)            = delete;
    NBodyManager& operator=(const NBodyManager&) = delete;

    // BaseManager interface --------------------------------------------------

    // Standard sequence: build container → allocate → fill IC → build solver
    // → first BH evaluation → optional post-solve hook → dump t=0 row. Derived
    // classes that need a different sequence can override pre_run() outright.
    void pre_run() override {
        static IpplTimings::TimerRef tInit      = IpplTimings::getTimer("init");
        static IpplTimings::TimerRef tFirstSolv = IpplTimings::getTimer("firstSolve");

        {
            ScopedIpplTimer scope(tInit);
            initializeContainer();
            // Allocate per-rank storage. Derived managers fill only the local
            // slice [0, localN()); after the first updateGrav() inside
            // runSolver(), cstone reshards globally and [startIndex(),
            // endIndex()) becomes this rank's owned range.
            pcontainer_m->create(static_cast<unsigned>(localN()));
            initializeParticles();

            SolverParams bhParams;
            initializeSolverParams(bhParams);
            solver_m = std::make_unique<Solver>(*pcontainer_m, bhParams);
        }

        {
            ScopedIpplTimer scope(tFirstSolv);
            // warmup=true: the t=0 solve is timed only by the outer
            // "firstSolve" timer; the per-scope bh.* accumulators and the
            // per-step bh.stats printout are reserved for the advance() loop
            // so they reflect a pure per-step number directly comparable to
            // sphexa's "Gravity" timer.
            solver_m->runSolver(/*warmup=*/true);
        }
        post_initial_solve();

        time_m = Tc(0);
        it_m   = 0;
        dump();
    }

    // BaseManager::advance() is pure-virtual. We seal it here to wrap the
    // derived implementation in a "step" timer; derived classes override
    // advanceImpl() instead.
    void advance() final {
        static IpplTimings::TimerRef t = IpplTimings::getTimer("step");
        ScopedIpplTimer scope(t);
        advanceImpl();
    }

    // Default post_step: advance time and emit a CSV row. Derived classes
    // override dumpImpl() rather than post_step() unless they need a different
    // ordering of time/it/dump.
    void post_step() override {
        time_m += dt_m;
        ++it_m;
        dump();
    }

    int getNt() const { return Nt_m; }

protected:
    // Required hooks -------------------------------------------------------
    virtual void initializeContainer() = 0;
    virtual void initializeParticles() = 0;
    virtual void advanceImpl()         = 0;

    // Optional hooks -------------------------------------------------------
    virtual void initializeSolverParams(SolverParams& /*p*/) {}
    virtual void post_initial_solve() {}
    virtual void dumpImpl() {}

    // Sealed dump() wraps dumpImpl() in a "dump" timer. Derived classes
    // implement dumpImpl() and don't need to instrument it themselves.
    void dump() {
        static IpplTimings::TimerRef t = IpplTimings::getTimer("dump");
        ScopedIpplTimer scope(t);
        dumpImpl();
    }

    // Common helpers — drive the loop pieces from advance() in derived classes.
    void kickHalf() {
        static IpplTimings::TimerRef t = IpplTimings::getTimer("kickHalf");
        ScopedIpplTimer scope(t);
        leapfrogKickHalf<P>(*pcontainer_m, dt_m);
    }

    void kickFull() {
        static IpplTimings::TimerRef t = IpplTimings::getTimer("kickFull");
        ScopedIpplTimer scope(t);
        leapfrogKick<P>(*pcontainer_m, dt_m);
    }

    void drift() {
        static IpplTimings::TimerRef t = IpplTimings::getTimer("drift");
        ScopedIpplTimer scope(t);
        leapfrogDrift<P>(*pcontainer_m, dt_m);
    }

    void wrap() {
        static IpplTimings::TimerRef t = IpplTimings::getTimer("wrap");
        ScopedIpplTimer scope(t);
        wrapToBox<P>(*pcontainer_m);
    }

    void solve() {
        static IpplTimings::TimerRef t = IpplTimings::getTimer("solve");
        ScopedIpplTimer scope(t);
        solver_m->runSolver();
    }

    // Accessors ------------------------------------------------------------
    Container&       pc()             { return *pcontainer_m; }
    const Container& pc() const       { return *pcontainer_m; }
    Solver&          solver()         { return *solver_m; }
    const Solver&    solver() const   { return *solver_m; }

    Tc               time() const     { return time_m; }
    int              it()   const     { return it_m;   }
    Tc               dt()   const     { return dt_m;   }
    unsigned long    N()    const     { return N_m;    }

    // MPI-aware accessors. rank/size come from ippl::Comm so derived managers
    // don't need to care whether we're running 1 rank or 1024.
    int              rank()    const  { return ippl::Comm->rank(); }
    int              numRanks() const { return ippl::Comm->size(); }

    // Per-rank slice [firstGlobal, lastGlobal) of the global N. Even split with
    // remainder on the trailing ranks: rank r owns [r*N/R, (r+1)*N/R). Single
    // rank: [0, N) so existing single-rank behavior is unchanged.
    unsigned long firstGlobal() const {
        return (N_m * static_cast<unsigned long>(rank())) /
               static_cast<unsigned long>(numRanks());
    }
    unsigned long lastGlobal() const {
        return (N_m * static_cast<unsigned long>(rank() + 1)) /
               static_cast<unsigned long>(numRanks());
    }
    unsigned long localN() const { return lastGlobal() - firstGlobal(); }

    // Derived classes assign the container during initializeContainer().
    void setContainer(std::unique_ptr<Container> pc) {
        pcontainer_m = std::move(pc);
    }

protected:
    std::unique_ptr<Container> pcontainer_m;
    std::unique_ptr<Solver>    solver_m;

    unsigned long N_m;
    int           Nt_m;
    Tc            dt_m;
    Tc            time_m{Tc(0)};
    int           it_m{0};
};

}  // namespace ippl::nbody

#endif  // IPPL_NBODY_MANAGER_HPP
