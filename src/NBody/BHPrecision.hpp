#ifndef IPPL_NBODY_BH_PRECISION_HPP
#define IPPL_NBODY_BH_PRECISION_HPP

namespace ippl::nbody {

// Precision policy for the gridless Barnes-Hut stack. Each policy bundles the
// five scalar template parameters that ryoanji's traversal kernel exposes,
// plus the multipole value type. The three policies below correspond one-to-one
// to ryoanji's pre-compiled TRAVERSE_MPOLE instantiations in
// extern/ryoanji/src/ryoanji/nbody/traversal_gpu.cu — adding a fourth combo
// here without a matching TRAVERSE row in ryoanji will produce a link error.
//
// Type roles:
//   Tc  — coordinate type for positions (R), velocity (P), box, cstone domain
//   Th  — smoothing length (h) as consumed by ryoanji P2P
//   Tm  — mass / charge as consumed by ryoanji P2P
//   Ta  — accelerations (Ex/Ey/Ez) as written by ryoanji and consumed by leapfrog
//   Tf  — multipole expansion-center coordinate type (Vec4<Tf> source centers)
//   Tmm — multipole *value* type (template arg to CartesianQuadrupole<...>)

struct DoublePrecision {
    using Tc  = double;
    using Th  = double;
    using Tm  = double;
    using Ta  = double;
    using Tf  = double;
    using Tmm = double;
};

struct MixedPrecision {
    using Tc  = double;
    using Th  = float;
    using Tm  = float;
    using Ta  = float;
    using Tf  = double;
    using Tmm = float;
};

struct FloatPrecision {
    using Tc  = float;
    using Th  = float;
    using Tm  = float;
    using Ta  = float;
    using Tf  = float;
    using Tmm = float;
};

}  // namespace ippl::nbody

#endif  // IPPL_NBODY_BH_PRECISION_HPP
