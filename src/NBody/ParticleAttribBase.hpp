#ifndef IPPL_NBODY_PARTICLE_ATTRIB_BASE_HPP
#define IPPL_NBODY_PARTICLE_ATTRIB_BASE_HPP

#include <cstddef>

namespace ippl::nbody {

class ParticleAttribBase {
public:
    virtual ~ParticleAttribBase() = default;

    virtual void        resize(std::size_t n) = 0;
    virtual std::size_t size() const          = 0;
};

} // namespace ippl::nbody

#endif // IPPL_NBODY_PARTICLE_ATTRIB_BASE_HPP
