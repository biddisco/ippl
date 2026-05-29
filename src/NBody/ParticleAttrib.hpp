#ifndef IPPL_NBODY_PARTICLE_ATTRIB_HPP
#define IPPL_NBODY_PARTICLE_ATTRIB_HPP

#include "NBody/Accelerator.hpp"
#include "NBody/ParticleAttribBase.hpp"

namespace ippl::nbody {

template <class T>
class ParticleAttrib : public ParticleAttribBase {
public:
    using value_type = T;

    void        resize(std::size_t n) override { data_.resize(n); }
    std::size_t size() const override          { return data_.size(); }

    T*       data()       { return data_.data(); }
    const T* data() const { return data_.data(); }

    FieldVector<T>&       container()       { return data_; }
    const FieldVector<T>& container() const { return data_; }

private:
    FieldVector<T> data_;
};

} // namespace ippl::nbody

#endif // IPPL_NBODY_PARTICLE_ATTRIB_HPP
