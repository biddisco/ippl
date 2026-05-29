#include "NBody/SphexaParticleContainer.hpp"

#include <cstdint>

#include "cstone/sfc/common.hpp"
#include "cstone/tree/csarray.hpp"

namespace ippl::nbody {
namespace detail {

template <class Th, class KeyType>
void setHFromLeaves(const KeyType* keys, const KeyType* leaves, int nLeafKeys,
                    Th cbrtVol, Th* h, unsigned n) {
#pragma omp parallel for schedule(static)
    for (unsigned i = 0; i < n; ++i) {
        KeyType               k         = keys[i];
        cstone::TreeNodeIndex leafIdx   = cstone::findNodeBelow(leaves, nLeafKeys, k);
        KeyType               codeRange = leaves[leafIdx + 1] - leaves[leafIdx];
        unsigned              level     = cstone::treeLevel(codeRange);
        h[i] = cbrtVol / Th(1u << level);
    }
}

template void setHFromLeaves<double, std::uint64_t>(
    const std::uint64_t*, const std::uint64_t*, int, double, double*, unsigned);
template void setHFromLeaves<float, std::uint64_t>(
    const std::uint64_t*, const std::uint64_t*, int, float, float*, unsigned);

}  // namespace detail
}  // namespace ippl::nbody
