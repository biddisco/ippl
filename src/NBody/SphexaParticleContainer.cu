#include "NBody/SphexaParticleContainer.hpp"

#include <cstdint>

#include "cstone/sfc/common.hpp"
#include "cstone/tree/csarray.hpp"

namespace ippl::nbody {
namespace detail {

namespace {

template <class KeyType, class Th>
__global__ void setHFromLeafKernel(const KeyType* keys,
                                   const KeyType* leaves,
                                   int            nLeafKeys,
                                   Th             cbrtVol,
                                   Th*            h,
                                   unsigned       n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    KeyType               k         = keys[i];
    cstone::TreeNodeIndex leafIdx   = cstone::findNodeBelow(leaves, nLeafKeys, k);
    KeyType               codeRange = leaves[leafIdx + 1] - leaves[leafIdx];
    unsigned              level     = cstone::treeLevel(codeRange);
    h[i] = cbrtVol / Th(1u << level);
}

}  // namespace

template <class Th, class KeyType>
void setHFromLeaves(const KeyType* keys, const KeyType* leaves, int nLeafKeys,
                    Th cbrtVol, Th* h, unsigned n) {
    if (n == 0) return;
    const unsigned blockSize = 256u;
    const unsigned numBlocks = (n + blockSize - 1u) / blockSize;
    setHFromLeafKernel<KeyType, Th>
        <<<numBlocks, blockSize>>>(keys, leaves, nLeafKeys, cbrtVol, h, n);
}

template void setHFromLeaves<double, std::uint64_t>(
    const std::uint64_t*, const std::uint64_t*, int, double, double*, unsigned);
template void setHFromLeaves<float, std::uint64_t>(
    const std::uint64_t*, const std::uint64_t*, int, float, float*, unsigned);

}  // namespace detail
}  // namespace ippl::nbody
