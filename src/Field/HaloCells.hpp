//
// Class HaloCells
//   The guard / ghost cells of BareField.
//

#include <memory>
#include <vector>

#include "Utility/IpplException.h"

#include "Communicate/Communicator.h"

namespace ippl {
    namespace detail {
        template <typename T, unsigned Dim, class... ViewArgs>
        HaloCells<T, Dim, ViewArgs...>::HaloCells() {}

        template <typename T, unsigned Dim, class... ViewArgs>
        void HaloCells<T, Dim, ViewArgs...>::accumulateHalo(view_type& view, Layout_t* layout) {
            exchangeBoundaries<lhs_plus_assign>(view, layout, HALO_TO_INTERNAL);
        }

        template <typename T, unsigned Dim, class... ViewArgs>
        void HaloCells<T, Dim, ViewArgs...>::accumulateHalo_noghost(view_type& view,
                                                                    Layout_t* layout, int nghost) {
            exchangeBoundaries<lhs_plus_assign>(view, layout, HALO_TO_INTERNAL_NOGHOST, nghost);
        }
        template <typename T, unsigned Dim, class... ViewArgs>
        void HaloCells<T, Dim, ViewArgs...>::fillHalo(view_type& view, Layout_t* layout) {
            exchangeBoundaries<assign>(view, layout, INTERNAL_TO_HALO);
        }

        template <typename T, unsigned Dim, class... ViewArgs>
        template <class Op>
        void HaloCells<T, Dim, ViewArgs...>::exchangeBoundaries(view_type& view, Layout_t* layout,
                                                                SendOrder order, int nghost) {
            using neighbor_list = typename Layout_t::neighbor_list;
            using range_list    = typename Layout_t::neighbor_range_list;

            auto& comm = layout->comm;

            const neighbor_list& neighbors = layout->getNeighbors();
            const range_list &sendRanges   = layout->getNeighborsSendRange(),
                             &recvRanges   = layout->getNeighborsRecvRange();

            auto ldom = layout->getLocalNDIndex();
            for (const auto& axis : ldom) {
                if ((axis.length() == 1) && (Dim != 1)) {
                    throw std::runtime_error(
                        "HaloCells: Cannot do neighbour exchange when domain decomposition "
                        "contains planes!");
                }
            }

            // needed for the NOGHOST approach - we want to remove the ghost
            // cells on the boundaries of the global domain from the halo
            // exchange when we set HALO_TO_INTERNAL_NOGHOST
            const auto domain    = layout->getDomain();
            const auto& ldomains = layout->getHostLocalDomains();

            size_t sendRequests = 0;
            for (const auto& componentNeighbors : neighbors) {
                sendRequests += componentNeighbors.size();
            }

            int me = Comm->rank();

            using memory_space = typename view_type::memory_space;
            using buffer_type  = mpi::Communicator::buffer_type<memory_space>;
            std::vector<MPI_Request> send_requests(sendRequests);
            std::vector<MPI_Request> recv_requests;

            constexpr size_t cubeCount = detail::countHypercubes(Dim) - 1;

            // ------------------------------------
            // pre-post receives loop
            // ------------------------------------
            struct bc_irecv_data {
                buffer_type pre_posted_recv_buf;
                bound_type range;
            };
            std::vector<bc_irecv_data> pre_posted_bufs;
            //
            for (size_t index = 0; index < cubeCount; index++) {
                int tag                        = mpi::tag::HALO + Layout_t::getMatchingIndex(index);
                const auto& componentNeighbors = neighbors[index];
                for (size_t i = 0; i < componentNeighbors.size(); i++) {
                    int sourceRank = componentNeighbors[i];

                    bound_type range;
                    if (order == INTERNAL_TO_HALO) {
                        range = recvRanges[index][i];
                    } else if (order == HALO_TO_INTERNAL_NOGHOST) {
                        range = sendRanges[index][i];

                        for (size_t j = 0; j < Dim; ++j) {
                            bool isLower = ((range.lo[j] + ldomains[me][j].first() - nghost)
                                            == domain[j].min());
                            bool isUpper = ((range.hi[j] - 1 + ldomains[me][j].first() - nghost)
                                            == domain[j].max());
                            range.lo[j] += isLower * (nghost);
                            range.hi[j] -= isUpper * (nghost);
                        }
                    } else {
                        range = sendRanges[index][i];
                    }

                    size_type nrecvs = range.size();

                    // std::cout << haloData_m.get_buffer();
                    buffer_type buf = comm.template getBuffer<memory_space, T>(nrecvs);
                    MPI_Request request;
                    // comm.recv(sourceRank, tag, haloData_m, *buf, nrecvs * sizeof(T), nrecvs);
                    comm.irecv(sourceRank, tag, *buf, request, nrecvs * sizeof(T));
                    pre_posted_bufs.push_back({buf, range});
                    recv_requests.push_back(request);
                }
            }

            // sending loop
            size_t requestIndex = 0;
            for (size_t index = 0; index < cubeCount; index++) {
                int tag                        = mpi::tag::HALO + index;
                const auto& componentNeighbors = neighbors[index];
                for (size_t i = 0; i < componentNeighbors.size(); i++) {
                    int targetRank = componentNeighbors[i];

                    bound_type range;
                    if (order == INTERNAL_TO_HALO) {
                        /*We store only the sending and receiving ranges
                         * of INTERNAL_TO_HALO and use the fact that the
                         * sending range of HALO_TO_INTERNAL is the receiving
                         * range of INTERNAL_TO_HALO and vice versa
                         */
                        range = sendRanges[index][i];
                    } else if (order == HALO_TO_INTERNAL_NOGHOST) {
                        range = recvRanges[index][i];

                        for (size_t j = 0; j < Dim; ++j) {
                            bool isLower = ((range.lo[j] + ldomains[me][j].first() - nghost)
                                            == domain[j].min());
                            bool isUpper = ((range.hi[j] - 1 + ldomains[me][j].first() - nghost)
                                            == domain[j].max());
                            range.lo[j] += isLower * (nghost);
                            range.hi[j] -= isUpper * (nghost);
                        }
                    } else {
                        range = recvRanges[index][i];
                    }

                    size_type nsends;
                    pack(range, view, haloData_m, nsends);

                    buffer_type buf = comm.template getBuffer<memory_space, T>(nsends);

                    comm.isend(targetRank, tag, haloData_m, *buf, send_requests[requestIndex++],
                               nsends);
                    buf->resetWritePos();
                }
            }

            if (sendRequests > 0) {
                // std::cout << "WaitAll for " << sendRequests << " " << send_requests.size()
                //           << " buffers" << std::endl;
                MPI_Waitall(sendRequests, send_requests.data(), MPI_STATUSES_IGNORE);
            }

            // ------------------------------------
            // deserialize and unpack pre-posted receives
            // ------------------------------------
            if (pre_posted_bufs.size() > 0) {
                // std::cout << "WaitAll for " << pre_posted_bufs.size() << " " <<
                // recv_requests.size()
                //           << " buffers" << std::endl;
                MPI_Waitall(recv_requests.size(), recv_requests.data(), MPI_STATUSES_IGNORE);
                for (auto& bc_data : pre_posted_bufs) {
                    haloData_m.deserialize(*bc_data.pre_posted_recv_buf, bc_data.range.size());
                    unpack<Op>(bc_data.range, view, haloData_m);
                    auto& my_view = bc_data.pre_posted_recv_buf->buffer_m;
                    std::cout << "extent " << my_view.extent(0) << std::endl;
                    // Use parallel_for to set all elements to 0
                    Kokkos::parallel_for(
                        "buffer clear", my_view.extent(0),
                        KOKKOS_CLASS_LAMBDA(int i) { my_view(i) = 0; });
                    Kokkos::fence();

                    comm.template freeBuffer(bc_data.pre_posted_recv_buf);
                }
            }

            // comm.freeAllBuffers();
        }

        template <typename T, unsigned Dim, class... ViewArgs>
        void HaloCells<T, Dim, ViewArgs...>::pack(const bound_type& range, const view_type& view,
                                                  databuffer_type& fd, size_type& nsends) {
            auto subview = makeSubview(view, range);

            auto& buffer = fd.buffer;

            size_t size = subview.size();
            nsends      = size;
            if (buffer.size() < size) {
                int overalloc = Comm->getDefaultOverallocation();
                Kokkos::realloc(buffer, size * overalloc);
            }

            using index_array_type =
                typename RangePolicy<Dim, typename view_type::execution_space>::index_array_type;
            ippl::parallel_for(
                "HaloCells::pack()", getRangePolicy(subview),
                KOKKOS_LAMBDA(const index_array_type& args) {
                    int l = 0;

                    for (unsigned d1 = 0; d1 < Dim; d1++) {
                        int next = args[d1];
                        for (unsigned d2 = 0; d2 < d1; d2++) {
                            next *= subview.extent(d2);
                        }
                        l += next;
                    }

                    buffer(l) = apply(subview, args);
                });
            Kokkos::fence();
        }

        template <typename T, unsigned Dim, class... ViewArgs>
        template <typename Op>
        void HaloCells<T, Dim, ViewArgs...>::unpack(const bound_type& range, const view_type& view,
                                                    databuffer_type& fd) {
            auto subview = makeSubview(view, range);
            auto buffer  = fd.buffer;

            // 29. November 2020
            // https://stackoverflow.com/questions/3735398/operator-as-template-parameter
            Op op;

            using index_array_type =
                typename RangePolicy<Dim, typename view_type::execution_space>::index_array_type;
            ippl::parallel_for(
                "HaloCells::unpack()", getRangePolicy(subview),
                KOKKOS_LAMBDA(const index_array_type& args) {
                    int l = 0;

                    for (unsigned d1 = 0; d1 < Dim; d1++) {
                        int next = args[d1];
                        for (unsigned d2 = 0; d2 < d1; d2++) {
                            next *= subview.extent(d2);
                        }
                        l += next;
                    }

                    op(apply(subview, args), buffer(l));
                });
            Kokkos::fence();
        }

        template <typename T, unsigned Dim, class... ViewArgs>
        auto HaloCells<T, Dim, ViewArgs...>::makeSubview(const view_type& view,
                                                         const bound_type& intersect) {
            auto makeSub = [&]<size_t... Idx>(const std::index_sequence<Idx...>&) {
                return Kokkos::subview(view,
                                       Kokkos::make_pair(intersect.lo[Idx], intersect.hi[Idx])...);
            };
            return makeSub(std::make_index_sequence<Dim>{});
        }

        template <typename T, unsigned Dim, class... ViewArgs>
        template <typename Op>
        void HaloCells<T, Dim, ViewArgs...>::applyPeriodicSerialDim(view_type& view,
                                                                    const Layout_t* layout,
                                                                    const int nghost) {
            int myRank           = layout->comm.rank();
            const auto& lDomains = layout->getHostLocalDomains();
            const auto& domain   = layout->getDomain();

            using exec_space = typename view_type::execution_space;
            using index_type = typename RangePolicy<Dim, exec_space>::index_type;

            Kokkos::Array<index_type, Dim> ext, begin, end;

            for (size_t i = 0; i < Dim; ++i) {
                ext[i]   = view.extent(i);
                begin[i] = 0;
            }

            Op op;

            for (unsigned d = 0; d < Dim; ++d) {
                end    = ext;
                end[d] = nghost;

                if (lDomains[myRank][d].length() == domain[d].length()) {
                    int N = view.extent(d) - 1;

                    using index_array_type =
                        typename RangePolicy<Dim,
                                             typename view_type::execution_space>::index_array_type;
                    ippl::parallel_for(
                        "applyPeriodicSerialDim", createRangePolicy<Dim, exec_space>(begin, end),
                        KOKKOS_LAMBDA(index_array_type & coords) {
                            // The ghosts are filled starting from the inside
                            // of the domain proceeding outwards for both lower
                            // and upper faces. The extra brackets and explicit
                            // mention

                            // nghost + i
                            coords[d] += nghost;
                            auto&& left = apply(view, coords);

                            // N - nghost - i
                            coords[d]    = N - coords[d];
                            auto&& right = apply(view, coords);

                            // nghost - 1 - i
                            coords[d] += 2 * nghost - 1 - N;
                            op(apply(view, coords), right);

                            // N - (nghost - 1 - i) = N - (nghost - 1) + i
                            coords[d] = N - coords[d];
                            op(apply(view, coords), left);
                        });

                    Kokkos::fence();
                }
            }
        }
    }  // namespace detail
}  // namespace ippl
