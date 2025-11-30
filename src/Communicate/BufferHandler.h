#ifndef IPPL_BUFFER_HANDLER_H
#define IPPL_BUFFER_HANDLER_H

#include <memory>
#include <set>

#include "Types/IpplTypes.h"
#include "Types/ViewTypes.h"

#include "Utility/TypeUtils.h"
#include "Utility/logging.h"

#include "Communicate/Archive.h"
#include "Communicate/pool.h"

// for boundary alignment requirement
// inline std::uintptr_t align_offset(uint bits, std::uintptr_t address) {
//     auto GPU_BOUND_SIZE   = ((std::uint64_t)1 << bits);
//     auto GPU_BOUND_OFFSET = (GPU_BOUND_SIZE - 1);
//     return GPU_BOUND_OFFSET;
// }

// inline std::uintptr_t align_to_next(uint bits, std::uintptr_t address) {
//     auto offset = align_offset(bits, address);
//     return (address + offset) & (~offset);
// }

namespace ippl {

    template <typename MemorySpace>
    struct Kokkos_Provider {
        using size_type = ippl::detail::size_type;
        //
        using provider_domain = size_type;
        using region_type =
            typename detail::ViewType<char, 1, MemorySpace,
                                      Kokkos::MemoryTraits<Kokkos::Aligned>>::view_type;

        using buffer_type =
            typename detail::ViewType<char, 1, MemorySpace,
                                      Kokkos::MemoryTraits<Kokkos::Aligned>>::view_type;
        using pointer_type = typename buffer_type::pointer_type;
        using memory_space = MemorySpace;
    };

    // simple alias that maps our kokkos based region into a simpler name
    template <typename MemorySpace>
    using rma_buffer = communication_pool::rma_memory_region<Kokkos_Provider<MemorySpace>>;

    // archive wrapper around an arbitrary buffer
    template <typename BufferType>
    struct rma_archive {
        using type = detail::Archive<BufferType>;
    };

    // shared pointer wrapper around an archive wrapper
    template <typename BufferType>
    struct comm_buff_type {
        using type = std::shared_ptr<typename rma_archive<BufferType>::type>;
    };

    // buffer per MemorySpace
    template <typename MemorySpace>
    struct mpi_comm_buff {
        using type = rma_buffer<MemorySpace>;
    };

    // vector per MemorySpace
    template <template <typename> class BufferType, typename MemorySpace>
    struct comm_buff_vector_for_space {
        using type = std::vector<typename comm_buff_type<BufferType<MemorySpace>>::type>;
    };

    // // binder to convert 2-param template to 1-param
    // template <template <typename> class BufferType>
    // struct comm_buff_vector_binder {
    //     template <typename MemorySpace>
    //     struct apply {
    //         using type = typename comm_buff_vector_for_space<BufferType, MemorySpace>::type;
    //     };
    // };

    template <template <typename> class BufferType>
    struct comm_buff_vector_binder {
        template <typename MemorySpace>
        using apply = typename comm_buff_vector_for_space<BufferType, MemorySpace>::type;
    };

    // // ContainerForAllSpaces fake
    // template <template <typename> class PerSpaceTemplate>
    // struct ContainerForAllSpaces {
    //     using type = int;  // just for demonstration
    // };

    // alias template for all spaces
    template <template <typename> class BufferType>
    using mpi_comm_buffer_container_for_all_spaces = typename detail::ContainerForAllSpaces<
        comm_buff_vector_binder<BufferType>::template apply>::type;

    using mpi_buffer_container = mpi_comm_buffer_container_for_all_spaces<rma_buffer>;

    /**raw_buffer
     * @brief Interface for memory buffer handling.
     *
     * Defines methods for acquiring, freeing, and managing memory buffers.
     * Implementations are responsible for managing buffers efficiently,
     * ensuring that allocated buffers are reused where possible.
     *
     * @tparam MemorySpace The memory space type used for buffer allocation.
     */
    template <typename Buffer, typename MemorySpace>
    class BufferHandler {
    public:
        using buffer_type = std::shared_ptr<Buffer>;
        using size_type   = ippl::detail::size_type;

        virtual ~BufferHandler() {}

        /**
         * @brief Requests a memory buffer of a specified size.
         *
         * Provides a buffer of at least the specified size, with the option
         * to allocate additional space based on an overallocation multiplier.
         * This function attempts to reuse available buffers if possible.
         *
         * @param size The required size of the buffer, in bytes.
         * @param overallocation A multiplier to allocate extra space, which may help
         *                       avoid frequent reallocation in some use cases.
         * @return A shared pointer to the allocated buffer.
         */
        virtual buffer_type getBuffer(size_type size, double overallocation) = 0;

        /**
         * @brief Frees a specified buffer.
         *
         * Moves the specified buffer to a free state, making it available
         * for reuse in future buffer requests.
         *
         * @param buffer The buffer to be freed.
         */
        virtual void freeBuffer(buffer_type buffer) = 0;

        /**
         * @brief Frees all currently used buffers.
         *
         * Transfers all used buffers to the free state, making them available
         * for reuse. This does not deallocate memory but resets buffer usage.
         */
        virtual void freeAllBuffers() = 0;

        /**
         * @brief Deletes all buffers.
         *
         * Releases all allocated memory buffers, both used and free.
         * After this call, no buffers are available until new allocations.
         */
        virtual void deleteAllBuffers() = 0;

        /**
         * @brief Gets the size of all buffers in use.
         *
         * @return Total size of buffers that are in use in bytes.
         */
        virtual size_type getUsedSize() const = 0;

        /**
         * @brief Gets the size of all free buffers.
         *
         * @return Total size of free buffers in bytes.
         */
        virtual size_type getFreeSize() const = 0;

        virtual int getUsedN() const = 0;
        virtual int getFreeN() const = 0;
    };

    /**
     * @class DefaultBufferHandler
     * @brief Concrete implementation of BufferHandler for managing memory buffers.
     *
     * This class implements the BufferHandler interface, providing concrete behavior for
     * buffer allocation, freeing, and memory management. It maintains two sorted sets of free and
     * in-use buffers to allow for efficient queries.
     *
     * @tparam MemorySpace The memory space type for the buffer (e.g., `Kokkos::HostSpace`).
     */
    // template <typename MemorySpace>
    // class DefaultBufferHandler : public BufferHandler<MemorySpace> {
    // public:
    //     using typename BufferHandler<MemorySpace>::archive_type;
    //     using typename BufferHandler<MemorySpace>::buffer_type;
    //     using typename BufferHandler<MemorySpace>::size_type;

    //     ~DefaultBufferHandler() override;

    //     /**
    //      * @brief Acquires a buffer of at least the specified size.
    //      *
    //      * Requests a memory buffer of the specified size, with the option
    //      * to request a buffer larger than the base size by an overallocation
    //      * multiplier. If a sufficiently large buffer is available, it is returned. If not, the
    //      * largest free buffer is reallocated. If there are no free buffers available, only then
    //      a
    //      * new buffer is allocated.
    //      *
    //      * @param size The required buffer size.
    //      * @param overallocation A multiplier to allocate additional buffer space.
    //      * @return A shared pointer to the allocated buffer.
    //      */
    //     buffer_type getBuffer(size_type size, double overallocation) override;

    //     /**
    //      * @copydoc BufferHandler::freeBuffer
    //      */
    //     void freeBuffer(buffer_type buffer) override;

    //     /**
    //      * @copydoc BufferHandler::freeBuffer
    //      */
    //     void freeAllBuffers() override;

    //     /**
    //      * @copydoc BufferHandler::freeBuffer
    //      */
    //     void deleteAllBuffers() override;

    //     /**
    //      * @copydoc BufferHandler::freeBuffer
    //      */
    //     size_type getUsedSize() const override;

    //     /**
    //      * @copydoc BufferHandler::freeBuffer
    //      */
    //     size_type getFreeSize() const override;

    //     int getUsedN() const override;
    //     int getFreeN() const override;

    // private:
    //     using buffer_comparator_type = bool (*)(const buffer_type&, const buffer_type&);
    //     using buffer_set_type        = std::set<buffer_type, buffer_comparator_type>;

    //     static bool bufferSizeCobufferSizeComparatormparator(const buffer_type& lhs,
    //                                                          const buffer_type& rhs);

    //     bool isBufferUsed(buffer_type buffer) const;
    //     void releaseUsedBuffer(buffer_type buffer);
    //     buffer_type findFreeBuffer(size_type requiredSize);
    //     typename buffer_set_type::iterator findSmallestSufficientBuffer(size_type requiredSize);
    //     buffer_type getFreeBuffer(buffer_type buffer);
    //     buffer_type reallocateLargestFreeBuffer(size_type requiredSize);
    //     buffer_type allocateNewBuffer(size_type requiredSize);

    //     size_type usedSize_m{0};  ///< Total size of all allocated buffers
    //     size_type freeSize_m{0};  ///< Total size of all free buffers

    // protected:
    //     buffer_set_type used_buffers{
    //         &DefaultBufferHandler::bufferSizeComparator};  ///< Set of used buffers
    //     buffer_set_type free_buffers{
    //         &DefaultBufferHandler::bufferSizeComparator};  ///< Set of free buffers
    // };

    template <typename MemorySpace>
    struct communication_pool::rma_memory_region<Kokkos_Provider<MemorySpace>> {
        using provider        = Kokkos_Provider<MemorySpace>;
        using size_type       = provider::size_type;
        using provider_domain = provider::provider_domain;
        using provider_region = provider::region_type;
        using pointer_type    = Kokkos_Provider<MemorySpace>::pointer_type;
        using memory_space    = Kokkos_Provider<MemorySpace>::memory_space;
        using execution_space = Kokkos_Provider<MemorySpace>::memory_space;

        // --------------------------------------------------------------------
        // flags used for management of lifetime
        enum {
            BLOCK_USER    = 1,
            BLOCK_TEMP    = 2,
            BLOCK_PARTIAL = 4,
        };

        // typedef rma_memory_region<RegionProvider> region_type;
        // typedef memory_region_allocator<RegionProvider> allocator_type;
        // typedef std::shared_ptr<region_type> region_ptr;

        // --------------------------------------------------------------------
        // empty default constructor
        rma_memory_region()
            : region_(nullptr)
            , address_(nullptr)
            , base_addr_(nullptr)
            , size_(0)
            , used_space_(0)
            , flags_(0) {
            // std::cout << "rma_memory_region construct" << std::endl;
        }

        // --------------------------------------------------------------------
        rma_memory_region(provider_region region, char* address, char* base_address, uint64_t size,
                          uint32_t flags)
            : address_(address)
            , base_addr_(base_address)
            , size_(size)
            , used_space_(0)
            , flags_(flags)  //
        {
            // auto offset            = align_offset(12, (uintptr_t)(address));
            // std::uintptr_t aligned = align_to_next(12, (std::uintptr_t)(void*)(address));
            std::uintptr_t d = address - region.data();
            region_          = Kokkos::subview(
                region, std::make_pair<uint64_t, uint64_t>(uint64_t(d), uint64_t(d + size)));

            SPDLOG_DEBUG("{} Sub-region, size {}, base_addr {}, addr {}, region_addr {}, extent {}",
                         grox::debug::print_type<Kokkos_Provider<MemorySpace>>(), size,
                         (void*)(base_address), (void*)(address), (void*)(region_.data()),
                         region_.extent(0));

            assert(region_.data() == address);
        }

        // domain is not used for kokkos yet
        void allocate(provider_domain* /*domain*/, size_type bytes);

        inline KOKKOS_FUNCTION char* get_base_address() const { return base_addr_; }
        inline KOKKOS_FUNCTION char* get_address() const { return region_.data(); }
        inline KOKKOS_FUNCTION char* data() const { return region_.data(); }
        inline KOKKOS_FUNCTION provider_region get_region() { return region_; }
        inline KOKKOS_FUNCTION std::size_t get_size() const { return size_; }
        inline KOKKOS_FUNCTION std::size_t size() const { return size_; }

        // The internal network type dependent memory region handle
        provider_region region_;

        // we may be a piece of a larger region, this gives the start address
        // of this piece of the region. This is the address that should be used for data
        // storage
        char* address_;

        // if we are part of a larger region, this is the base address of
        // that larger region
        char* base_addr_;

        // The size of the memory buffer, if this is a partial region
        // it will be smaller than the value returned by region_->length
        uint64_t size_;

        // space used by a message in the memory region.
        uint64_t used_space_;

        // flags to control lifetime of blocks
        uint32_t flags_;

        friend std::ostream& operator<<(std::ostream& os, rma_memory_region const& r) {
            os << "base " << hexpointer(r.base_addr_) << " "       //
               << "addr " << hexpointer(r.address_) << " "         //
               << "size " << decnumber(r.size_) << " "             //
               << "kaddr " << hexpointer(r.region_.data()) << " "  //
               << "ksize " << decnumber(r.region_.size()) << " "   //
                ;
            return os;
        }
    };

    template <>
    inline void communication_pool::rma_memory_region<Kokkos_Provider<Kokkos::HostSpace>>::allocate(
        provider_domain* /*domain*/, size_type bytes)  //
    {
        char* device_ptr = nullptr;
        device_ptr       = (char*)aligned_alloc(4096, bytes);
        region_          = provider_region(device_ptr, size_);
        address_         = (char*)(device_ptr);
        base_addr_       = (char*)(device_ptr);
        size_            = bytes;
        flags_           = 0;
        SPDLOG_DEBUG("{} Aligned: bytes {}, base address {}, original {}, extent {}",
                     grox::debug::print_type<Kokkos_Provider<Kokkos::HostSpace>>(), bytes,
                     (void*)base_addr_, (void*)(region_.data()), region_.extent(0));
    }

    template <>
    inline void communication_pool::rma_memory_region<Kokkos_Provider<Kokkos::CudaSpace>>::allocate(
        provider_domain* /*domain*/, size_type bytes)  //
    {
        char* device_ptr = nullptr;
        cudaMalloc(&device_ptr, bytes);
        region_    = provider_region(device_ptr, size_);
        address_   = (char*)(device_ptr);
        base_addr_ = (char*)(device_ptr);
        size_      = bytes;
        flags_     = 0;
        SPDLOG_DEBUG("{} Aligned: bytes {}, base address {}, original {}, extent {}",
                     grox::debug::print_type<Kokkos_Provider<Kokkos::CudaSpace>>(), bytes,
                     (void*)base_addr_, (void*)(region_.data()), region_.extent(0));
    }

    template <>
    inline void
    communication_pool::rma_memory_region<Kokkos_Provider<Kokkos::CudaUVMSpace>>::allocate(
        provider_domain* /*domain*/, size_type bytes)  //
    {
        char* device_ptr = nullptr;
        // cudaMalloc(&device_ptr, bytes);
        region_    = provider_region(device_ptr, size_);
        address_   = (char*)(device_ptr);
        base_addr_ = (char*)(device_ptr);
        size_      = bytes;
        flags_     = 0;
    }

    template <>
    inline void
    communication_pool::rma_memory_region<Kokkos_Provider<Kokkos::CudaHostPinnedSpace>>::allocate(
        provider_domain* /*domain*/, size_type bytes)  //
    {
        char* device_ptr = nullptr;
        // cudaMalloc(&device_ptr, bytes);
        region_    = provider_region(device_ptr, size_);
        address_   = (char*)(device_ptr);
        base_addr_ = (char*)(device_ptr);
        size_      = bytes;
        flags_     = 0;
    }

    template <typename MemorySpace>
    class PoolBufferHandler
        : public BufferHandler<typename rma_archive<rma_buffer<MemorySpace>>::type, MemorySpace> {
    public:
        using region_type  = rma_buffer<MemorySpace>;
        using archive_type = rma_archive<rma_buffer<MemorySpace>>::type;
        using buffer_type  = comm_buff_type<rma_buffer<MemorySpace>>::type;
        using typename BufferHandler<typename rma_archive<rma_buffer<MemorySpace>>::type,
                                     MemorySpace>::size_type;

        static inline std::vector<region_type*> buffers_in_use;

        PoolBufferHandler() {};
        ~PoolBufferHandler() override {};

        virtual buffer_type getBuffer(size_type size, double overallocation) override {
            (void)overallocation;

            communication_pool::rma_memory_region<Kokkos_Provider<MemorySpace>>* buff =
                _pool->allocate_region(size);

            buffers_in_use.push_back(buff);

            return std::make_shared<archive_type>(buff);
        }

        /**
         * @brief Frees a specified buffer.
         *
         * Moves the specified buffer to a free state, making it available
         * for reuse in future buffer requests.
         *
         * @param buffer The buffer to be freed.
         */
        virtual void freeBuffer(buffer_type buffer) override {  //
            if constexpr (std::is_same_v<MemorySpace, typename region_type::memory_space>) {
                auto buff = buffer->buffer_m;
                _pool->deallocate(buff);
                SPDLOG_TRACE("freeBuffer {} buffers_in_use before erase {}, {}", (void*)buff,
                             buffers_in_use.size(), *buff);
                for (auto buf : buffers_in_use) {
                    std::cout << (void*)buf << std::endl;
                }
                buffers_in_use.erase(
                    std::remove(buffers_in_use.begin(), buffers_in_use.end(), buff),
                    buffers_in_use.end());
            }
            SPDLOG_TRACE("freeBuffer - size after erase {}", buffers_in_use.size());
        }

        /**
         * @brief Frees all currently used buffers.
         *
         * Transfers all used buffers to the free state, making them available
         * for reuse. This does not deallocate memory but resets buffer usage.
         */
        virtual void freeAllBuffers() override {
            //
            SPDLOG_TRACE("freeAllBuffers");
            for (auto buff : buffers_in_use) {
                SPDLOG_TRACE("deallocate {}", *buff);
                _pool->deallocate(buff);
            }
            buffers_in_use.clear();
        }

        /**
         * @brief Deletes all buffers.
         *
         * Releases all allocated memory buffers, both used and free.
         * After this call, no buffers are available until new allocations.
         */
        virtual void deleteAllBuffers() override {}

        /**
         * @brief Gets the size of all buffers in use.
         *
         * @return Total size of buffers that are in use in bytes.
         */
        virtual size_type getUsedSize() const override { return 0; }

        /**
         * @brief Gets the size of all free buffers.
         *
         * @return Total size of free buffers in bytes.
         */
        virtual size_type getFreeSize() const override { return 0; }

        virtual int getUsedN() const override { return 0; }

        virtual int getFreeN() const override { return 0; }

        std::shared_ptr<communication_pool::rma_memory_pool<Kokkos_Provider<MemorySpace>>> _pool{
            ippl::communication_pool::get_instance<Kokkos_Provider<MemorySpace>>()};
    };
}  // namespace ippl

template <>
template <typename MemorySpace>
struct fmt::formatter<
    ippl::communication_pool::rma_memory_region<ippl::Kokkos_Provider<MemorySpace>>>
    : ostream_formatter {};

#include "Communicate/BufferHandler.hpp"

#endif
