#ifndef IPPL_LOGGING_BUFFER_HANDLER_HPP
#define IPPL_LOGGING_BUFFER_HANDLER_HPP

#include <iostream>
#include <mpi.h>

namespace ippl {

    using DefaultExec = Kokkos::DefaultExecutionSpace;

    template <class ExecSpace>
    void try_create_view() {
        if constexpr (std::is_same_v<ExecSpace, DefaultExec>) {
            Kokkos::View<int*, ExecSpace> v("v", 10);
            std::cout << ExecSpace::name() << " is safe to use\n";
        } else {
            std::cout << ExecSpace::name() << " exists but is NOT initialized; skipping\n";
        }
    }

    template <class MemorySpace>
    struct DefaultExecFor;

    template <>
    struct DefaultExecFor<Kokkos::HostSpace> {
        using type = Kokkos::DefaultHostExecutionSpace;
    };

    template <>
    struct DefaultExecFor<Kokkos::SharedHostPinnedSpace> {
        using type = Kokkos::DefaultHostExecutionSpace;
    };

    template <>
    struct DefaultExecFor<Kokkos::CudaSpace> {
        using type = Kokkos::Cuda;
    };

    template <>
    struct DefaultExecFor<Kokkos::CudaUVMSpace> {
        using type = Kokkos::Cuda;
    };
#if 0
    template <>
    struct DefaultExecFor<Kokkos::HIPSpace> {
        using type = Kokkos::HIP;
    };
#endif

    // template <>
    // struct DefaultExecFor<Kokkos::HIPSpace> {
    //     using type = Kokkos::HIP;
    // };

    // ---------------------------------------
    // singleton access to network/testnet
    // ---------------------------------------
    /////////////////////////////////////////////////////////////////////////////////////
    template <typename MemorySpace>
    static std::shared_ptr<ippl::BufferHandler<MemorySpace>> get_buffer_handler_instance() {
        static std::shared_ptr<ippl::BufferHandler<MemorySpace>> comm_buff_handler_ptr = nullptr;

        using Exec = typename DefaultExecFor<MemorySpace>::type;
        if constexpr (std::is_same_v<Exec, DefaultExec>) {
            if (comm_buff_handler_ptr == nullptr) {
                // std::cout << "Creating pool buffer handler "
                //           << grox::debug::print_type<ippl::BufferHandler<MemorySpace>>()
                //           << std::endl;
                comm_buff_handler_ptr = std::make_shared<ippl::PoolBufferHandler<MemorySpace>>();
            }
        } else {
        }

        return comm_buff_handler_ptr;
    }

    template <typename MemorySpace>
    static std::shared_ptr<ippl::PoolBufferHandler<MemorySpace>>
    get_comm_buffer_handler_instance() {
        return std::dynamic_pointer_cast<PoolBufferHandler<MemorySpace>>(
            get_buffer_handler_instance<MemorySpace>());
    }

    template <typename MemorySpace>
    static std::shared_ptr<ippl::PoolBufferHandler<MemorySpace>> get_multispace_bufferhandler() {
        return std::dynamic_pointer_cast<PoolBufferHandler<MemorySpace>>(
            get_buffer_handler_instance<MemorySpace>());
    }

    // ---------------------------------------
    //
    // ---------------------------------------
    template <typename MemorySpace>
    LoggingBufferHandler<MemorySpace>::LoggingBufferHandler(
        std::shared_ptr<ippl::BufferHandler<MemorySpace>> handler, int rank)
        : handler_m(handler)
        , rank_m(rank) {}

    template <typename MemorySpace>
    LoggingBufferHandler<MemorySpace>::LoggingBufferHandler() {
        // using Exec = typename DefaultExecFor<MemorySpace>::type;
        // if constexpr (Exec::is_available()) {
        handler_m = get_comm_buffer_handler_instance<MemorySpace>();
        // }
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_m);
    }

    template <typename MemorySpace>
    typename LoggingBufferHandler<MemorySpace>::buffer_type
    LoggingBufferHandler<MemorySpace>::getBuffer(size_type size, double overallocation) {
        auto buffer = handler_m->getBuffer(size, overallocation);
        logMethod("getBuffer", {{"size", std::to_string(size)},
                                {"overallocation", std::to_string(overallocation)}});
        return buffer;
    }

    template <typename MemorySpace>
    void LoggingBufferHandler<MemorySpace>::freeBuffer(buffer_type buffer) {
        handler_m->freeBuffer(buffer);
        logMethod("freeBuffer", {});
    }

    template <typename MemorySpace>
    void LoggingBufferHandler<MemorySpace>::freeAllBuffers() {
        if (handler_m) {
            handler_m->freeAllBuffers();
            logMethod("freeAllBuffers", {});
        }
    }

    template <typename MemorySpace>
    void LoggingBufferHandler<MemorySpace>::deleteAllBuffers() {
        if (handler_m) {
            handler_m->deleteAllBuffers();
            logMethod("deleteAllBuffers", {});
        }
    }

    template <typename MemorySpace>
    typename LoggingBufferHandler<MemorySpace>::size_type
    LoggingBufferHandler<MemorySpace>::getUsedSize() const {
        return handler_m ? handler_m->getUsedSize() : 0;
    }

    template <typename MemorySpace>
    typename LoggingBufferHandler<MemorySpace>::size_type
    LoggingBufferHandler<MemorySpace>::getFreeSize() const {
        return handler_m ? handler_m->getFreeSize() : 0;
    }

    template <typename MemorySpace>
    int LoggingBufferHandler<MemorySpace>::getUsedN() const {
        return handler_m ? handler_m->getUsedN() : 0;
    }

    template <typename MemorySpace>
    int LoggingBufferHandler<MemorySpace>::getFreeN() const {
        return handler_m ? handler_m->getFreeN() : 0;
    }

    template <typename MemorySpace>
    const std::vector<LogEntry>& LoggingBufferHandler<MemorySpace>::getLogs() const {
        return logEntries_m;
    }

    template <typename MemorySpace>
    void LoggingBufferHandler<MemorySpace>::logMethod(
        const std::string& methodName, const std::map<std::string, std::string>& parameters) {
        auto t = std::chrono::high_resolution_clock::now();

        // std::stringstream temp;
        // temp << t.time_since_epoch().count() << "\t" << methodName << "\t"
        //      << handler_m->getUsedSize() << "\t" << handler_m->getFreeSize() << "\t"
        //      << handler_m->getUsedN() << "\t" << handler_m->getFreeN() << "\t"
        //      << MemorySpace::name() << "\t" << rank_m << std::endl;
        // ;
        // std::cout << temp.str();

        logEntries_m.push_back({methodName, parameters, handler_m->getUsedSize(),
                                handler_m->getFreeSize(), handler_m->getUsedN(),
                                handler_m->getFreeN(), MemorySpace::name(), rank_m, t});
    }

}  // namespace ippl

#endif
