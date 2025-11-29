//
// Class Environment
//
#include "Ippl.h"

#include "Environment.h"

namespace ippl {
    namespace mpi {
        namespace detail {
            // -------------------------------------------------------------
            // an MPI error handling type that we can use to intercept
            // MPI errors if we enable the error handler
            MPI_Errhandler ippl_mpi_errhandler = 0;
            bool error_handler_initialized_    = true;

            std::string error_message(int code) {
                int N          = 1023;
                int const len  = 1024;
                char buff[len] = {0};
                MPI_Error_string(code, buff, &N);
                return std::string(buff);
            }

            // -------------------------------------------------------------
            // function that converts an MPI error into an exception
            void ippl_MPI_Handler(MPI_Comm*, int* errorcode, ...) {
                std::string err = error_message(*errorcode);
                std::cout << err << std::endl;
                throw std::runtime_error(err);
            }

            // -------------------------------------------------------------
            // set an error handler for communicators that will be called
            // on any error instead of the default behavior of program termination
            void set_error_handler() {
                std::cout << "MPI Error handler installed " << std::endl;
                MPI_Comm_create_errhandler(ippl_MPI_Handler, &detail::ippl_mpi_errhandler);
                MPI_Comm_set_errhandler(MPI_COMM_WORLD, detail::ippl_mpi_errhandler);
            }
        }  // namespace detail

        Environment::Environment(int& argc, char**& argv, const MPI_Comm& comm)
            : comm_m(comm) {
            if (!initialized()) {
                MPI_Init(&argc, &argv);
            }
            detail::set_error_handler();
            detail::error_handler_initialized_ = true;
        }

        Environment::~Environment() {
            // remove error handler if we installed it
            if (detail::error_handler_initialized_) {
                // PIKA_ASSERT(detail::pika_mpi_errhandler != 0);
                detail::error_handler_initialized_ = false;
                MPI_Errhandler_free(&detail::ippl_mpi_errhandler);
                detail::ippl_mpi_errhandler = 0;
            }

            if (!finalized()) {
                MPI_Finalize();
            }
        }

        bool Environment::initialized() {
            int flag = 0;
            MPI_Initialized(&flag);
            return (flag != 0);
        }

        bool Environment::finalized() {
            int flag = 0;
            MPI_Finalized(&flag);
            return (flag != 0);
        }
    }  // namespace mpi
}  // namespace ippl
