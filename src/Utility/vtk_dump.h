#include <iostream>
#include <iomanip>
#include <fstream>
//
#include "Ippl.h"
#include "Manager/datatypes.h"

template <typename T, unsigned Dim=3>
using ScalarField_t = ippl::Field<T, Dim, Mesh_t<Dim>, Centering_t<Dim>>;

// Define vtk write function for plotting the fields
template <typename T>
void write_VTK_field(std::string path, ScalarField_t<T>& rho, int iteration) {
    typename ScalarField_t<T>::view_type::host_mirror_type host_view = rho.getHostMirror();
    Kokkos::deep_copy(host_view, rho.getView());
    std::ofstream vtkout;
    vtkout.precision(10);
    vtkout.setf(std::ios::scientific, std::ios::floatfield);

    std::stringstream fname;
    fname << path;
    fname << "/scalar_";
    fname << std::setw(4) << std::setfill('0') << iteration;
    fname << ".vtk";

    // open a new data file for this iteration and start with header
    vtkout.open(fname.str().c_str(), std::ios::out);
    if (!vtkout) {
        std::cout << "couldn't open" << std::endl;
    }
    auto gridspacing = rho.get_mesh().getMeshSpacing();
    auto gridsize = rho.get_mesh().getGridsize();

    vtkout << "# vtk DataFile Version 2.0" << std::endl;
    vtkout << "GaussianSource" << std::endl;
    vtkout << "ASCII" << std::endl;
    vtkout << "DATASET STRUCTURED_POINTS" << std::endl;
    vtkout << "DIMENSIONS "
           << static_cast<int>(gridsize[0])+1 << " "
           << static_cast<int>(gridsize[1])+1 << " "
           << static_cast<int>(gridsize[2])+1 << std::endl;
    vtkout << "ORIGIN "
           << 0.0 << " "
           << 0.0 << " "
           << 0.0 << std::endl;
    vtkout << "SPACING "
           << gridspacing[0] << " "
           << gridspacing[1] << " "
           << gridspacing[2] << std::endl;
    vtkout << "CELL_DATA " << static_cast<int64_t>((gridsize[0]) * (gridsize[1]) * (gridsize[2])) << std::endl;

    vtkout << "SCALARS Phi float" << std::endl;
    vtkout << "LOOKUP_TABLE default" << std::endl;
    for (int z = 1; z < gridsize[2] + 1; z++) {
        for (int y = 1; y < gridsize[1] + 1; y++) {
            for (int x = 1; x < gridsize[0] + 1; x++) {
                vtkout << host_view(x, y, z) << std::endl;
            }
        }
    }

    // close the output file for this iteration:
    vtkout.close();
}


