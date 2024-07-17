// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifdef HAVE_CONFIG_H
# include "config.h"
#endif
#include <iostream>
#include <dune/common/parallel/mpihelper.hh> // An initializer of MPI
#include <dune/common/exceptions.hh> // We use exceptions

#include <string>
#include <cmath>

#include <dune/common/parametertreeparser.hh>
#include <dune/common/fvector.hh>

#include <duneuro/driver/driver_factory.hh>
#include <duneuro/io/point_vtk_writer.hh>
#include <duneuro/io/field_vector_reader.hh>
#include <duneuro/common/dense_matrix.hh>
#include <duneuro/common/function.hh>

int main(int argc, char** argv)
{
  try{
    // Maybe initialize MPI
    Dune::MPIHelper& helper = Dune::MPIHelper::instance(argc, argv);
    
    std::cout << "The goal of this program is to quickly test the tDCS solver implemented in DUNEuro." << std::endl;
    
    using Scalar = double;
    constexpr int dim = 3;
    using Driver = duneuro::DriverInterface<dim>;
    
    // read parameter tree
    std::cout << "Reading parameter tree" << std::endl;
    Dune::ParameterTree config_tree;
    Dune::ParameterTreeParser config_parser;
    
    config_parser.readINITree("fitted_configs.ini", config_tree);
    std::cout << "Parameter tree read" << std::endl;
    
    bool write_output = config_tree.get<bool>("output.write");
    
    // create driver
    std::cout << "Creating driver" << std::endl;
    std::unique_ptr<Driver> driver_ptr = duneuro::DriverFactory<dim>::make_driver(config_tree);
    std::cout << "Driver created" << std::endl;
    
    // set stimulation electrodes
    std::cout << "Setting electrodes" << std::endl;
    Dune::ParameterTree electrode_config = config_tree.sub("electrodes");
    std::vector<Dune::FieldVector<Scalar, dim>> electrodes = duneuro::FieldVectorReader<Scalar, dim>::read(electrode_config.get<std::string>("filename"));
    driver_ptr->setElectrodes(electrodes, electrode_config);
    std::cout << "Electrodes set" << std::endl;
    
    // solve tDCS forward problem
    std::cout << "Solving tDCS forward problem for one electrode pair" << std::endl;
    std::cout << "First computing potential" << std::endl;
    std::unique_ptr<duneuro::DenseMatrix<Scalar>> potential = driver_ptr->solveTDCSForward(config_tree);
    std::cout << "Potential computed" << std::endl;
    std::cout << "Computing volume currents" << std::endl;
    Dune::ParameterTree gradient_config = config_tree.sub("potential_gradient");
    std::unique_ptr<duneuro::DenseMatrix<Scalar>> electrical_current = driver_ptr->evaluateMultipleFunctionsAtElementCenters(*potential, gradient_config);
    std::cout << "Volume currents computed" << std::endl;
    std::cout << "Computing potentials at element centers" << std::endl;
    gradient_config["evaluation_return_type"] = "potential";
    std::unique_ptr<duneuro::DenseMatrix<Scalar>> electrical_potential_at_centers = driver_ptr->evaluateMultipleFunctionsAtElementCenters(*potential, gradient_config);
    std::cout << "Potentials computed" << std::endl;
    std::cout << "Computing element centers" << std::endl;
    std::vector<Dune::FieldVector<Scalar, dim>> element_centers = std::get<0>(driver_ptr->elementStatistics());
    std::cout << "Element centers computed" << std::endl;
    
    // visualization
    if(write_output) {
      std::cout << "We now visualize the solution using the vtk and vtu formats" << std::endl;
      std::cout << "We first write the headmodel" << std::endl;
      auto volume_writer_ptr = driver_ptr->volumeConductorVTKWriter(config_tree);
      std::unique_ptr<duneuro::Function> tdcsSolution = driver_ptr->makeDomainFunctionFromMatrixRow(*potential, 1);
      volume_writer_ptr->addVertexData(*tdcsSolution, "potential");
      volume_writer_ptr->write(config_tree.sub("output"));
      
      std::cout << "We now write the electrodes" << std::endl;
      duneuro::PointVTKWriter<Scalar, dim> electrode_writer{electrodes};
      std::string filename_electrodes = config_tree.get<std::string>("output.filename_electrodes");
      electrode_writer.write(filename_electrodes);
      
      std::cout << "Writing electrical current" << std::endl;
      size_t nrElements = element_centers.size();
      std::vector<Dune::FieldVector<Scalar, dim>> currentsAtCenters(nrElements);
      std::vector<Scalar> potentialAtCenters(nrElements);
      std::vector<Scalar> currentMagnitudes(nrElements);
      
      for(size_t i = 0; i < nrElements; ++i) {
        currentMagnitudes[i] = 0.0;
        for(size_t j = 0; j < dim; ++j) {
          currentsAtCenters[i][j] = (*electrical_current)(1, 3 * i + j);
          currentMagnitudes[i] += currentsAtCenters[i][j] * currentsAtCenters[i][j];
        }
        potentialAtCenters[i] = (*electrical_potential_at_centers)(1, i);
        currentMagnitudes[i] = std::sqrt(currentMagnitudes[i]);
      }
      duneuro::PointVTKWriter<Scalar, dim> currentWriter{element_centers};
      currentWriter.addVectorData("electrical_current", currentsAtCenters);
      currentWriter.addScalarData("electrical_potential", potentialAtCenters);
      currentWriter.addScalarData("electrical_current_magnitude", currentMagnitudes);
      std::string filename_current = config_tree.get<std::string>("output.filename_current");
      currentWriter.write(filename_current);
      
      std::cout << "Output written" << std::endl;
    }
    
    std::cout << "The program didn't crash!" << std::endl;
    
    return 0;
  }
  catch (Dune::Exception &e){
    std::cerr << "Dune reported error: " << e << std::endl;
  }
  catch (...){
    std::cerr << "Unknown exception thrown!" << std::endl;
  }
}
