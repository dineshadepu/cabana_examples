/****************************************************************************
 * Copyright (c) 2018-2022 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <Cabana_Core.hpp>

#include <iostream>

//---------------------------------------------------------------------------//
// HDF5 output example
//---------------------------------------------------------------------------//
void hdf5_output()
{
    /*
      HDF5 is a parallel file format for large datasets. In this example, we
      will illustrate the process of storing a list of particles with
      properties, such as position, velocity, mass, radius, etc., in an HDF5
      file format.
    */

    /*
       Get parameters from the communicator. We will use MPI_COMM_WORLD for
       this example but any MPI communicator may be used.
    */
    int comm_rank = -1;
    MPI_Comm_rank( MPI_COMM_WORLD, &comm_rank );

    if ( comm_rank == 0 )
      std::cout << "Cabana AoSoA copy mirror example\n" << std::endl;

    /*
      Start by declaring the types the particles will store. The first element
      will represent the coordinates, the second will be the particle's ID, the
      third velocity, and the fourth the radius of the particle.
    */
    using DataTypes = Cabana::MemberTypes<double[3], int, double[3], double>;

    /*
      Next declare the data layout of the AoSoA. We use the default spaces to
      work across all supported backends. Further, this example can be
      configured for a specific memory space, vector lengths, and member type.
    */
    const int VectorLength = 8;
    using ExecutionSpace = Kokkos::DefaultExecutionSpace;
    using MemorySpace = typename ExecutionSpace::memory_space;
    using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;

    /*
      Create the AoSoA.
    */
    int num_particles = 2;
    Cabana::AoSoA<DataTypes, DeviceType, VectorLength> aosoa( "A",
                                                              num_particles );

    // Create a mirror view of the aosoa on the host for accessing it legally
    auto aosoa_host =
      Cabana::create_mirror_view_and_copy( Kokkos::HostSpace(), aosoa );
    /*
      Get the particle ids, coordinates, velocity, radius.

      The following slices are in hostspace.
    */
    auto positions_host = Cabana::slice<0>( aosoa_host, "positions" );
    auto ids_host = Cabana::slice<1>( aosoa_host, "ids" );
    auto velocity_host = Cabana::slice<2>( aosoa_host, "velocity" );
    auto radius_host = Cabana::slice<3>( aosoa_host, "radius" );

    // initialize the particle properties
    for ( std::size_t i = 0; i < aosoa.size(); ++i )
    {
        // set ids of the particles
        ids_host( i ) = i;

	// set positions of the particles
        positions_host( i, 0 ) = 2. * i;
        positions_host( i, 1 ) = 0.;
        positions_host( i, 2 ) = 0.;

        // set the velocity of each particle
        velocity_host( i, 0 ) = 10.;
        velocity_host( i, 1 ) = 0.;
        velocity_host( i, 2 ) = 0.;

        // set the radius of each particle
        radius_host( i ) = 0.1;
    }

    std::cout << "Before positions";
    for ( std::size_t i = 0; i < positions_host.size(); ++i )
      {
        std::cout << "\n";
        std::cout << positions_host(i, 0) << ", ";
        std::cout << positions_host(i, 1) << ", ";
        std::cout << positions_host(i, 2);
      }
    //
    Cabana::deep_copy( aosoa, aosoa_host );

    // A configuration object is necessary for tuning HDF5 options.
    Cabana::Experimental::HDF5ParticleOutput::HDF5Config h5_config;

    // For example, MPI I/O is independent by default, but collective operations
    // can be enabled by setting:
    h5_config.collective = true;

    /*
      We will evolve the system for a total time of 100 timesteps and
      output the properties at an interval of every 10 steps.
    */
    auto dt = 1e-2;
    auto final_time = 100. * dt;
    auto time = 0.;
    int steps = final_time / dt;
    std::cout << std::endl;
    std::cout << "steps: " << steps << std::endl;
    int print_freq = 10;

    // Create slices in the execution space to be used in the
    // Kokkos::parallel_for loops
    auto positions = Cabana::slice<0>( aosoa, "positions" );
    auto ids = Cabana::slice<1>( aosoa, "ids" );
    auto velocity = Cabana::slice<2>( aosoa, "velocity" );
    auto radius = Cabana::slice<3>( aosoa, "radius" );

    // Main timestep loop
    for ( int step = 0; step < steps; step++ )
	{
	  // Update positions
	  Kokkos::RangePolicy<ExecutionSpace> policy( 0, aosoa.size() );
	  Kokkos::parallel_for(
			       "move_particles", policy, KOKKOS_LAMBDA( const int i ) {
				 positions( i, 0 ) += velocity( i, 0 ) * dt;
				 positions( i, 1 ) += velocity( i, 1 ) * dt;
				 positions( i, 2 ) += velocity( i, 2 ) * dt;
			       } );
	  Kokkos::fence();

	  if ( step % print_freq == 0 )
	    {
	      /*
	        Now we write all particle properties to HDF5 files. The arguments
	        are as follows:

	        1. HDF5 configuration: specify tuning details
	        2. output file prefix
	        3. MPI communicator
	        4. Current time step
	        5. Current simulation time
	        6. Number of particles to output (it is often desirable to ignore
		ghosted particles)
	        7. Position slice
	        8. Variadic list of particle fields
	        9. ...
	      */
	      Cabana::Experimental::HDF5ParticleOutput::writeTimeStep(h5_config, "particles", MPI_COMM_WORLD, step, time,
								      num_particles, positions_host, ids_host, velocity_host, radius_host );

	      if ( comm_rank == 0 )
		std::cout << "Output for step " << step << "/" << steps
			  << std::endl;
	    }

	  time += dt;
	}

      /*
	The created HDF5 files with XMF metadata can be read by many visualization
	programs. This example can be run with any number of MPI ranks, for
	parallel particle output.
      */

      // Create a mirror view of the aosoa on the host for accessing it legally
    Cabana::deep_copy( aosoa_host, aosoa );
    // Cabana::deep_copy( aosoa_vel, 1. );
    std::cout << "After positions";
    for ( std::size_t i = 0; i < positions_host.size(); ++i )
      {
	std::cout << "\n";
	std::cout << positions_host(i, 0) << ", ";
	std::cout << positions_host(i, 1) << ", ";
	std::cout << positions_host(i, 2);
      }
}

int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    hdf5_output();

    Kokkos::finalize();

    MPI_Finalize();
    return 0;
}
