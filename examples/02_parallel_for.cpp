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


/*
  Start by declaring the types the particles will store. The first element
  will represent the coordinates, the second will be the particle's ID, the
  third velocity, and the fourth the radius of the particle.
*/

using DataTypes = Cabana::MemberTypes<double[3], // position(0)
                                      int, // ids(1)
                                      double[3] // velocity(2)
                                      >;

/*
  Next declare the data layout of the AoSoA. We use the host space here
  for the purposes of this example but all memory spaces, vector lengths,
  and member type configurations are compatible.
*/
const int VectorLength = 8;
using MemorySpace = Kokkos::HostSpace;
using ExecutionSpace = Kokkos::DefaultHostExecutionSpace;
using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;
using AoSoAType = Cabana::AoSoA<DataTypes, DeviceType, VectorLength>;

// auto aosoa_position = Cabana::slice<0>     ( aosoa,    "aosoa_position");
// auto aosoa_ids = Cabana::slice<1>          ( aosoa,    "aosoa_ids");
// auto aosoa_velocity = Cabana::slice<2>     ( aosoa,    "aosoa_velocity");

void fluid_stage_1(auto aosoa, double dt, auto limits){
  auto aosoa_pos = Cabana::slice<0>     ( aosoa,    "aosoa_pos");
  auto aosoa_vel = Cabana::slice<2>          ( aosoa,    "aosoa_vel");

  auto half_dt = dt * 0.5;
  auto fluid_stage_1_lambda_func = KOKKOS_LAMBDA( const int i )
    {
      aosoa_pos(i, 0) += aosoa_vel( i, 0 ) * half_dt;
      aosoa_pos(i, 1) += aosoa_vel( i, 1 ) * half_dt;
      aosoa_pos(i, 2) += aosoa_vel( i, 2 ) * half_dt;
    };
  Kokkos::RangePolicy<ExecutionSpace> policy( limits[0], limits[1] );
  Kokkos::parallel_for( "CabanaSPH:Integrator:FluidStage1", policy,
                        fluid_stage_1_lambda_func );
}


void run()
{
  int comm_rank = -1;
  MPI_Comm_rank( MPI_COMM_WORLD, &comm_rank );

  if ( comm_rank == 0 )
    std::cout << "Cabana Rigid body solver example\n" << std::endl;

  auto num_particles = 16;
  AoSoAType aosoa( "particles", num_particles );
  auto aosoa_pos = Cabana::slice<0>     ( aosoa,    "aosoa_pos");
  auto aosoa_vel = Cabana::slice<2>          ( aosoa,    "aosoa_vel");
  Cabana::deep_copy( aosoa_pos, 0. );
  Cabana::deep_copy( aosoa_vel, 1. );

  auto dt = 3.;

  std::vector<int> fluid_limits = {0, 12};
  fluid_stage_1(aosoa, 2. * dt, fluid_limits);
  for ( std::size_t i = 0; i < aosoa_pos.size(); ++i )
    {
      std::cout << "\n";
      std::cout << "position of " << i << " is " << aosoa_pos( i, 0) << ", " << aosoa_pos( i, 1 ) << ", " << aosoa_pos( i, 2 ) << "\n";
    }

}

int main( int argc, char* argv[] )
{

  MPI_Init( &argc, &argv );
  Kokkos::initialize( argc, argv );

  run();

  Kokkos::finalize();

  MPI_Finalize();
  return 0;
}
