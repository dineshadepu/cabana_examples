// /****************************************************************************
//  * Copyright (c) 2018-2022 by the Cabana authors                            *
//  * All rights reserved.                                                     *
//  *                                                                          *
//  * This file is part of the Cabana library. Cabana is distributed under a   *
//  * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
//  * the top-level directory.                                                 *
//  *                                                                          *
//  * SPDX-License-Identifier: BSD-3-Clause                                    *
//  ****************************************************************************/
// // https://cg.informatik.uni-freiburg.de/course_notes/sim_03_particleFluids.pdf : Slide 28

// #include <Cabana_Core.hpp>
// #include <math.h>

// #include <iostream>

// /*
//   Start by declaring the types the particles will store. The first element
//   will represent the coordinates, the second will be the particle's ID, the
//   third velocity, and the fourth the radius of the particle.
// */

// using DataTypes = Cabana::MemberTypes<double[3], // position(0)
//                                       int, // ids(1)
//                                       double[3], // velocity(2)
//                                       double[3], // acceleration(3)
//                                       double, // mass(4)
//                                       double, // density(5)
//                                       double, // h (smoothing length) (6)
//                                       double, // pressure (7)
//                                       int, // is_fluid(8)
//                                       int, // is_boundary(9)
//                                       double, // rate of density change (10)
//                                       double, // rate of pressure change (11)
//   double, // wij (12)
//   double, // sin (13)
//   double // sin_appr (14)
//   >;
// typedef Kokkos::View<double*>   ViewVectorType;

// /*
//   Next declare the data layout of the AoSoA. We use the host space here
//   for the purposes of this example but all memory spaces, vector lengths,
//   and member type configurations are compatible.
// */
// const int VectorLength = 8;
// using MemorySpace = Kokkos::HostSpace;
// using ExecutionSpace = Kokkos::DefaultHostExecutionSpace;
// // using ExecutionSpace = Kokkos::Cuda;
// // using MemorySpace = ExecutionSpace::memory_space;
// using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;
// using AoSoAType = Cabana::AoSoA<DataTypes, DeviceType, VectorLength>;

// // auto aosoa_position = Cabana::slice<0>     ( aosoa,    "aosoa_position");
// // auto aosoa_ids = Cabana::slice<1>          ( aosoa,    "aosoa_ids");
// // auto aosoa_velocity = Cabana::slice<2>     ( aosoa,    "aosoa_velocity");
// // auto aosoa_acc = Cabana::slice<3>          ( aosoa,    "aosoa_acc");
// // auto aosoa_mass = Cabana::slice<4>        ( aosoa,    "aosoa_mass");
// // auto aosoa_density = Cabana::slice<5>     ( aosoa,    "aosoa_density");
// // auto aosoa_h = Cabana::slice<6>           ( aosoa,    "aosoa_h");
// // auto aosoa_p = Cabana::slice<7>           ( aosoa,    "aosoa_p");
// // auto aosoa_is_fluid = Cabana::slice<8>    ( aosoa,    "aosoa_is_fluid");
// // auto aosoa_is_boundary = Cabana::slice<9>    ( aosoa,    "aosoa_is_boundary");
// // auto aosoa_density_acc = Cabana::slice<10>           ( aosoa,    "aosoa_density_acc");
// // auto aosoa_p_acc = Cabana::slice<11>           ( aosoa,    "aosoa_p_acc");
// // auto aosoa_wij = Cabana::slice<12>           ( aosoa,    "aosoa_wij");
// // auto aosoa_sin = Cabana::slice<13>           ( aosoa,    "aosoa_sin");
// // auto aosoa_sin_appr = Cabana::slice<14>           ( aosoa,    "aosoa_sin_appr");

// // Neighbour list
// // using ListAlgorithm = Cabana::FullNeighborTag;
// // using ListType =
// // Cabana::VerletList<MemorySpace, ListAlgorithm, Cabana::VerletLayout2D>;
// using ListAlgorithm = Cabana::FullNeighborTag;
// using ListType =
//   Cabana::VerletList<MemorySpace, ListAlgorithm, Cabana::VerletLayoutCSR,
// 		     Cabana::TeamOpTag>;



// KOKKOS_INLINE_FUNCTION
// void compute_quintic_wij(double rij, double h, double *result){
//   double h1 =  1. / h;
//   double q =  rij * h1;
//   // TODO this changes with the dimension
//   double fac = 1.0 / 120.0 * h1;
//   // double fac = M_1_PI *  7. / 478. * h1 * h1;

//   double tmp3 = 3. - q;
//   double tmp2 = 2. - q;
//   double tmp1 = 1. - q;

//   double val = 0.;
//   if (q > 3.) {
//     val = 0.;
//   } else if ( q > 2.) {
//     val = tmp3 * tmp3 * tmp3 * tmp3 * tmp3;
//   } else if ( q > 1.) {
//     val = tmp3 * tmp3 * tmp3 * tmp3 * tmp3;
//     val -= 6.0 * tmp2 * tmp2 * tmp2 * tmp2 * tmp2;
//   } else {
//     val = tmp3 * tmp3 * tmp3 * tmp3 * tmp3;
//     val -= 6.0 * tmp2 * tmp2 * tmp2 * tmp2 * tmp2;
//     val += 15. * tmp1 * tmp1 * tmp1 * tmp1 * tmp1;
//   }

//   *result = val * fac;
// }


// KOKKOS_INLINE_FUNCTION
// void compute_quintic_gradient_wij(double *xij, double rij, double h, double *result){
//   double h1 =  1. / h;
//   double q =  rij * h1;

//   // TODO this changes with the dimension
//   double fac = 1.0 / 120.0 * h1;
//   // double fac = M_1_PI *  7. / 478. * h1 * h1;

//   double tmp3 = 3. - q;
//   double tmp2 = 2. - q;
//   double tmp1 = 1. - q;

//   double val = 0.;
//   if (rij > 1e-12){
//     if (q > 3.) {
//       val = 0.;
//     } else if ( q > 2.) {
//       val = -5.0 * tmp3 * tmp3 * tmp3 * tmp3;
//       val *= h1 / rij;
//     } else if ( q > 1.) {
//       val = -5.0 * tmp3 * tmp3 * tmp3 * tmp3;
//       val += 30.0 * tmp2 * tmp2 * tmp2 * tmp2;
//       val *= h1 / rij;
//     } else {
//       val = -5.0 * tmp3 * tmp3 * tmp3 * tmp3;
//       val += 30.0 * tmp2 * tmp2 * tmp2 * tmp2;
//       val -= 75.0 * tmp1 * tmp1 * tmp1 * tmp1;
//       val *= h1 / rij;
//     }
//   } else {
//     val = 0.;
//   }

//   double tmp = val * fac;
//   result[0] = tmp * xij[0];
//   result[1] = tmp * xij[1];
//   result[2] = tmp * xij[2];
// }


// void get_equispaced_points_gpu(double start, int end, int no_particles, ViewVectorType x) {
//   double spacing = (end - start) / no_particles;

//   auto equi_func = KOKKOS_LAMBDA( const int i )
//     {
//       x(i) = start + i * spacing;
//     };
//   Kokkos::RangePolicy<ExecutionSpace> policy( 0, no_particles);
//   Kokkos::parallel_for( "EquiFunc", policy,
// 			equi_func );
// }


// void compute_sin_gpu(int no_particles, ViewVectorType x, ViewVectorType sin_x) {
//   auto sin_func = KOKKOS_LAMBDA( const int i )
//     {
//       sin_x(i) = sin(x(i));
//     };
//   Kokkos::RangePolicy<ExecutionSpace> policy( 0, no_particles);
//   Kokkos::parallel_for( "SinFunc", policy,
// 			sin_func );
// }


// std::vector<double> get_equispaced_points(double start, int end, int no_particles) {
//   std::vector<double> x;
//   double spacing = (end - start) / no_particles;

//   for (int i = 0; i < no_particles; ++i) {
//     x.push_back(start + i * spacing);
//   }

//   return x;
// }


// std::vector<double> compute_sin(std::vector<double> x) {
//   std::vector<double> sin_x;

//   for (int i = 0; i < x.size(); ++i) {
//     sin_x.push_back(sin(x[i]));
//   }

//   return sin_x;
// }


// // auto create_aosoa_particles(int num_particles)
// // {
// //   // sum all the number of particles and create aosoa
// //   AoSoAType aosoa( "particles", num_particles );

// //   auto ids = Cabana::slice<0>( aosoa, "ids" );
// //   auto m = Cabana::slice<4>( aosoa, "mass" );
// //   for ( std::size_t i = 0; i < aosoa.size(); ++i )
// //     {
// //       ids( i ) = i;
// //       m( i ) = m_array[i];
// //     }
// //   return aosoa;
// // }

// void approximate_sin(AoSoAType aosoa,
// 		     ListType * verlet_list_source,
// 		     int * limits){
//   auto position = Cabana::slice<0>     ( aosoa,    "position");
//   auto mass = Cabana::slice<4>        ( aosoa,    "mass");
//   auto density = Cabana::slice<5>     ( aosoa,    "density");
//   auto h = Cabana::slice<6>           ( aosoa,    "h");
//   auto aosoa_wij = Cabana::slice<12>           ( aosoa,    "wij");
//   auto sin_val = Cabana::slice<13>           ( aosoa,    "sin");
//   auto sin_appr = Cabana::slice<14>           ( aosoa,    "sin_appr");

//   auto sin_lambda_func = KOKKOS_LAMBDA( const int i, const int j )
//     {
//       // Function approximation starts here
//       const double h_i = h( i );

//       double wij = 0.;
//       const double pos_i[3] = {position( i, 0 ),
// 	position( i, 1 ),
// 	position( i, 2 )};

//       const double pos_j[3] = {position( j, 0 ),
// 	position( j, 1 ),
// 	position( j, 2 )};

//       const double pos_ij[3] = {position( i, 0 ) - position( j, 0 ),
// 	position( i, 1 ) - position( j, 1 ),
// 	position( i, 2 ) - position( j, 2 )};

//       const double r2ij = pos_ij[0] * pos_ij[0] + pos_ij[1] * pos_ij[1] + pos_ij[2] * pos_ij[2];
//       const double rij = sqrt(r2ij);

//       // compute WIJ
//       compute_quintic_wij(rij, h_i, &wij);

//       // Function approximation starts here
//       const double m_j = mass( j );
//       const double rho_j = density( j );

//       sin_appr( i ) += m_j / rho_j * sin_val( j ) * wij;
//       // if (i == 0){
//       // std::cout << "inside" << "\n";
//       // 	std::cout << "j is " << j << "\n";
// 	// }
//     };

//   Kokkos::RangePolicy<ExecutionSpace> policy(limits[0], limits[1]);


//   Cabana::neighbor_parallel_for( policy,
//                                  sin_lambda_func,
//                                  *verlet_list_source,
//                                  Cabana::FirstNeighborsTag(),
//                                  Cabana::SerialOpTag(),
//                                  "SinAppr" );
//   Kokkos::parallel_for("SinAppr0", policy, KOKKOS_LAMBDA(int i) { sin_lambda_func(i, i); });
//   Kokkos::fence();
// }

// void output_data(AoSoAType aosoa, int num_particles, int step, double time)
// {
//   auto position = Cabana::slice<0>     ( aosoa,    "position");
//   auto ids = Cabana::slice<1>          ( aosoa,    "ids");
//   auto velocity = Cabana::slice<2>     ( aosoa,    "velocity");
//   auto acc = Cabana::slice<3>          ( aosoa,    "acc");
//   auto mass = Cabana::slice<4>        ( aosoa,    "mass");
//   auto density = Cabana::slice<5>     ( aosoa,    "density");
//   auto h = Cabana::slice<6>           ( aosoa,    "h");
//   auto p = Cabana::slice<7>           ( aosoa,    "p");
//   auto is_fluid = Cabana::slice<8>    ( aosoa,    "is_fluid");
//   auto is_boundary = Cabana::slice<9>    ( aosoa,    "is_boundary");
//   auto density_acc = Cabana::slice<10>           ( aosoa,    "density_acc");
//   auto p_acc = Cabana::slice<11>           ( aosoa,    "p_acc");
//   auto wij = Cabana::slice<12>           ( aosoa,    "wij");
//   auto sin_val = Cabana::slice<13>           ( aosoa,    "sin");
//   auto sin_appr = Cabana::slice<14>           ( aosoa,    "sin_appr");

//   Cabana::Experimental::HDF5ParticleOutput::HDF5Config h5_config;
//   Cabana::Experimental::HDF5ParticleOutput::
//     writeTimeStep(
//                   h5_config, "particles", MPI_COMM_WORLD,
//                   step, time, num_particles, position,
//                   ids, sin_val, sin_appr);
// }



// //---------------------------------------------------------------------------//
// // TODO: explain this function in short
// //---------------------------------------------------------------------------//
// void run()
// {
//   int comm_rank = -1;
//   MPI_Comm_rank( MPI_COMM_WORLD, &comm_rank );

//   if ( comm_rank == 0 )
//     std::cout << "Cabana SPH solver sin example\n" << std::endl;

//   // create points on a 1d grid from 0 to 2 * pi
//   int no_particles = 100;

//   int limits[2] = {0, no_particles};
//   auto length = 2. * M_PI;
//   auto spacing = length / no_particles;
//   // ViewVectorType x_d( "x", no_particles );
//   // ViewVectorType sin_x_d( "sin_x", no_particles );

//   // get_equispaced_points_gpu(0., length, no_particles, x_d);
//   // compute_sin_gpu(no_particles, x_d, sin_x_d);

//   std::vector<double> x_d = get_equispaced_points(0., length, no_particles);
//   std::vector<double> sin_x_d = compute_sin(x_d);


//   AoSoAType aosoa( "particles", no_particles );
//   auto aosoa_position = Cabana::slice<0>     ( aosoa,    "position");
//   auto aosoa_ids = Cabana::slice<1>          ( aosoa,    "ids");
//   auto aosoa_velocity = Cabana::slice<2>     ( aosoa,    "velocity");
//   auto aosoa_acc = Cabana::slice<3>          ( aosoa,    "acc");
//   auto aosoa_mass = Cabana::slice<4>        ( aosoa,    "mass");
//   auto aosoa_density = Cabana::slice<5>     ( aosoa,    "density");
//   auto aosoa_h = Cabana::slice<6>           ( aosoa,    "h");
//   auto aosoa_p = Cabana::slice<7>           ( aosoa,    "p");
//   auto aosoa_is_fluid = Cabana::slice<8>    ( aosoa,    "is_fluid");
//   auto aosoa_is_boundary = Cabana::slice<9>    ( aosoa,    "is_boundary");
//   auto aosoa_density_acc = Cabana::slice<10>           ( aosoa,    "density_acc");
//   auto aosoa_p_acc = Cabana::slice<11>           ( aosoa,    "p_acc");
//   auto aosoa_wij = Cabana::slice<12>           ( aosoa,    "wij");
//   auto aosoa_sin_val = Cabana::slice<13>           ( aosoa,    "sin");
//   auto aosoa_sin_appr = Cabana::slice<14>           ( aosoa,    "sin_appr");
//   // for ( std::size_t i = 0; i < aosoa.size(); ++i )
//   //   {
//   //   }

//   auto setup_aosoa_func = KOKKOS_LAMBDA( const int i )
//     {
//       aosoa_position ( i, 0 ) = x_d(i);
//       aosoa_position ( i, 1 ) = 0.;
//       aosoa_position ( i, 2 ) = 0.;
//       aosoa_ids ( i ) = i;
//       aosoa_mass ( i ) = 1. * spacing;
//       aosoa_density ( i ) = 1.;
//       aosoa_h ( i ) = 1. * spacing;
//       aosoa_sin_val ( i ) = sin_x_d(i);
//       aosoa_sin_appr ( i ) = 0.;
//     };
//   Kokkos::RangePolicy<ExecutionSpace> policy( limits[0], limits[1] );
//   Kokkos::parallel_for( "SetupAosoaFunc", policy,
// 			setup_aosoa_func );



//   // for ( std::size_t i = 0; i < x_d.size(); ++i )
//   //   {
//   //     std::cout << "\n";
//   //     std::cout << "x is " << x_d[i];
//   //   }




//   // compute the neighbor lists
//   // ================================================
//   // ================================================
//   // create the neighbor list
//   // ================================================
//   // ================================================
//   double neighborhood_radius = 3. * spacing;
//   double grid_min[3] = { -M_PI, -neighborhood_radius -  spacing, -neighborhood_radius -  spacing};
//   double grid_max[3] = { 3. * M_PI, neighborhood_radius + spacing, neighborhood_radius + spacing};
//   double cell_ratio = 2.0;
//   // using ListAlgorithm = Cabana::FullNeighborTag;
//   // using ListType =
//   //   Cabana::VerletList<MemorySpace, ListAlgorithm, Cabana::VerletLayoutCSR,
//   // 		       Cabana::TeamOpTag>;
//   ListType verlet_list( aosoa_position, 0, aosoa_position.size(), neighborhood_radius,
// 			cell_ratio, grid_min, grid_max );

//   /*
//     Now lets get the Verlet list data using the neighbor list
//     interface. This data is accessible on any device compatible with the
//     memory space of the neighbor list. Each particle should have 2
//     neighbors.
//   */
//   // for ( std::size_t i = 0; i < aosoa.size(); ++i )
//   //   {
//   //     int no_n =
//   // 	Cabana::NeighborList<ListType>::numNeighbor( verlet_list, i );
//   //     std::cout << "Particle " << i << " # neighbor = " << no_n << std::endl;
//   //     std::cout << "Particle x pos: " << aosoa_position( i, 0 ) << ", " << aosoa_position( i, 1 ) << std::endl;
//   //     for ( int j = 0; j < no_n; ++j )
//   // 	std::cout << "    neighbor " << j << " = "
//   // 		  << Cabana::NeighborList<ListType>::getNeighbor(verlet_list, i, j )
//   // 		  << std::endl;
//   //   }
//   output_data(aosoa, no_particles, 10, 0.);
//   for ( int j = 0; j < 1000; ++j ){

//     approximate_sin(aosoa, &verlet_list, limits);
//   }
//   // save the output in a file
//   output_data(aosoa, no_particles, 0, 0.);
// }

// int main( int argc, char* argv[] )
// {

//   MPI_Init( &argc, &argv );
//   Kokkos::initialize( argc, argv );

//   run();

//   Kokkos::finalize();

//   MPI_Finalize();
//   return 0;
// }

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
// https://cg.informatik.uni-freiburg.de/course_notes/sim_03_particleFluids.pdf : Slide 28

#include <Cabana_Core.hpp>
#include <math.h>

#include <iostream>

/*
  Start by declaring the types the particles will store. The first element
  will represent the coordinates, the second will be the particle's ID, the
  third velocity, and the fourth the radius of the particle.
*/

using DataTypes = Cabana::MemberTypes<double[3], // position(0)
                                      int, // ids(1)
                                      double[3], // velocity(2)
                                      double[3], // acceleration(3)
                                      double, // mass(4)
                                      double, // density(5)
                                      double, // h (smoothing length) (6)
                                      double, // pressure (7)
                                      int, // is_fluid(8)
                                      int, // is_boundary(9)
                                      double, // rate of density change (10)
                                      double, // rate of pressure change (11)
  double, // wij (12)
  double, // sin (13)
  double // sin_appr (14)
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
// auto aosoa_acc = Cabana::slice<3>          ( aosoa,    "aosoa_acc");
// auto aosoa_mass = Cabana::slice<4>        ( aosoa,    "aosoa_mass");
// auto aosoa_density = Cabana::slice<5>     ( aosoa,    "aosoa_density");
// auto aosoa_h = Cabana::slice<6>           ( aosoa,    "aosoa_h");
// auto aosoa_p = Cabana::slice<7>           ( aosoa,    "aosoa_p");
// auto aosoa_is_fluid = Cabana::slice<8>    ( aosoa,    "aosoa_is_fluid");
// auto aosoa_is_boundary = Cabana::slice<9>    ( aosoa,    "aosoa_is_boundary");
// auto aosoa_density_acc = Cabana::slice<10>           ( aosoa,    "aosoa_density_acc");
// auto aosoa_p_acc = Cabana::slice<11>           ( aosoa,    "aosoa_p_acc");
// auto aosoa_wij = Cabana::slice<12>           ( aosoa,    "aosoa_wij");
// auto aosoa_sin = Cabana::slice<13>           ( aosoa,    "aosoa_sin");
// auto aosoa_sin_appr = Cabana::slice<14>           ( aosoa,    "aosoa_sin_appr");

// Neighbour list
// using ListAlgorithm = Cabana::FullNeighborTag;
// using ListType =
// Cabana::VerletList<MemorySpace, ListAlgorithm, Cabana::VerletLayout2D>;
using ListAlgorithm = Cabana::FullNeighborTag;
using ListType =
  Cabana::VerletList<MemorySpace, ListAlgorithm, Cabana::VerletLayoutCSR,
		     Cabana::TeamOpTag>;



void compute_quintic_wij(double rij, double h, double *result){
  double h1 =  1. / h;
  double q =  rij * h1;
  // TODO this changes with the dimension
  double fac = 1.0 / 120.0 * h1;
  // double fac = M_1_PI *  7. / 478. * h1 * h1;

  double tmp3 = 3. - q;
  double tmp2 = 2. - q;
  double tmp1 = 1. - q;

  double val = 0.;
  if (q > 3.) {
    val = 0.;
  } else if ( q > 2.) {
    val = tmp3 * tmp3 * tmp3 * tmp3 * tmp3;
  } else if ( q > 1.) {
    val = tmp3 * tmp3 * tmp3 * tmp3 * tmp3;
    val -= 6.0 * tmp2 * tmp2 * tmp2 * tmp2 * tmp2;
  } else {
    val = tmp3 * tmp3 * tmp3 * tmp3 * tmp3;
    val -= 6.0 * tmp2 * tmp2 * tmp2 * tmp2 * tmp2;
    val += 15. * tmp1 * tmp1 * tmp1 * tmp1 * tmp1;
  }

  *result = val * fac;
}


void compute_quintic_gradient_wij(double xij, double rij, double h, double *result){
  double h1 =  1. / h;
  double q =  rij * h1;

  // TODO this changes with the dimension
  double fac = M_1_PI *  7. / 478. * h1 * h1;

  double tmp3 = 3. - q;
  double tmp2 = 2. - q;
  double tmp1 = 1. - q;

  double val = 0.;
  if (rij > 1e-12){
    if (q > 3.) {
      val = 0.;
    } else if ( q > 2.) {
      val = -5.0 * tmp3 * tmp3 * tmp3 * tmp3;
      val *= h1 / rij;
    } else if ( q > 1.) {
      val = -5.0 * tmp3 * tmp3 * tmp3 * tmp3;
      val += 30.0 * tmp2 * tmp2 * tmp2 * tmp2;
      val *= h1 / rij;
    } else {
      val = -5.0 * tmp3 * tmp3 * tmp3 * tmp3;
      val += 30.0 * tmp2 * tmp2 * tmp2 * tmp2;
      val -= 75.0 * tmp1 * tmp1 * tmp1 * tmp1;
      val *= h1 / rij;
    }
  } else {
    val = 0.;
  }

  double tmp = val * fac;
  // result[0] = tmp * xij[0];
  // result[1] = tmp * xij[1];
  // result[2] = tmp * xij[2];
}

std::vector<double> get_equispaced_points(double start, int end, int no_particles) {
  std::vector<double> x;
  double spacing = (end - start) / no_particles;

  for (int i = 0; i < no_particles; ++i) {
    x.push_back(start + i * spacing);
  }

  return x;
}


std::vector<double> compute_sin(std::vector<double> x) {
  std::vector<double> sin_x;

  for (int i = 0; i < x.size(); ++i) {
    sin_x.push_back(sin(x[i]));
  }

  return sin_x;
}


// auto create_aosoa_particles(int no_particles)
// {
//   // sum all the number of particles and create aosoa
//   AoSoAType aosoa( "particles", no_particles );

//   auto ids = Cabana::slice<0>( aosoa, "ids" );
//   auto m = Cabana::slice<4>( aosoa, "mass" );
//   for ( std::size_t i = 0; i < aosoa.size(); ++i )
//     {
//       ids( i ) = i;
//       m( i ) = m_array[i];
//     }
//   return aosoa;
// }

void approximate_sin(AoSoAType aosoa,
		     ListType * verlet_list_source,
		     int * limits){
  auto position = Cabana::slice<0>     ( aosoa,    "position");
auto mass = Cabana::slice<4>        ( aosoa,    "mass");
auto density = Cabana::slice<5>     ( aosoa,    "density");
auto h = Cabana::slice<6>           ( aosoa,    "h");
auto aosoa_wij = Cabana::slice<12>           ( aosoa,    "wij");
auto sin_val = Cabana::slice<13>           ( aosoa,    "sin");
auto sin_appr = Cabana::slice<14>           ( aosoa,    "sin_appr");

auto sin_lambda_func = KOKKOS_LAMBDA( const int i, const int j )
    {
      // Function approximation starts here
      const double h_i = h( i );

      double wij = 0.;
      const double pos_i[3] = {position( i, 0 ),
	position( i, 1 ),
	position( i, 2 )};

      const double pos_j[3] = {position( j, 0 ),
	position( j, 1 ),
	position( j, 2 )};

      const double pos_ij[3] = {position( i, 0 ) - position( j, 0 ),
	position( i, 1 ) - position( j, 1 ),
	position( i, 2 ) - position( j, 2 )};

      const double r2ij = pos_ij[0] * pos_ij[0] + pos_ij[1] * pos_ij[1] + pos_ij[2] * pos_ij[2];
      const double rij = sqrt(r2ij);

      // compute WIJ
      compute_quintic_wij(rij, h_i, &wij);

      // Function approximation starts here
      const double m_j = mass( j );
      const double rho_j = density( j );

      sin_appr( i ) += m_j / rho_j * sin_val( j ) * wij;
      // if (i == 0){
      // std::cout << "inside" << "\n";
      // 	std::cout << "j is " << j << "\n";
	// }
    };

  Kokkos::RangePolicy<ExecutionSpace> policy(limits[0], limits[1]);


  Cabana::neighbor_parallel_for( policy,
                                 sin_lambda_func,
                                 *verlet_list_source,
                                 Cabana::FirstNeighborsTag(),
                                 Cabana::SerialOpTag(),
                                 "SinAppr" );
  Kokkos::parallel_for("SinAppr0", policy, KOKKOS_LAMBDA(int i) { sin_lambda_func(i, i); });
  Kokkos::fence();
}


void output_data(auto aosoa, int no_particles, int step, double time)
{
  auto position = Cabana::slice<0>     ( aosoa,    "position");
  auto ids = Cabana::slice<1>          ( aosoa,    "ids");
  auto velocity = Cabana::slice<2>     ( aosoa,    "velocity");
  auto acc = Cabana::slice<3>          ( aosoa,    "acc");
  auto mass = Cabana::slice<4>        ( aosoa,    "mass");
  auto density = Cabana::slice<5>     ( aosoa,    "density");
  auto h = Cabana::slice<6>           ( aosoa,    "h");
  auto p = Cabana::slice<7>           ( aosoa,    "p");
  auto is_fluid = Cabana::slice<8>    ( aosoa,    "is_fluid");
  auto is_boundary = Cabana::slice<9>    ( aosoa,    "is_boundary");
  auto density_acc = Cabana::slice<10>           ( aosoa,    "density_acc");
  auto p_acc = Cabana::slice<11>           ( aosoa,    "p_acc");
  auto wij = Cabana::slice<12>           ( aosoa,    "wij");
  auto sin_val = Cabana::slice<13>           ( aosoa,    "sin");
  auto sin_appr = Cabana::slice<14>           ( aosoa,    "sin_appr");

  Cabana::Experimental::HDF5ParticleOutput::HDF5Config h5_config;
  Cabana::Experimental::HDF5ParticleOutput::
    writeTimeStep(
                  h5_config, "particles", MPI_COMM_WORLD,
                  step, time, no_particles, position,
                  ids, sin_val, sin_appr);
}



//---------------------------------------------------------------------------//
// TODO: explain this function in short
//---------------------------------------------------------------------------//
void run()
{
  int comm_rank = -1;
  MPI_Comm_rank( MPI_COMM_WORLD, &comm_rank );

  if ( comm_rank == 0 )
    std::cout << "Cabana SPH solver sin example\n" << std::endl;

  // create points on a 1d grid from 0 to 2 * pi
  int no_particles = 10000;
  int limits[2] = {0, no_particles};
  auto length = 2. * M_PI;
  auto spacing = length / no_particles;
  std::vector<double> x = get_equispaced_points(0., length, no_particles);
  std::vector<double> sin_x = compute_sin(x);

  // for ( std::size_t i = 0; i < x.size(); ++i )
  //   {
  //     std::cout << "\n";
  //     std::cout << "x is " << x[i] << " and sin is " << sin_x[i];
  //   }

  AoSoAType aosoa( "particles", no_particles );

  auto aosoa_position = Cabana::slice<0>     ( aosoa,    "position");
  auto aosoa_ids = Cabana::slice<1>          ( aosoa,    "ids");
  auto aosoa_velocity = Cabana::slice<2>     ( aosoa,    "velocity");
  auto aosoa_acc = Cabana::slice<3>          ( aosoa,    "acc");
  auto aosoa_mass = Cabana::slice<4>        ( aosoa,    "mass");
  auto aosoa_density = Cabana::slice<5>     ( aosoa,    "density");
  auto aosoa_h = Cabana::slice<6>           ( aosoa,    "h");
  auto aosoa_p = Cabana::slice<7>           ( aosoa,    "p");
  auto aosoa_is_fluid = Cabana::slice<8>    ( aosoa,    "is_fluid");
  auto aosoa_is_boundary = Cabana::slice<9>    ( aosoa,    "is_boundary");
  auto aosoa_density_acc = Cabana::slice<10>           ( aosoa,    "density_acc");
  auto aosoa_p_acc = Cabana::slice<11>           ( aosoa,    "p_acc");
  auto aosoa_wij = Cabana::slice<12>           ( aosoa,    "wij");
  auto aosoa_sin_val = Cabana::slice<13>           ( aosoa,    "sin");
  auto aosoa_sin_appr = Cabana::slice<14>           ( aosoa,    "sin_appr");
  for ( std::size_t i = 0; i < aosoa.size(); ++i )
    {
      aosoa_position ( i, 0 ) = x[i];
      aosoa_position ( i, 1 ) = 0.;
      aosoa_position ( i, 2 ) = 0.;
      aosoa_ids ( i ) = i;
      aosoa_mass ( i ) = 1. * spacing;
      aosoa_density ( i ) = 1.;
      aosoa_h ( i ) = 1. * spacing;
      aosoa_sin_val ( i ) = sin_x[i];
      aosoa_sin_appr ( i ) = 0.;
    }

  // compute the neighbor lists
  // ================================================
  // ================================================
  // create the neighbor list
  // ================================================
  // ================================================
  double neighborhood_radius = 3. * spacing;
  double grid_min[3] = { -M_PI, -neighborhood_radius -  spacing, -neighborhood_radius -  spacing};
  double grid_max[3] = { 3. * M_PI, neighborhood_radius + spacing, neighborhood_radius + spacing};
  double cell_ratio = 2.0;
  // using ListAlgorithm = Cabana::FullNeighborTag;
  // using ListType =
  //   Cabana::VerletList<MemorySpace, ListAlgorithm, Cabana::VerletLayoutCSR,
  // 		       Cabana::TeamOpTag>;
  ListType verlet_list( aosoa_position, 0, aosoa_position.size(), neighborhood_radius,
			cell_ratio, grid_min, grid_max );

  /*
    Now lets get the Verlet list data using the neighbor list
    interface. This data is accessible on any device compatible with the
    memory space of the neighbor list. Each particle should have 2
    neighbors.
  */
  for ( std::size_t i = 0; i < aosoa.size(); ++i )
    {
      int no_n =
	Cabana::NeighborList<ListType>::numNeighbor( verlet_list, i );
      std::cout << "Particle " << i << " # neighbor = " << no_n << std::endl;
      std::cout << "Particle x pos: " << aosoa_position( i, 0 ) << ", " << aosoa_position( i, 1 ) << std::endl;
      for ( int j = 0; j < no_n; ++j )
	std::cout << "    neighbor " << j << " = "
		  << Cabana::NeighborList<ListType>::getNeighbor(verlet_list, i, j )
		  << std::endl;
    }

  output_data(aosoa, no_particles, 0, 0.);
  approximate_sin(aosoa, &verlet_list, limits);

  // save the output in a file
  // output_data(aosoa, no_particles, 0, 0.);
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
