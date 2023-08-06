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

// auto aosoa_ids = Cabana::slice<1>          ( aosoa,    "aosoa_ids");
// auto aosoa_position = Cabana::slice<0>     ( aosoa,    "aosoa_position");
// auto aosoa_velocity = Cabana::slice<2>     ( aosoa,    "aosoa_velocity");
// auto aosoa_acc = Cabana::slice<3>          ( aosoa,    "aosoa_acc");
// auto aosoa_mass = Cabana::slice<4>        ( aosoa,    "aosoa_mass");
// auto aosoa_density = Cabana::slice<5>     ( aosoa,    "aosoa_density");
// auto aosoa_h = Cabana::slice<6>           ( aosoa,    "aosoa_h");
// auto aosoa_p = Cabana::slice<7>           ( aosoa,    "aosoa_p");
// auto aosoa_is_fluid = Cabana::slice<8>    ( aosoa,    "aosoa_is_fluid");
// auto aosoa_is_boundary = Cabana::slice<9>    ( aosoa,    "aosoa_is_boundary");
// auto aosoa_density_acc = Cabana::slice10>           ( aosoa,    "aosoa_density_acc");
// auto aosoa_p_acc = Cabana::slice<11>           ( aosoa,    "aosoa_p_acc");
// auto aosoa_wij = Cabana::slice<12>           ( aosoa,    "aosoa_wij");

void fluid_stage_1(auto rb, double dt, auto limits){
  auto aosoa_velocity = Cabana::slice<2>     ( aosoa,    "aosoa_velocity");
  auto aosoa_acc = Cabana::slice<3>          ( aosoa,    "aosoa_acc");
  auto aosoa_mass = Cabana::slice<4>        ( aosoa,    "aosoa_mass");

  auto half_dt = dt * 0.5;
  auto fluid_stage_1_lambda_func = KOKKOS_LAMBDA( const int i )
    {
      aosoa_velocity(i, 0) += aosoa_acc( i, 0 ) * half_dt;
      aosoa_velocity(i, 1) += aosoa_acc( i, 1 ) * half_dt;
      aosoa_velocity(i, 2) += aosoa_acc( i, 2 ) * half_dt;
    };
  Kokkos::RangePolicy<ExecutionSpace> policy( limits[0], limits[1] );
  Kokkos::parallel_for( "CabanaSPH:Integrator:FluidStage1", policy,
                        fluid_stage_1_lambda_func );
}


void fluid_stage_2(auto rb, double dt, auto limits){
  auto aosoa_position = Cabana::slice<0>     ( aosoa,    "aosoa_position");
  auto aosoa_velocity = Cabana::slice<2>     ( aosoa,    "aosoa_velocity");
  auto aosoa_density = Cabana::slice<5>     ( aosoa,    "aosoa_density");
  auto aosoa_density_acc = Cabana::slice10>           ( aosoa,    "aosoa_density_acc");

  auto half_dt = dt * 0.5;
  auto fluid_stage_2_lambda_func = KOKKOS_LAMBDA( const int i )
    {
      aosoa_density( i ) += aosoa_density_acc( i ) * dt;

      aosoa_position(i, 0) += aosoa_velocity( i, 0 ) * dt;
      aosoa_position(i, 1) += aosoa_velocity( i, 1 ) * dt;
      aosoa_position(i, 2) += aosoa_velocity( i, 2 ) * dt;
    };
  Kokkos::RangePolicy<ExecutionSpace> policy( limits[0], limits[1] );
  Kokkos::parallel_for( "CabanaSPH:Integrator:FluidStage2", policy,
                        fluid_stage_2_lambda_func );
}


void fluid_stage_3(auto rb, double dt, auto limits){
  auto aosoa_velocity = Cabana::slice<2>     ( aosoa,    "aosoa_velocity");
  auto aosoa_acc = Cabana::slice<3>          ( aosoa,    "aosoa_acc");
  auto aosoa_mass = Cabana::slice<4>        ( aosoa,    "aosoa_mass");

  auto half_dt = dt * 0.5;
  auto fluid_stage_1_lambda_func = KOKKOS_LAMBDA( const int i )
    {
      aosoa_velocity(i, 0) += aosoa_acc( i, 0 ) * half_dt;
      aosoa_velocity(i, 1) += aosoa_acc( i, 1 ) * half_dt;
      aosoa_velocity(i, 2) += aosoa_acc( i, 2 ) * half_dt;
    };
  Kokkos::RangePolicy<ExecutionSpace> policy( limits[0], limits[1] );
  Kokkos::parallel_for( "CabanaSPH:Integrator:FluidStage3", policy,
                        fluid_stage_3_lambda_func );
}


void continuity_equation(auto aosoa, auto dt,
                         ListType * verlet_list_source,
                         auto limits){
  auto aosoa_ids = Cabana::slice<1>          ( aosoa,    "aosoa_ids");
  auto aosoa_position = Cabana::slice<0>     ( aosoa,    "aosoa_position");
  auto aosoa_velocity = Cabana::slice<2>     ( aosoa,    "aosoa_velocity");
  auto aosoa_mass = Cabana::slice<4>        ( aosoa,    "aosoa_mass");
  auto aosoa_density = Cabana::slice<5>     ( aosoa,    "aosoa_density");
  auto aosoa_density_acc = Cabana::slice10>           ( aosoa,    "aosoa_density_acc");

  Cabana::deep_copy( aosoa_density_acc, 0. );

  auto continuity_lambda_func = KOKKOS_LAMBDA( const int i, const int j )
    {
      const double mass_i = aosoa_mass( i );
      const double mass_j = aosoa_mass( j );

      const double[3] pos_i = {aosoa_position( i, 0 ),
                               aosoa_position( i, 1 ),
                               aosoa_position( i, 2 )};

      const double[3] pos_j = {aosoa_position( j, 0 ),
                               aosoa_position( j, 1 ),
                               aosoa_position( j, 2 )};

      const double[3] pos_ij = {aosoa_position( i, 0 ) - aosoa_position( j, 0 ),
                                aosoa_position( i, 1 ) - aosoa_position( j, 1 ),
                                aosoa_position( i, 2 ) - aosoa_position( j, 2 )}

      const double[3] vel_ij = {aosoa_velocity( i, 0 ) - aosoa_velocity( j, 0 ),
                                aosoa_velocity( i, 1 ) - aosoa_velocity( j, 1 ),
                                aosoa_velocity( i, 2 ) - aosoa_velocity( j, 2 )}

      const double[3] dwij = {0., 0., 0.};

      double vijdotdwij = dwij[0]*vel_ij[0] + dwij[1]*vel_ij[1] + dwij[2]*vel_ij[2];
      aosoa_density_acc (i) += m_j * vijdotdwij;
    };

  Kokkos::RangePolicy<ExecutionSpace> policy(index_limits[0], index_limits[1]);


  Cabana::neighbor_parallel_for( policy,
                                 continuity_lambda_func,
                                 *verlet_list_source,
                                 Cabana::FirstNeighborsTag(),
                                 Cabana::SerialOpTag(),
                                 "CabanaSPH:Equations:Continuity" );
  Kokkos::fence();
}


void state_equation(auto aosoa, auto dt, auto limits, auto b_rho0_p0){
  auto aosoa_density = Cabana::slice<5>     ( aosoa,    "aosoa_density");
  auto aosoa_p = Cabana::slice<7>           ( aosoa,    "aosoa_p");

  auto state_eq_lambda_func = KOKKOS_LAMBDA( const int i )
    {
      d_p[d_idx] = self.p0 * (d_rho[d_idx] / self.rho0 - self.b) + self.p0

      double tmp = b_rho_p0[2] * (aosoa_density( i ) / b_rho0_p0[1] - b_rho0_p0[0]);
      aosoa_p( i ) = tmp + b_rho_p0[2];
    };
  Kokkos::RangePolicy<ExecutionSpace> policy( limits[0], limits[1] );
  Kokkos::parallel_for( "CabanaSPH:Equations:StatePressure", policy,
                        state_eq_lambda_func );
}


void momentum_equation(auto aosoa, auto dt,
                       ListType * verlet_list_source,
                       auto limits, auto gravity){
  auto aosoa_acc = Cabana::slice<3>          ( aosoa,    "aosoa_acc");
  auto aosoa_mass = Cabana::slice<4>        ( aosoa,    "aosoa_mass");
  auto aosoa_density = Cabana::slice<5>     ( aosoa,    "aosoa_density");
  auto aosoa_h = Cabana::slice<6>           ( aosoa,    "aosoa_h");
  auto aosoa_p = Cabana::slice<7>           ( aosoa,    "aosoa_p");

  Cabana::deep_copy( aosoa_acceleration, 0. );

  auto momentum_lambda_func = KOKKOS_LAMBDA( const int i, const int j )
    {
      const double[3] dwij = {0., 0., 0.};

      const double rhoi2 = 1. / (aosoa_density( i ) * aosoa_density( i ));
      const double rhoj2 = 1. / (aosoa_density( j ) * aosoa_density( j ));
      double pij = aosoa_p ( i ) / rhoi2 + aosoa_p ( j ) / rhoj2;
      double tmp = - aosoa_mass ( i ) * pij;

      // pressure acceleration
      aosoa_acc ( i, 0 ) += tmp * dwij[0];
      aosoa_acc ( i, 1 ) += tmp * dwij[1];
      aosoa_acc ( i, 2 ) += tmp * dwij[2];

      // artificial viscosity
      const double mass_i = aosoa_mass( i );
      const double mass_j = aosoa_mass( j );
      const double h_i = aosoa_h( i );
      const double h_j = aosoa_h( j );

      const double[3] pos_i = {aosoa_position( i, 0 ),
                               aosoa_position( i, 1 ),
                               aosoa_position( i, 2 )};

      const double[3] pos_j = {aosoa_position( j, 0 ),
                               aosoa_position( j, 1 ),
                               aosoa_position( j, 2 )};

      const double[3] pos_ij = {aosoa_position( i, 0 ) - aosoa_position( j, 0 ),
                                aosoa_position( i, 1 ) - aosoa_position( j, 1 ),
                                aosoa_position( i, 2 ) - aosoa_position( j, 2 )}

      const double[3] vel_ij = {aosoa_velocity( i, 0 ) - aosoa_velocity( j, 0 ),
                                aosoa_velocity( i, 1 ) - aosoa_velocity( j, 1 ),
                                aosoa_velocity( i, 2 ) - aosoa_velocity( j, 2 )}

      const double[3] dwij = {0., 0., 0.};

      double vijdotrij = pos_ij[0]*vel_ij[0] + pos_ij[1]*vel_ij[1] + pos_ij[2]*vel_ij[2];

      // TODO these are not correct, fix it
      double pij = 0.0;
      double hij = (h_i + h_j) / 2.;
      double r2ij = 0.;
      double rhoij1 = 1. / (rho_i + rho_j);
      double piij = 0.;
      if (vijdotrij < 0.){
        double muij = ( hij * vijdotrij ) / (r2ij + EPS);

        piij = -alpha * c0 * muij;
        piij = m_j * piij * rhoij1;
      }
      aosoa_acc ( i, 0 ) += -piij * dwij[0];
      aosoa_acc ( i, 1 ) += -piij * dwij[1];
      aosoa_acc ( i, 2 ) += -piij * dwij[2];

    };

  Kokkos::RangePolicy<ExecutionSpace> policy(index_limits[0], index_limits[1]);


  Cabana::neighbor_parallel_for( policy,
                                 momentum_lambda_func,
                                 *verlet_list_source,
                                 Cabana::FirstNeighborsTag(),
                                 Cabana::SerialOpTag(),
                                 "CabanaSPH:Equations:Momentum" );
  Kokkos::fence();
}


void solid_wall_pressure_bc(auto aosoa, auto dt,
                            ListType * verlet_list_source,
                            auto limits){
  auto aosoa_acc = Cabana::slice<3>          ( aosoa,    "aosoa_acc");
  auto aosoa_mass = Cabana::slice<4>        ( aosoa,    "aosoa_mass");
  auto aosoa_density = Cabana::slice<5>     ( aosoa,    "aosoa_density");
  auto aosoa_h = Cabana::slice<6>           ( aosoa,    "aosoa_h");
  auto aosoa_p = Cabana::slice<7>           ( aosoa,    "aosoa_p");
  auto aosoa_wij = Cabana::slice<12>           ( aosoa,    "aosoa_wij");

  Cabana::deep_copy( aosoa_p, 0. );
  Cabana::deep_copy( aosoa_wij, 0. );

  auto pressure_bc_lambda_func = KOKKOS_LAMBDA( const int i, const int j )
    {
      // TODO: Add condition (do this only if j is fluid)
      const double[3] dwij = {0., 0., 0.};
      const double wij = 0.;

      double tmp1 = (gravity[0] - aosoa_acc( i, 0 )) * pos_ij [ 0 ];
      double tmp2 = (gravity[1] - aosoa_acc( i, 1 )) * pos_ij [ 1 ];
      double tmp3 = (gravity[2] - aosoa_acc( i, 2 )) * pos_ij [ 2 ];
      double gdotxij = tmp1 +  tmp2 + tmp3;

      // pressure acceleration
      aosoa_p ( i ) += p_j * wij + rho_j * gdotxij * wij;

      // sum the wij
      aosoa_wij ( i ) += wij;
    };

  Kokkos::RangePolicy<ExecutionSpace> policy(index_limits[0], index_limits[1]);


  Cabana::neighbor_parallel_for( policy,
                                 pressure_bc_lambda_func,
                                 *verlet_list_source,
                                 Cabana::FirstNeighborsTag(),
                                 Cabana::SerialOpTag(),
                                 "CabanaSPH:Equations:PressureBC" );
  Kokkos::fence();

  // Divide by wij as the end
  auto pressure_bc_divide_wij_lambda = KOKKOS_LAMBDA( const int i )
    {
      if (aosoa_wij( i ) > 1e-12){
        aosoa_p( i ) /= aosoa_wij( i );
      }
    };
  Kokkos::RangePolicy<ExecutionSpace> policy( limits[0], limits[1] );
  Kokkos::parallel_for( "CabanaSPH:Integrator:PressureBCWij", policy,
                        pressure_bc_divide_wij_lambda);
}


void output_data(auto aosoa, int num_particles, int step, double time)
{
  // This is for setting HDF5 options
  auto ids = Cabana::slice<1>( aosoa, "ids" );
  auto mass = Cabana::slice<4>        ( aosoa,    "mass");
  auto position = Cabana::slice<0>( aosoa, "position" );
  auto velocity = Cabana::slice<2>( aosoa, "velocity" );
  auto is_fluid = Cabana::slice<8>    ( aosoa, "is_fluid");
  auto is_boundary = Cabana::slice<9>    ( aosoa, "is_boundary");
  auto is_rb = Cabana::slice<10>       ( aosoa, "is_rb");
  auto radius = Cabana::slice<11>( aosoa, "radius" );
  auto body_ids = Cabana::slice<12>( aosoa, "body_id" );
  auto frc_dem = Cabana::slice<14>( aosoa, "frc_dem");

  Cabana::Experimental::HDF5ParticleOutput::HDF5Config h5_config;
  Cabana::Experimental::HDF5ParticleOutput::
    writeTimeStep(
                  h5_config, "particles", MPI_COMM_WORLD,
                  step, time, num_particles, position,
                  ids, velocity, radius, body_ids, mass, frc_dem, is_fluid,
                  is_boundary, is_rb);
}


//---------------------------------------------------------------------------//
// TODO: explain this function in short
//---------------------------------------------------------------------------//
void run()
{
  int comm_rank = -1;
  MPI_Comm_rank( MPI_COMM_WORLD, &comm_rank );

  if ( comm_rank == 0 )
    std::cout << "Cabana Rigid body solver example\n" << std::endl;

  auto num_particles = 16;
  auto aosoa = create_aosoa_particles(num_particles);
  /*
    Create the rigid body data type.
  */
  int num_bodies = 2;
  std::vector<int> rigid_limits = {8, 16};
  RBAoSoA rb( "rb", num_bodies );
  // get the rigid body limits
  {
  auto limits = Cabana::slice<1>( rb, "total_no_bodies" );
  // limits of the first body
  limits(0, 0) = 8;
  limits(0, 1) = 12;

  limits(1, 0) = 12;
  limits(1, 1) = 16;

  setup_rigid_body_properties(aosoa, rb, rigid_limits);
  }

  std::vector<double> linear_velocity_0 = {0., 1., 0.};
  set_linear_velocity_rigid_body(aosoa, rb, 0, linear_velocity_0);
  std::vector<double> linear_velocity_1 = {0., -1., 0.};
  set_linear_velocity_rigid_body(aosoa, rb, 1, linear_velocity_1);

  // ================================================
  // ================================================
  // create the neighbor list
  // ================================================
  // ================================================
  double grid_min[3] = { 0.0, -2.0, -1.0 };
  double grid_max[3] = { 25.0, 6.0, 1.0 };
  double neighborhood_radius = 2.0;
  double cell_ratio = 1.0;

  // ListType verlet_list( position_slice, 0, position_slice.size(), neighborhood_radius,
  //            cell_ratio, grid_min, grid_max );

  auto dt = 1e-4;
  auto final_time = 1.;
  auto time = 0.;
  int steps = final_time / dt;
  int print_freq = 100;

  // compute the neighbor lists
  auto aosoa_position = Cabana::slice<0>( aosoa,    "aosoa_position");
  auto aosoa_frc_dem = Cabana::slice<14>     ( aosoa,    "aosoa_frc_dem");
  // ListType verlet_list( aosoa_position, 0,
  //           16, neighborhood_radius,
  //           cell_ratio, grid_min, grid_max );
  output_data(aosoa, num_particles, 0, 0.);


  // Main timestep loop
  for ( int step = 0; step < steps; step++ )
    {
      rigid_body_gtvf_stage_1(rb, dt);
      rigid_body_particles_gtvf_stage_1(aosoa, rb, dt, rigid_limits);

      rigid_body_gtvf_stage_2(rb, dt);
      rigid_body_particles_gtvf_stage_2(aosoa, rb, dt, rigid_limits);

      ListType verlet_list( aosoa_position, 0,
                            aosoa_position.size(), neighborhood_radius,
                            cell_ratio, grid_min, grid_max );
      compute_force_on_rigid_body_particles(aosoa, dt, &verlet_list, rigid_limits);
      compute_effective_force_and_torque_on_rigid_body(rb, aosoa);

            std::cout << "time is: " << time << "\n";

      // std::cout << "\n";
      // std::cout << "\n";
      // std::cout << "\n";
      // for ( std::size_t i = 0; i < aosoa_position.size(); ++i )
      //   {
      //     std::cout << "\n";
      //     std::cout << "\n";
      //     std::cout << "rb force on " << i << " is " << aosoa_frc_dem( i, 0) << ", " << aosoa_frc_dem( i, 1 ) << ", " << aosoa_frc_dem( i, 2 ) << "\n";
      //   }

      rigid_body_gtvf_stage_3(rb, dt);
      rigid_body_particles_gtvf_stage_3(aosoa, rb, dt, rigid_limits);

      // // initialize the rigid body properties
      // auto rb_position = Cabana::slice<2>     ( rb,    "rb_position");
      // auto rb_velocity = Cabana::slice<3>     ( rb,    "rb_velocity");

      // // A configuration object is necessary for tuning HDF5 options.
      // Cabana::Experimental::HDF5ParticleOutput::HDF5Config h5_config;

      // // For example, MPI I/O is independent by default, but collective operations
      // // can be enabled by setting:
      // h5_config.collective = true;
      if ( step % print_freq == 0 )
        {
       output_data(aosoa, num_particles, step, time);
        }

      time += dt;

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
