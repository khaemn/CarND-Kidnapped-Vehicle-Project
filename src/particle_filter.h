/**
 * particle_filter.h
 * 2D particle filter class.
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#ifndef PARTICLE_FILTER_H_
#define PARTICLE_FILTER_H_

#include "helper_functions.h"

#include <string>
#include <vector>
#include <random>

struct Sigmas
{
  const double x;
  const double y;
  const double theta;
};

struct Particle
{
  int                 id;
  double              x;
  double              y;
  double              theta;
  double              weight;
  std::vector<int>    associations;
  std::vector<double> sense_x;
  std::vector<double> sense_y;

  std::string coordsToString() const
  {
    using std::to_string;
    return {"[ " + to_string(x) + " ; " + to_string(y) + " ](" + to_string(id) + ")"};
  }
};

class ParticleFilter
{
public:
  ParticleFilter()
    : total_particles_(0)
    , is_initialized_(false)
  {}

  ~ParticleFilter() = default;

  /**
   * init Initializes particle filter by initializing particles to Gaussian
   *   distribution around first position and all the weights to 1.
   * @param x Initial x position [m] (simulated estimate from GPS)
   * @param y Initial y position [m]
   * @param theta Initial orientation [rad]
   * @param std[] Array of dimension 3 [standard deviation of x [m],
   *   standard deviation of y [m], standard deviation of yaw [rad]]
   */
  void init(double x, double y, double theta, Sigmas sigmas);
  void init(int particle_count, double x, double y, double theta, Sigmas sigmas);

  /**
   * prediction Predicts the state for the next time step
   *   using the process model.
   * @param delta_t Time between time step t and t+1 in measurements [s]
   * @param std_pos[] Array of dimension 3 [standard deviation of x [m],
   *   standard deviation of y [m], standard deviation of yaw [rad]]
   * @param velocity Velocity of car from t to t+1 [m/s]
   * @param yaw_rate Yaw rate of car from t to t+1 [rad/s]
   */
  void predict(double delta_t, Sigmas sigmas, double velocity, double yaw_rate);

  /**
   * dataAssociation Finds which observations correspond to which landmarks
   *   (likely by using a nearest-neighbors data association).
   * @param predicted Vector of predicted landmark observations
   * @param observations Vector of landmark observations
   */
  void dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations);

  /**
   * updateWeights Updates the weights for each particle based on the likelihood
   *   of the observed measurements.
   * @param sensor_range Range [m] of sensor
   * @param std_landmark[] Array of dimension 2
   *   [Landmark measurement uncertainty [x [m], y [m]]]
   * @param observations Vector of landmark observations
   * @param map Map class containing map landmarks
   */
  void updateWeights(double sensor_range, Sigmas std_landmark,
                     const std::vector<LandmarkObs> &observations, const Map &map_landmarks);

  /**
   * resample Resamples from the updated set of particles to form
   *   the new set of particles.
   */
  void resample();

  /**
   * Set a particles list of associations, along with the associations'
   *   calculated world x,y coordinates
   * This can be a very useful debugging tool to make sure transformations
   *   are correct and assocations correctly connected
   */
  void SetAssociations(Particle &particle, const std::vector<int> &associations,
                       const std::vector<double> &sense_x, const std::vector<double> &sense_y);

  /**
   * initialized Returns whether particle filter is initialized yet or not.
   */
  const bool initialized() const
  {
    return is_initialized_;
  }

  /**
   * Used for obtaining debugging information related to particles.
   */
  std::string getAssociations(Particle best);
  std::string getSenseCoord(Particle best, std::string coord);

  // Set of current particles
  const std::vector<Particle> &particles();

private:
  std::vector<Particle> particles_;

  // Number of particles to draw
  size_t total_particles_;

  // Flag, if filter is initialized
  bool is_initialized_;

  // Vector of weights of all particles
  std::vector<double> weights_;

  // Member random generator
  std::default_random_engine rnd_;
};

#endif  // PARTICLE_FILTER_H_
