/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include "helper_functions.h"

#include <algorithm>
#include <iostream>
#include <iterator>
#include <math.h>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using std::string;
using std::vector;
using std::normal_distribution;

void ParticleFilter::init(double x, double y, double theta, Sigmas sigmas)
{
  /**
   * TODO: Set the number of particles. Initialize all particles to
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1.
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method
   *   (and others in this file).
   */
  particles_.reserve(num_particles_);
  weights_.reserve(num_particles_);

  normal_distribution<double> dist_x(x, sigmas.x);
  normal_distribution<double> dist_y(y, sigmas.y);
  normal_distribution<double> dist_theta(theta, sigmas.theta);

  for (size_t i{0}; i < num_particles_; ++i)
  {
    particles_.emplace_back(
        Particle{int(i), dist_x(rnd_), dist_y(rnd_), dist_theta(rnd_), 1.0, {}, {}, {}});
    weights_.emplace_back(1.0);
  }

  is_initialized_ = true;
}

void ParticleFilter::init(int particle_count, double x, double y, double theta, Sigmas sigmas)
{
  num_particles_ = size_t(particle_count);
  init(x, y, theta, sigmas);
}

void ParticleFilter::predict(double delta_t, Sigmas sigmas, double velocity, double yaw_rate)
{
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::default_random_engine gen;
  for (auto &particle : particles_)
  {
    const auto d_thetha{delta_t * yaw_rate};
    const auto old_thetha{particle.theta};
    const auto new_thetha{old_thetha + d_thetha};
    auto       new_x{particle.x};
    auto       new_y{particle.y};

    const auto cos_new_thetha{cos(new_thetha)};
    const auto sin_new_thetha{cos(new_thetha)};

    if (fabs(yaw_rate) <= 0.01)
    {
      // If yaw rate is negligible, x and y are updated according to the traveled distance.
      const auto traveled_distance{velocity * delta_t};
      new_x += traveled_distance * cos_new_thetha;
      new_y += traveled_distance * sin_new_thetha;
    }
    else
    {
      const double rotation{velocity * yaw_rate};
      new_x += rotation * (sin_new_thetha - sin(old_thetha));
      new_y += rotation * (cos(old_thetha) - cos_new_thetha);
    }

    normal_distribution<double> dist_x(new_x, sigmas.x);
    normal_distribution<double> dist_y(new_y, sigmas.y);
    normal_distribution<double> dist_theta(new_thetha, sigmas.theta);

    particle.x     = dist_x(rnd_);
    particle.y     = dist_y(rnd_);
    particle.theta = dist_theta(rnd_);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs>  predicted,
                                     vector<LandmarkObs> &observations)
{
  /**
   * TODO: Find the predicted measurement that is closest to each
   *   observed measurement and assign the observed measurement to this
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will
   *   probably find it useful to implement this method and use it as a helper
   *   during the updateWeights phase.
   */
}

void ParticleFilter::updateWeights(double sensor_range, Sigmas std_landmark,
                                   const vector<LandmarkObs> &observations,
                                   const Map &                map_landmarks)
{
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian
   *   distribution. You can read more about this distribution here:
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system.
   *   Your particles are located according to the MAP'S coordinate system.
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
}

void ParticleFilter::resample()
{
  /**
   * TODO: Resample particles with replacement with probability proportional
   *   to their weight.
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
}

void ParticleFilter::SetAssociations(Particle &particle, const vector<int> &associations,
                                     const vector<double> &sense_x, const vector<double> &sense_y)
{
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations = associations;
  particle.sense_x      = sense_x;
  particle.sense_y      = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
  vector<int>       v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s        = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord)
{
  vector<double> v;

  if (coord == "X")
  {
    v = best.sense_x;
  }
  else
  {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s        = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}

const std::vector<Particle> &ParticleFilter::particles()
{
  return particles_;
}
