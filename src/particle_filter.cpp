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
using std::endl;
using std::cout;

void ParticleFilter::init(double x, double y, double theta, Sigmas sigmas)
{
  particles_.reserve(total_particles_);
  weights_.reserve(total_particles_);

  normal_distribution<double> dist_x(x, sigmas.x);
  normal_distribution<double> dist_y(y, sigmas.y);
  normal_distribution<double> dist_theta(theta, sigmas.theta);

  for (size_t i{0}; i < total_particles_; ++i)
  {
    particles_.emplace_back(
        Particle{int(i), dist_x(rnd_), dist_y(rnd_), dist_theta(rnd_), 1.0, {}, {}, {}});
    weights_.emplace_back(1.0);
  }

  is_initialized_ = true;
}

void ParticleFilter::init(int particle_count, double x, double y, double theta, Sigmas sigmas)
{
  total_particles_ = size_t(particle_count);
  init(x, y, theta, sigmas);
}

void ParticleFilter::predict(double delta_t, Sigmas sigmas, double velocity, double yaw_rate)
{
  std::default_random_engine gen;
  for (auto &particle : particles_)
  {
    const auto d_theta{delta_t * yaw_rate};
    const auto old_theta{particle.theta};
    const auto new_theta{old_theta + d_theta};
    auto       new_x{particle.x};
    auto       new_y{particle.y};

    const auto cos_new_thetha{cos(new_theta)};
    const auto sin_new_thetha{sin(new_theta)};

    if (fabs(yaw_rate) <= 0.01)
    {
      // If yaw rate is negligible, x and y are updated according to the traveled distance.
      const auto traveled_distance{velocity * delta_t};
      new_x += traveled_distance * cos_new_thetha;
      new_y += traveled_distance * sin_new_thetha;
    }
    else
    {
      // Otherwise we use a bicycle movement model to predict next position.
      const auto curve_coeff{velocity / yaw_rate};
      new_x += curve_coeff * (sin_new_thetha - sin(old_theta));
      new_y += curve_coeff * (cos(old_theta) - cos_new_thetha);
    }

    normal_distribution<double> dist_x(new_x, sigmas.x);
    normal_distribution<double> dist_y(new_y, sigmas.y);
    normal_distribution<double> dist_theta(new_theta, sigmas.theta);

    particle.x     = dist_x(rnd_);
    particle.y     = dist_y(rnd_);
    particle.theta = dist_theta(rnd_);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs>  predicted,
                                     vector<LandmarkObs> &observations)
{
  auto squared_distance = [](const LandmarkObs &a, const LandmarkObs &b) {
    // As we are interested only in which distance is _minimal_,
    // the real value of the distance is irrelevant, and I just
    // save sqrt computing time here.
    return pow(a.x - b.x, 2) + pow(a.y - b.y, 2);
  };
  for (auto &observation : observations)
  {
    double min_squared_dist = std::numeric_limits<double>::max();
    for (const auto &landmark : predicted)
    {
      const auto sq_dist = squared_distance(landmark, observation);
      if (sq_dist < min_squared_dist)
      {
        min_squared_dist = sq_dist;
        observation.id   = landmark.id;
      }
    }
  }

  std::sort(observations.begin(), observations.end());
}

// Returns all landmarks that lies within the rectangular region,
// set by upper left and lower right corners. IDs of the corner
// points are being ignored.
vector<LandmarkObs> getLandmarksInRegion(const Map &map_landmarks, LandmarkObs top_left,
                                         LandmarkObs bottom_right)
{
  vector<LandmarkObs> found;
  if (top_left.x >= bottom_right.x || top_left.y <= bottom_right.y)
  {
    return found;
  }

  for (const auto &landmark : map_landmarks.landmark_list)
  {
    if (landmark.x_f >= float(top_left.x) && landmark.y_f <= float(top_left.y) &&
        landmark.x_f <= float(bottom_right.x) && landmark.y_f >= float(bottom_right.y))
    {
      found.emplace_back(LandmarkObs{landmark.id_i, double(landmark.x_f), double(landmark.y_f)});
    }
  }

  // To quickly find a necessary landmark ID later, we need to sort them by ID.
  std::sort(found.begin(), found.end());

  return found;
}

void ParticleFilter::updateWeights(double sensor_range, Sigmas std_landmark,
                                   const vector<LandmarkObs> &observations,
                                   const Map &                map_landmarks)
{
  const auto gauss_norm{1. / 2 * M_PI * std_landmark.x * std_landmark.y};
  const auto doubled_sig_square_x{2. * pow(std_landmark.x, 2.)};
  const auto doubled_sig_square_y{2. * pow(std_landmark.y, 2.)};
  double     weight_sum = 0.0;

  for (size_t i{0}; i < total_particles_; ++i)
  {
    auto &     particle = particles_[i];
    const auto x_part{particle.x};
    const auto y_part{particle.y};
    const auto cos_theta{cos(particle.theta)};
    const auto sin_theta{sin(particle.theta)};

    LandmarkObs sensor_field_top_left{0, x_part - sensor_range, y_part + sensor_range};
    LandmarkObs sensor_field_bot_right{0, x_part + sensor_range, y_part - sensor_range};

    const auto predicted_landmarks =
        getLandmarksInRegion(map_landmarks, sensor_field_top_left, sensor_field_bot_right);

    // Transform all observations to a map coordinate system, with respect
    // to the current particle's map coordinates as origin.
    vector<LandmarkObs> observations_on_map;
    for (const auto &observation : observations)
    {
      const auto x_obs{observation.x};
      const auto y_obs{observation.y};
      const auto x_obs_on_map = x_part + (cos_theta * x_obs) - (sin_theta * y_obs);
      const auto y_obs_on_map = y_part + (sin_theta * x_obs) + (cos_theta * y_obs);
      observations_on_map.emplace_back(
          LandmarkObs{LandmarkObs::EMPTY_ID, x_obs_on_map, y_obs_on_map});
    }

    // Associate observations to closest landmarks
    dataAssociation(predicted_landmarks, observations_on_map);

    // And now calculate particle's weight
    double weight = 1.0;
    for (const auto &obs : observations_on_map)
    {
      auto found_lm =
          std::lower_bound(predicted_landmarks.begin(), predicted_landmarks.end(), obs.id);
      if (found_lm == predicted_landmarks.end())
      {
        // Should never happen during normal operation of the filter.
        continue;
      }
      const auto mu_x       = found_lm->x;
      const auto mu_y       = found_lm->y;
      const auto x_var      = pow((obs.x - mu_x), 2.) / doubled_sig_square_x;
      const auto y_var      = pow((obs.y - mu_y), 2.) / doubled_sig_square_y;
      const auto exponent   = -1. * (x_var + y_var);
      const auto obs_weight = gauss_norm * exp(exponent);
      weight *= obs_weight;
    }
    particles_[i].weight = weight;
    weights_[i]          = weight;
    weight_sum += weight;
  }

  // Prevent division by zero for edge cases when the sum is precisely 0.
  // This actually should never happen.
  if (weight_sum == 0.)
  {
    weight_sum = 1.;
  }

  // Normalize weights for further resampling
  for (size_t i{0}; i < total_particles_; ++i)
  {
    weights_[i] /= weight_sum;
    particles_[i].weight = weights_[i];
  }
}

void ParticleFilter::resample()
{
  std::discrete_distribution<size_t> distr(weights_.begin(), weights_.end());
  std::vector<Particle>              resampled;
  resampled.reserve(total_particles_);
  for (size_t i{0}; i < total_particles_; ++i)
  {
    resampled.emplace_back(particles_[distr(rnd_)]);
  }
  particles_ = std::move(resampled);
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
