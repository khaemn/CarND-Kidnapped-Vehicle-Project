#include "json.hpp"
#include "particle_filter.h"

#include <iostream>
#include <math.h>
#include <string>
#include <uWS/uWS.h>

// for convenience
using nlohmann::json;
using std::string;
using std::vector;

static constexpr int PARTICLE_COUNT = 1000;

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s)
{
  auto found_null = s.find("null");
  auto b1         = s.find_first_of("[");
  auto b2         = s.find_first_of("]");
  if (found_null != string::npos)
  {
    return "";
  }
  else if (b1 != string::npos && b2 != string::npos)
  {
    return s.substr(b1, b2 - b1 + 1);
  }
  return "";
}

int main()
{
  uWS::Hub hub;

  // Set up parameters here
  double delta_t_sec    = 0.1;  // Time elapsed between measurements [sec]
  double sensor_range_m = 50;   // Sensor range [m]

  // GPS measurement uncertainty [x [m], y [m], theta [rad]]

  const Sigmas sigma_pos {0.3, 0.3, 0.01};
  // Landmark measurement uncertainty [x [m], y [m]]
  const Sigmas sigma_landmark {0.3, 0.3, 0.0};

  // Read map data
  Map map;
  if (!read_map_data("../data/map_data.txt", map))
  {
    std::cout << "Error: Could not open map file" << std::endl;
    return -1;
  }

  ParticleFilter filter;

  hub.onMessage([&filter, &map, &delta_t_sec, &sensor_range_m, &sigma_pos, &sigma_landmark](
                    uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length, uWS::OpCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    const bool is_websocket_message = length && length > 2 && data[0] == '4' && data[1] == '2';
    if (!is_websocket_message)
    {
      return;
    }

    auto s = hasData(string(data));

    if (s == "")
    {
      static const string msg = "42[\"manual\",{}]";
      ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      return;
    }

    auto j = json::parse(s);

    const string event = j[0].get<string>();

    if (event != "telemetry")
    {
      return;
    }

    // j[1] is the data JSON object
    static constexpr auto DATA_IDX{1};
    if (!filter.initialized())
    {
      // Sense noisy position data from the simulator
      double sense_x     = std::stod(j[DATA_IDX]["sense_x"].get<string>());
      double sense_y     = std::stod(j[DATA_IDX]["sense_y"].get<string>());
      double sense_theta = std::stod(j[DATA_IDX]["sense_theta"].get<string>());

      filter.init(PARTICLE_COUNT, sense_x, sense_y, sense_theta, sigma_pos);
    }
    else
    {
      // Predict the vehicle's next state from previous
      //   (noiseless control) data.
      double previous_velocity = std::stod(j[DATA_IDX]["previous_velocity"].get<string>());
      double previous_yawrate  = std::stod(j[DATA_IDX]["previous_yawrate"].get<string>());

      filter.predict(delta_t_sec, sigma_pos, previous_velocity, previous_yawrate);
    }

    // receive noisy observation data from the simulator
    // sense_observations in JSON format
    //   [{obs_x,obs_y},{obs_x,obs_y},...{obs_x,obs_y}]
    vector<LandmarkObs> noisy_observations;
    string              sense_observations_x = j[DATA_IDX]["sense_observations_x"];
    string              sense_observations_y = j[DATA_IDX]["sense_observations_y"];

    vector<float>      x_sense;
    std::istringstream iss_x(sense_observations_x);

    std::copy(std::istream_iterator<float>(iss_x), std::istream_iterator<float>(),
              std::back_inserter(x_sense));

    vector<float>      y_sense;
    std::istringstream iss_y(sense_observations_y);

    std::copy(std::istream_iterator<float>(iss_y), std::istream_iterator<float>(),
              std::back_inserter(y_sense));

    for (size_t i = 0; i < x_sense.size(); ++i)
    {
      LandmarkObs obs;
      obs.x = double(x_sense[i]);
      obs.y = double(y_sense[i]);
      noisy_observations.push_back(obs);
    }

    // Update the weights and resample
    filter.updateWeights(sensor_range_m, sigma_landmark, noisy_observations, map);
    filter.resample();

    // Calculate and output the average weighted error of the particle
    //   filter over all time steps so far.
    vector<Particle> particles      = filter.particles();
    size_t           num_particles  = particles.size();
    double           highest_weight = -1.0;
    Particle         best_particle;
    double           weight_sum = 0.0;
    for (size_t i = 0; i < num_particles; ++i)
    {
      if (particles[i].weight > highest_weight)
      {
        highest_weight = particles[i].weight;
        best_particle  = particles[i];
      }

      weight_sum += particles[i].weight;
    }

    std::cout << "highest w " << highest_weight << std::endl;
    std::cout << "average w " << weight_sum / num_particles << std::endl;

    json msgJson;
    msgJson["best_particle_x"]     = best_particle.x;
    msgJson["best_particle_y"]     = best_particle.y;
    msgJson["best_particle_theta"] = best_particle.theta;

    // Optional message data used for debugging particle's sensing
    //   and associations
    msgJson["best_particle_associations"] = filter.getAssociations(best_particle);
    msgJson["best_particle_sense_x"]      = filter.getSenseCoord(best_particle, "X");
    msgJson["best_particle_sense_y"]      = filter.getSenseCoord(best_particle, "Y");

    auto msg = "42[\"best_particle\"," + msgJson.dump() + "]";
    // std::cout << msg << std::endl;
    ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
  });  // end h.onMessage

  hub.onConnection(
      [](uWS::WebSocket<uWS::SERVER>, uWS::HttpRequest) { std::cout << "Connected" << std::endl; });

  hub.onDisconnection([](uWS::WebSocket<uWS::SERVER> ws, int, char *, size_t) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  constexpr int port = 4567;
  if (hub.listen(port))
  {
    std::cout << "Listening to port " << port << std::endl;
  }
  else
  {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }

  hub.run();
}
