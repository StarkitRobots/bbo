#include "rhoban_bbo/simulated_annealing.h"

#include "rhoban_random/tools.h"

#include <iostream>

namespace rhoban_bbo
{
SimulatedAnnealing::SimulatedAnnealing() : initial_temperature(100), nb_trials(100), verbose(false)
{
}

SimulatedAnnealing::SimulatedAnnealing(const SimulatedAnnealing& other)
  : initial_temperature(other.initial_temperature), nb_trials(other.nb_trials), verbose(other.verbose)
{
}

Eigen::VectorXd SimulatedAnnealing::train(RewardFunc& reward_sampler, const Eigen::VectorXd& initial_candidate,
                                          std::default_random_engine* engine)
{
  Eigen::VectorXd state = initial_candidate;

  /// Acceptance of new samples is random and based on a random distribution
  std::uniform_real_distribution<double> distribution(0.0, 1.0);

  // Since we have a reward function and the goal is to minimize energy,
  // current_energy is -reward
  double current_energy = -reward_sampler(state, engine);

  for (int trial = 0; trial < nb_trials; trial++)
  {
    double temperature = getTemperature(trial);
    Eigen::VectorXd new_state = sampleNeighbor(state, temperature, engine);
    double energy = -reward_sampler(new_state, engine);
    double delta_energy = energy - current_energy;

    if (verbose)
    {
      std::cout << "SA: testing parameters set: trial " << trial << std::endl
                << new_state.transpose() << std::endl
                << "Energy: " << energy << std::endl;
    }

    // If energy has been reduced, accept new state and update energy
    if (delta_energy < 0)
    {
      if (verbose)
      {
        std::cout << "\tSample accepted" << std::endl;
      }
      state = new_state;
      current_energy = energy;
    }
    else
    {
      double p_accept = exp(-delta_energy / temperature);
      if (verbose)
      {
        std::cout << "Probability of acceptance: " << p_accept << std::endl;
      }
      if (distribution(*engine) < p_accept)
      {
        if (verbose)
        {
          std::cout << "\tAccepted" << std::endl;
        }
        // If state has been accepted update both, state and energy
        state = new_state;
        current_energy = energy;
      }
    }
  }

  return state;
}

double SimulatedAnnealing::getTemperature(int trial) const
{
  return (nb_trials - trial) * initial_temperature / nb_trials;
}

Eigen::VectorXd SimulatedAnnealing::sampleNeighbor(const Eigen::VectorXd& state, double temperature,
                                                   std::default_random_engine* engine)
{
  const Eigen::MatrixXd& limits = getLimits();
  Eigen::VectorXd amplitude = limits.col(1) - limits.col(0);

  amplitude = amplitude * (temperature / initial_temperature);

  Eigen::MatrixXd delta_limits(limits.rows(), limits.cols());
  delta_limits.col(0) = -amplitude;
  delta_limits.col(1) = amplitude;

  Eigen::VectorXd delta = rhoban_random::getUniformSamplesMatrix(delta_limits, 1, engine);
  Eigen::VectorXd neighbor = state + delta;

  // Bounding inside limits
  for (int dim = 0; dim < neighbor.rows(); dim++)
  {
    double min = limits(dim, 0);
    double max = limits(dim, 1);
    if (neighbor(dim) < min)
      neighbor(dim) = min;
    if (neighbor(dim) > max)
      neighbor(dim) = max;
  }
  return neighbor;
}

void SimulatedAnnealing::setMaxCalls(int max_calls)
{
  nb_trials = max_calls;
  // std::cout << "SA: new nb_trials: " << nb_trials << std::endl;
}

std::string SimulatedAnnealing::getClassName() const
{
  return "SimulatedAnnealing";
}

Json::Value SimulatedAnnealing::toJson() const
{
  Json::Value v;
  v["nb_trials"] = nb_trials;
  v["initial_temperature"] = initial_temperature;
  v["verbose"] = verbose;
  return v;
}

void SimulatedAnnealing::fromJson(const Json::Value& v, const std::string& dir_name)
{
  (void)dir_name;
  rhoban_utils::tryRead(v, "nb_trials", &nb_trials);
  rhoban_utils::tryRead(v, "initial_temperature", &initial_temperature);
  rhoban_utils::tryRead(v, "verbose", &verbose);
}

std::unique_ptr<Optimizer> SimulatedAnnealing::clone() const
{
  return std::unique_ptr<Optimizer>(new SimulatedAnnealing(*this));
}

}  // namespace rhoban_bbo
