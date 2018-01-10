#include "rosban_bbo/monte_carlo_optimizer.h"

#include "rosban_random/tools.h"

namespace rosban_bbo
{

MonteCarloOptimizer::MonteCarloOptimizer()
  : nb_trials(100)
{}

Eigen::VectorXd MonteCarloOptimizer::train(RewardFunc & reward_sampler,
                                           const Eigen::VectorXd & initial_candidate,
                                           std::default_random_engine * engine)
{
  // No need for initial candidate when doing pure random
  (void) initial_candidate;

  // Sampling candidates
  std::vector<Eigen::VectorXd> candidates;
  candidates = rosban_random::getUniformSamples(getLimits(),
                                                nb_trials,
                                                engine);

  // Getting best candidates
  double best_reward = std::numeric_limits<double>::lowest();
  Eigen::VectorXd best_candidate;
  for (const Eigen::VectorXd & candidate : candidates) {
    double reward = reward_sampler(candidate, engine);
    if (reward > best_reward) {
      best_reward = reward;
      best_candidate = candidate;
    }
  }
  if (best_candidate.rows() == 0) {
    std::ostringstream oss;
    oss << "MonteCarloOptimizer::train: no candidate has been found: " << std::endl
        << "nb_trials: " << nb_trials << std::endl
        << "limits" << std::endl << getLimits() << std::endl;
    throw std::logic_error(oss.str());
  }
  return best_candidate;
}

void MonteCarloOptimizer::setMaxCalls(int max_calls) {
  nb_trials = max_calls;
}

std::string MonteCarloOptimizer::getClassName() const {
  return "MonteCarloOptimizer";
}

Json::Value MonteCarloOptimizer::toJson() const {
  Json::Value v;
  v["nb trials"] = nb_trials;
  return v;
}

void MonteCarloOptimizer::fromJson(const Json::Value & v, const std::string & dir_name) {
  (void) dir_name;
  rhoban_utils::tryRead(v, "nb_trials", &nb_trials);
}


}
