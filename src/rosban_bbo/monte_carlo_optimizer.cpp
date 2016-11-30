#include "rosban_bbo/monte_carlo_optimizer.h"

#include "rosban_random/tools.h"

#include "rosban_utils/xml_tools.h"

namespace rosban_bbo
{

MonteCarloOptimizer::MonteCarloOptimizer()
  : nb_trials(100)
{}

Eigen::VectorXd MonteCarloOptimizer::train(RewardFunc & reward_sampler,
                                           std::default_random_engine * engine)
{
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
  return best_candidate;
}

std::string MonteCarloOptimizer::class_name() const {
  return "MonteCarloOptimizer";
}

void MonteCarloOptimizer::to_xml(std::ostream &out) const {
  rosban_utils::xml_tools::write("nb_trials", nb_trials, out);
}
void MonteCarloOptimizer::from_xml(TiXmlNode *node) {
  rosban_utils::xml_tools::try_read(node, "nb_trials", nb_trials);
}


}
