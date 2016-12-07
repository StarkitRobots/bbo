#include "rosban_bbo/monte_carlo_optimizer.h"

#include "rosban_random/tools.h"

#include "rosban_utils/xml_tools.h"

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
