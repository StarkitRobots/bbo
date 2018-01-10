#include "rosban_bbo/composite_optimizer.h"

#include "rosban_bbo/optimizer_factory.h"

#include "rosban_random/tools.h"
#include "rhoban_utils/timing/time_stamp.h"

using rhoban_utils::TimeStamp;

namespace rosban_bbo
{

CompositeOptimizer::CompositeOptimizer()
  : validation_trials(1), debug_level(0)
{}

Eigen::VectorXd CompositeOptimizer::train(RewardFunc & reward,
                                          const Eigen::VectorXd & initial_candidate,
                                          std::default_random_engine * engine) {
  if (optimizers.size() == 0) {
    throw std::logic_error("CompositeOptimizer::train: no optimizers found");
  }
  // If weights have been provided choose an optimizer randomly
  if (weights.size() != 0) {
    size_t optimizer_idx = rosban_random::sampleWeightedIndices(weights, 1, engine)[0];
    std::string name = optimizers[optimizer_idx]->getClassName();
    if (names.size() > optimizer_idx) {
      name = names[optimizer_idx] ;
    }
    if (debug_level > 0) {
      std::cout << "Optimizing with " << name << std::endl;
    }
    return optimizers[optimizer_idx]->train(reward, initial_candidate, engine);
  }
  else {
    double best_reward = std::numeric_limits<double>::lowest();
    Eigen::VectorXd best_sol;
    std::string best_name;
    for (unsigned int idx = 0; idx < optimizers.size(); idx++) {
      std::string curr_name = optimizers[idx]->getClassName();
      if (names.size() > idx) {
        curr_name = names[idx];
      }
      TimeStamp start = TimeStamp::now();
      Eigen::VectorXd sol = optimizers[idx]->train(reward, initial_candidate, engine);
      TimeStamp end = TimeStamp::now();
      double avg_reward = 0;
      for (int i = 0; i < validation_trials; i++) {
        avg_reward += reward(sol, engine);
      }
      avg_reward /= validation_trials;
      if (debug_level > 0 ) {
        std::cout << "Optimization with " << curr_name << ": "
                  << diffMs(start,end) << " ms" << std::endl;
        std::cout << "-> Sol : " << sol.transpose() << std::endl;
        std::cout << "-> Average reward : " << avg_reward << std::endl;
      }
      if (avg_reward > best_reward) {
        best_sol = sol;
        best_reward = avg_reward;
        best_name = curr_name;
      }
    }
    if (debug_level > 0) {
      std::cout << "Best optimizer: " << best_name << std::endl;
    }
    return best_sol;
  }
}
std::string CompositeOptimizer::getClassName() const {
  return "CompositeOptimizer";
}

Json::Value CompositeOptimizer::toJson() const {
  Json::Value v;
  v["validation_trials"] = validation_trials;
  v["debug_level"]       = debug_level;
  v["names"]   = rhoban_utils::vector2Json(names);
  v["weights"] = rhoban_utils::vector2Json(weights);
  v["optimizers"] = Json::Value();
  for (Json::ArrayIndex idx = 0; idx < optimizers.size(); idx++) {
    v["optimizers"][idx] = optimizers[idx]->toFactoryJson();
  }
  return v;
}

void CompositeOptimizer::fromJson(const Json::Value & v, const std::string & dir_name) {
  rhoban_utils::tryRead(v, "validation_trials", &validation_trials);
  rhoban_utils::tryRead(v, "debug_level", &debug_level);
  rhoban_utils::tryReadVector(v, "names", &names);
  rhoban_utils::tryReadVector(v, "weights", &weights);
  optimizers = OptimizerFactory().readVector(v, "optimizers", dir_name);
  // Checking consistency of informations read
  if (names.size() != 0 && names.size() != optimizers.size()) {
    std::ostringstream oss;
    oss << "CompositeOptimizer::from_xml: Invalid length for names "
        << names.size() << " while 0 or " << optimizers.size() << " was expected";
    throw std::logic_error(oss.str());
  }
  if (weights.size() != 0 && weights.size() != optimizers.size()) {
    std::ostringstream oss;
    oss << "CompositeOptimizer::from_xml: Invalid length for weights "
        << weights.size() << " while 0 or " << optimizers.size() << " was expected";
    throw std::logic_error(oss.str());
  }
}

void CompositeOptimizer::setMaxCalls(int max_calls) {
  // If all optimizers are used, then max_calls has to be reduced
  if (weights.size() == 0) {
    // Part of the calls are required for validation
    max_calls = max_calls - validation_trials * optimizers.size();
    // All optimizers are given the same number of calls
    max_calls = (int)(max_calls / optimizers.size());
  }
  for (size_t idx = 0; idx < optimizers.size(); idx++) {
    optimizers[idx]->setMaxCalls(max_calls / optimizers.size());
  }
}

void CompositeOptimizer::setLimits(const Eigen::MatrixXd & new_limits) {
  Optimizer::setLimits(new_limits);
  for (unsigned int i = 0; i < optimizers.size(); i++) {
    optimizers[i]->setLimits(new_limits);
  }
}

}
