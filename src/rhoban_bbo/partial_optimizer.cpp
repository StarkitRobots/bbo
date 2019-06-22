#include "starkit_bbo/partial_optimizer.h"

#include "starkit_bbo/optimizer_factory.h"

#include "starkit_random/tools.h"
#include "starkit_utils/timing/time_stamp.h"

using starkit_utils::TimeStamp;

namespace starkit_bbo
{
PartialOptimizer::PartialOptimizer() : ratio_used(0.5)
{
}

PartialOptimizer::PartialOptimizer(const PartialOptimizer& other) : ratio_used(other.ratio_used)
{
}

Eigen::VectorXd PartialOptimizer::train(RewardFunc& reward, const Eigen::VectorXd& initial_candidate,
                                        std::default_random_engine* engine)
{
  int base_dims = getLimits().rows();
  int nb_dims_used = std::max(1, (int)(base_dims * ratio_used));
  std::vector<size_t> dims_used = starkit_random::getKDistinctFromN(nb_dims_used, base_dims, engine);
  // Preparing new reward function
  RewardFunc hacked_reward = [&](const Eigen::VectorXd& hacked_params, std::default_random_engine* engine) {
    /// Getting real params
    Eigen::VectorXd source_params = initial_candidate;
    for (size_t dim_idx = 0; dim_idx < dims_used.size(); dim_idx++)
    {
      source_params[dims_used[dim_idx]] = hacked_params[dim_idx];
    }
    return reward(source_params, engine);
  };
  // Compute hacked space and hacked guess
  Eigen::MatrixXd hacked_space = Eigen::MatrixXd::Zero(nb_dims_used, 2);
  Eigen::VectorXd hacked_guess(nb_dims_used);
  for (size_t dim_idx = 0; dim_idx < dims_used.size(); dim_idx++)
  {
    hacked_guess(dim_idx) = initial_candidate(dims_used[dim_idx]);
    hacked_space.block(dim_idx, 0, 1, 2) = getLimits().row(dims_used[dim_idx]);
  }
  // Compute hacked guess
  optimizer->setLimits(hacked_space);
  // Best hacked_params
  Eigen::VectorXd best_hacked_params = optimizer->train(hacked_reward, hacked_guess, engine);
  // Getting back best params
  Eigen::VectorXd best_params = initial_candidate;
  for (size_t dim_idx = 0; dim_idx < dims_used.size(); dim_idx++)
  {
    best_params(dims_used[dim_idx]) = best_hacked_params(dim_idx);
  }
  return best_params;
}

void PartialOptimizer::setMaxCalls(int max_calls)
{
  optimizer->setMaxCalls(max_calls);
}

std::string PartialOptimizer::getClassName() const
{
  return "PartialOptimizer";
}

Json::Value PartialOptimizer::toJson() const
{
  Json::Value v;
  v["ratio_used"] = ratio_used;
  v["optimizer"] = optimizer->toFactoryJson();
  return v;
}

void PartialOptimizer::fromJson(const Json::Value& v, const std::string& dir_name)
{
  (void)dir_name;
  starkit_utils::tryRead(v, "ratio_used", &ratio_used);
  optimizer = OptimizerFactory().read(v, "optimizer", dir_name);
}

std::unique_ptr<Optimizer> PartialOptimizer::clone() const
{
  return std::unique_ptr<Optimizer>(new PartialOptimizer(*this));
}

}  // namespace starkit_bbo
