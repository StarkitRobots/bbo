#include "starkit_bbo/optimizer.h"

namespace starkit_bbo
{
Eigen::VectorXd Optimizer::train(RewardFunc& reward, std::default_random_engine* engine)
{
  if (limits.rows() == 0)
  {
    throw std::logic_error("starkit_bbo::Optimizer: limits of the optimizer have not been initialized");
  }
  Eigen::VectorXd initial_candidate = (limits.col(0) + limits.col(1)) / 2;
  return train(reward, initial_candidate, engine);
}

void Optimizer::setLimits(const Eigen::MatrixXd& new_limits)
{
  limits = new_limits;
}

const Eigen::MatrixXd& Optimizer::getLimits() const
{
  return limits;
}

}  // namespace starkit_bbo
