#include "rosban_bbo/optimizer.h"

namespace rosban_bbo
{

Eigen::VectorXd Optimizer::train(RewardFunc & reward,
                                 std::default_random_engine * engine) {
  Eigen::VectorXd initial_candidate = (limits.col(0) + limits.col(1)) / 2;
  return train(reward, initial_candidate, engine);
}

void Optimizer::setLimits(const Eigen::MatrixXd & new_limits)
{
  limits = new_limits;
}

const Eigen::MatrixXd & Optimizer::getLimits() const
{
  return limits;
}

}
