#pragma once

#include "rosban_utils/serializable.h"

#include <Eigen/Core>

#include <random>

namespace rosban_bbo
{

class Optimizer : public rosban_utils::Serializable {
public:

  /// RewardFunc are functions (eventually stochastic) returning the sampled
  /// value for the given set of parameters
  typedef std::function<double(const Eigen::VectorXd & parameters,
                               std::default_random_engine * engine)> RewardFunc;
                               

  virtual Eigen::VectorXd train(RewardFunc & reward,
                                std::default_random_engine * engine) = 0;


  void setLimits(const Eigen::MatrixXd & new_limits);
  const Eigen::MatrixXd & getLimits() const;
private:
  Eigen::MatrixXd limits;
};

}
