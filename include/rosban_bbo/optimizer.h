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
                               

  /// Train the parameters inside the limits using default prior candidate
  /// default is the center of the space
  Eigen::VectorXd train(RewardFunc & reward, std::default_random_engine * engine);

  virtual Eigen::VectorXd train(RewardFunc & reward,
                                const Eigen::VectorXd & initial_candidate,
                                std::default_random_engine * engine) = 0;


  void setLimits(const Eigen::MatrixXd & new_limits);
  const Eigen::MatrixXd & getLimits() const;
private:
  Eigen::MatrixXd limits;
};

}
