#pragma once

#include "rhoban_utils/serialization/json_serializable.h"

#include <Eigen/Core>

#include <random>

namespace rhoban_bbo
{

class Optimizer : public rhoban_utils::JsonSerializable {
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

  /// Set the maximal number of calls to the reward function for the optimizer
  virtual void setMaxCalls(int max_calls) = 0;


  virtual void setLimits(const Eigen::MatrixXd & new_limits);
  const Eigen::MatrixXd & getLimits() const;
private:
  Eigen::MatrixXd limits;
};

}
