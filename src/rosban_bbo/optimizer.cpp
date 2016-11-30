#include "rosban_bbo/optimizer.h"

namespace rosban_bbo
{

void Optimizer::setLimits(const Eigen::MatrixXd & new_limits)
{
  limits = new_limits;
}

const Eigen::MatrixXd & Optimizer::getLimits() const
{
  return limits;
}

}
