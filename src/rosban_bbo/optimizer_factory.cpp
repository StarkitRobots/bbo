#include "rosban_bbo/optimizer_factory.h"

#include "rosban_bbo/cmaes_optimizer.h"
#include "rosban_bbo/monte_carlo_optimizer.h"

namespace rosban_bbo
{

OptimizerFactory::OptimizerFactory()
{
  registerBuilder("MonteCarloOptimizer",
                  []() { return std::unique_ptr<Optimizer>(new MonteCarloOptimizer); });
  registerBuilder("CMAESOptimizer",
                  []() { return std::unique_ptr<Optimizer>(new CMAESOptimizer); });
}

}
