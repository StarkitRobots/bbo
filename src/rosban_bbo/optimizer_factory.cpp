#include "rosban_bbo/optimizer_factory.h"

#include "rosban_bbo/cmaes_optimizer.h"
#include "rosban_bbo/composite_optimizer.h"
#include "rosban_bbo/cross_entropy.h"
#include "rosban_bbo/monte_carlo_optimizer.h"
#include "rosban_bbo/partial_optimizer.h"
#include "rosban_bbo/simulated_annealing.h"

namespace rosban_bbo
{

OptimizerFactory::OptimizerFactory()
{
  registerBuilder("MonteCarloOptimizer",
                  []() { return std::unique_ptr<Optimizer>(new MonteCarloOptimizer); });
  registerBuilder("CMAESOptimizer",
                  []() { return std::unique_ptr<Optimizer>(new CMAESOptimizer); });
  registerBuilder("CompositeOptimizer",
                  []() { return std::unique_ptr<Optimizer>(new CompositeOptimizer); });
  registerBuilder("CrossEntropy",
                  []() { return std::unique_ptr<Optimizer>(new CrossEntropy); });
  registerBuilder("PartialOptimizer",
                  []() { return std::unique_ptr<Optimizer>(new PartialOptimizer); });
  registerBuilder("SimulatedAnnealing",
                  []() { return std::unique_ptr<Optimizer>(new SimulatedAnnealing); });
}

}
