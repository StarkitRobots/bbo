#include "rhoban_bbo/optimizer_factory.h"

#include "rhoban_bbo/cmaes_optimizer.h"
#include "rhoban_bbo/composite_optimizer.h"
#include "rhoban_bbo/cross_entropy.h"
#include "rhoban_bbo/hoo.h"
#include "rhoban_bbo/monte_carlo_optimizer.h"
#include "rhoban_bbo/partial_optimizer.h"
#include "rhoban_bbo/simulated_annealing.h"

namespace rhoban_bbo
{
OptimizerFactory::OptimizerFactory()
{
  registerBuilder("MonteCarloOptimizer", []() { return std::unique_ptr<Optimizer>(new MonteCarloOptimizer); });
  registerBuilder("CMAESOptimizer", []() { return std::unique_ptr<Optimizer>(new CMAESOptimizer); });
  registerBuilder("CompositeOptimizer", []() { return std::unique_ptr<Optimizer>(new CompositeOptimizer); });
  registerBuilder("CrossEntropy", []() { return std::unique_ptr<Optimizer>(new CrossEntropy); });
  registerBuilder("HOO", []() { return std::unique_ptr<Optimizer>(new HOO); });
  registerBuilder("PartialOptimizer", []() { return std::unique_ptr<Optimizer>(new PartialOptimizer); });
  registerBuilder("SimulatedAnnealing", []() { return std::unique_ptr<Optimizer>(new SimulatedAnnealing); });
}

}  // namespace rhoban_bbo
