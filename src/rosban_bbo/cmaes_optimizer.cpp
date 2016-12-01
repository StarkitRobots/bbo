#include "rosban_bbo/cmaes_optimizer.h"

#include "libcmaes/cmaes.h"

using namespace libcmaes;

namespace rosban_bbo
{

CMAESOptimizer::CMAESOptimizer()
{
}

Eigen::VectorXd CMAESOptimizer::train(RewardFunc & reward,
                                      std::default_random_engine * engine)
{
  libcmaes::FitFuncEigen fitness =
    [reward, engine](const Eigen::VectorXd & params)
    {
      //TODO: generate another random engine based on engine and use a mutex
      return -reward(params, engine);
    };

  int dims = getLimits().rows();
  // Sigma depends on the smallest dimension:
  // TODO: normalize dimensions
  double sigma = (getLimits().col(1) - getLimits().col(0)).minCoeff() / 10;
  // Initial parameters are in the middle of the space
  Eigen::VectorXd init_params = (getLimits().col(0) + getLimits().col(1)) / 2;
  
  // cmaes boundaries
  GenoPheno<pwqBoundStrategy> gp(getLimits().col(0).data(),
                                 getLimits().col(1).data(),
                                 dims);
  // cmaes parameters
  // TODO: replace those parameter by custom parameters
  CMAParameters<GenoPheno<pwqBoundStrategy>> cma_params(init_params, sigma, -1, 0, gp);
  cma_params.set_quiet(true);
  cma_params.set_mt_feval(true);
  cma_params.set_str_algo("abipop");
  cma_params.set_noisy();
  cma_params.set_elitism(1);
  cma_params.set_restarts(1);
  cma_params.set_max_iter(1000);
  cma_params.set_max_fevals(1000);
  // Solve cmaes
  CMASolutions sols = cmaes<GenoPheno<pwqBoundStrategy>>(fitness,cma_params);
  return sols.get_best_seen_candidate().get_x_dvec();
}

std::string CMAESOptimizer::class_name() const
{
  return "CMAESOptimizer";
}

void CMAESOptimizer::to_xml(std::ostream &out) const
{
}

void CMAESOptimizer::from_xml(TiXmlNode *node)
{
}


}
