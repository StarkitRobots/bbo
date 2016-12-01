#include "rosban_bbo/cmaes_optimizer.h"

#include "libcmaes/cmaes.h"

using namespace libcmaes;

namespace rosban_bbo
{

CMAESOptimizer::CMAESOptimizer()
  : nb_iterations(100),
    nb_evaluations(1000),
    nb_restarts(1)
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
  cma_params.set_restarts(nb_restarts);
  cma_params.set_max_iter(nb_iterations);
  cma_params.set_max_fevals(nb_evaluations);
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
  rosban_utils::xml_tools::write<int>("nb_iterations" , nb_iterations , out);
  rosban_utils::xml_tools::write<int>("nb_evaluations", nb_evaluations, out);
  rosban_utils::xml_tools::write<int>("nb_restarts"   , nb_restarts   , out);
}

void CMAESOptimizer::from_xml(TiXmlNode *node)
{
  rosban_utils::xml_tools::try_read<int>(node, "nb_iterations" , nb_iterations );
  rosban_utils::xml_tools::try_read<int>(node, "nb_evaluations", nb_evaluations);
  rosban_utils::xml_tools::try_read<int>(node, "nb_restarts"   , nb_restarts   );
}


}
