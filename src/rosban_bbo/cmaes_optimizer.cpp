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
  // For each dimension, sigma depends on the overall size of the
  // parameters amplitude
  std::vector<double> sigma(dims), init_params(dims);
  std::vector<double> lower_bounds(dims), upper_bounds(dims);
  for (int dim = 0; dim < dims; dim++) {
    double dim_min = getLimits()(dim, 0);
    double dim_max = getLimits()(dim, 1);
    double dim_mean = (dim_max + dim_min) / 2.0;
    double dim_size = dim_max - dim_min;
    // According to CMA-ES documentation, solution should lie inside mu +- 2 sigma
    sigma[dim] = 0.25 * dim_size;
    // Initial parameters are in the middle of the space
    init_params[dim] = dim_mean;
    // Limits are also saved as vector<double>
    lower_bounds[dim] = dim_min;
    upper_bounds[dim] = dim_max;
  }
  // GenoType + PhenoType values (limits + linear scaling)
  GenoPheno<pwqBoundStrategy,linScalingStrategy> gp(lower_bounds.data(),
                                                    upper_bounds.data(),
                                                    dims);
  // cmaes parameters (probably not chosen optimally yet)
  CMAParameters<GenoPheno<pwqBoundStrategy,linScalingStrategy>> cma_params
    (dims,init_params.data(), 0.1,-1,0,gp);
  //CMAParameters<> cma_params(init_params, sigma, -1, lower_bounds, upper_bounds);
  cma_params.set_quiet(true);
  cma_params.set_mt_feval(true);
  cma_params.set_str_algo("abipop");
  cma_params.set_noisy();
  cma_params.set_elitism(1);
  cma_params.set_restarts(nb_restarts);
  cma_params.set_max_iter(nb_iterations);
  cma_params.set_max_fevals(nb_evaluations);
  // Solve cmaes
  CMASolutions sols = cmaes<GenoPheno<pwqBoundStrategy,linScalingStrategy>>(fitness,cma_params);
  Eigen::VectorXd best_sol = gp.pheno(sols.get_best_seen_candidate().get_x_dvec());
  // Checking bounds
  bool inside_bounds = true;
  for (int dim = 0; dim < dims; dim++) {
    if (best_sol(dim) < getLimits()(dim,0) ||
        best_sol(dim) > getLimits()(dim,1)) {
      inside_bounds = false;
    }
  }
  if (!inside_bounds) {
    std::cout << "WARNING: solution provided by CMAES optimizer is out of bounds" << std::endl;
    std::cout << "sol: " << best_sol.transpose() << std::endl;
    std::cout << "sol2: " << sols << std::endl;
    std::cout << "space:" << std::endl << getLimits().transpose() << std::endl;
  }
  return best_sol;
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
