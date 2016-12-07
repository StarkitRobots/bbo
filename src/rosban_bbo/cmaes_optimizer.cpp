#include "rosban_bbo/cmaes_optimizer.h"

#include "libcmaes/cmaes.h"

using namespace libcmaes;

namespace rosban_bbo
{

CMAESOptimizer::CMAESOptimizer()
  : quiet(true),
    nb_iterations(-1),
    nb_evaluations(1000),
    nb_restarts(1),
    population_size(-1),
    ftolerance(1e-10),
    max_history(-1)
{
}

Eigen::VectorXd CMAESOptimizer::train(RewardFunc & reward,
                                      const Eigen::VectorXd & initial_candidate,
                                      std::default_random_engine * engine)
{
  libcmaes::FitFuncEigen fitness =
    [reward, engine](const Eigen::VectorXd & params)
    {
      return -reward(params, engine);
    };

  int dims = getLimits().rows();
  // For each dimension, sigma depends on the overall size of the
  // parameters amplitude
  std::vector<double> init_params(dims), lower_bounds(dims), upper_bounds(dims);
  for (int dim = 0; dim < dims; dim++) {
    double dim_min = getLimits()(dim, 0);
    double dim_max = getLimits()(dim, 1);
    // Initial parameters are in the middle of the space
    init_params[dim] = initial_candidate(dim);
    // Limits are also saved as vector<double>
    lower_bounds[dim] = dim_min;
    upper_bounds[dim] = dim_max;
  }
  // GenoType + PhenoType values (limits + linear scaling)
  GenoPheno<pwqBoundStrategy,linScalingStrategy> gp(lower_bounds.data(),
                                                    upper_bounds.data(),
                                                    dims);
  // cmaes parameters (probably not chosen optimally yet)
  double sigma = -1;//Automatic choice for the step size
  // Population size
  int lambda = -1;
  if (population_size > 0) lambda = population_size;
  // Creating the cma_params object with bound and linear scaling strategy
  CMAParameters<GenoPheno<pwqBoundStrategy,linScalingStrategy>> cma_params
    (dims,init_params.data(), sigma,lambda,0,gp);
  // Updating params of
  //cma_params.set_noisy();//Effect of this function is quite unclear
  cma_params.set_quiet(quiet);
  cma_params.set_mt_feval(true);
  cma_params.set_str_algo("abipop");
  cma_params.set_elitism(1);
  cma_params.set_restarts(nb_restarts);
  cma_params.set_max_fevals(nb_evaluations);
  if (nb_iterations > 0) cma_params.set_max_iter(nb_iterations);
  if (ftolerance > 0) cma_params.set_ftolerance(ftolerance);
  if (max_history > 0) cma_params.set_max_hist(max_history);
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
  rosban_utils::xml_tools::write<bool>  ("quiet"          , quiet          , out);
  rosban_utils::xml_tools::write<int>   ("nb_iterations"  , nb_iterations  , out);
  rosban_utils::xml_tools::write<int>   ("nb_evaluations" , nb_evaluations , out);
  rosban_utils::xml_tools::write<int>   ("nb_restarts"    , nb_restarts    , out);
  rosban_utils::xml_tools::write<int>   ("population_size", population_size, out);
  rosban_utils::xml_tools::write<int>   ("max_history"    , max_history    , out);
  rosban_utils::xml_tools::write<double>("ftolerance"     , ftolerance     , out);
}

void CMAESOptimizer::from_xml(TiXmlNode *node)
{
  rosban_utils::xml_tools::try_read<bool>  (node, "quiet"          , quiet          );
  rosban_utils::xml_tools::try_read<int>   (node, "nb_iterations"  , nb_iterations  );
  rosban_utils::xml_tools::try_read<int>   (node, "nb_evaluations" , nb_evaluations );
  rosban_utils::xml_tools::try_read<int>   (node, "nb_restarts"    , nb_restarts    );
  rosban_utils::xml_tools::try_read<int>   (node, "population_size", population_size);
  rosban_utils::xml_tools::try_read<int>   (node, "max_history"    , max_history    );
  rosban_utils::xml_tools::try_read<double>(node, "ftolerance"     , ftolerance     );
}


}
