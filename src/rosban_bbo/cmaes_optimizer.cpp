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
    max_history(-1),
    elitism(1)
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
  if (initial_candidate.rows() != dims) {
    std::ostringstream oss;
    oss << "CMAESOptimizer: Provided initial candidate has invalid number of rows: "
        << initial_candidate.rows() << " while expecting " << dims;
    throw std::runtime_error(oss.str());
  } 

  // Store the required parameters in vectors
  std::vector<double> init_params(dims), lower_bounds(dims), upper_bounds(dims);
  for (int dim = 0; dim < dims; dim++) {
    double dim_min = getLimits()(dim, 0);
    double dim_max = getLimits()(dim, 1);
    init_params[dim] = initial_candidate(dim);
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
  cma_params.set_elitism(elitism);
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

void CMAESOptimizer::setMaxCalls(int max_calls) {
  //TODO: work on auto-conf
  nb_evaluations = max_calls / (nb_restarts + 1);
  // Check for errors
  if (population_size > 2 * nb_evaluations) {
    population_size = nb_evaluations / 2;
    //throw std::logic_error("Number of evaluations is smaller than population size");
  }
  //std::cout << "CMAES: New nb_evaluations: " << nb_evaluations << std::endl;
}

std::string CMAESOptimizer::getClassName() const
{
  return "CMAESOptimizer";
}

Json::Value CMAESOptimizer::toJson() const
{
  Json::Value v;
  v["quiet"          ] = quiet          ;
  v["nb_iterations"  ] = nb_iterations  ;
  v["nb_evaluations" ] = nb_evaluations ;
  v["nb_restarts"    ] = nb_restarts    ;
  v["population_size"] = population_size;
  v["max_history"    ] = max_history    ;
  v["elitism"        ] = elitism        ;
  v["ftolerance"     ] = ftolerance     ;
  return v;
}

void CMAESOptimizer::fromJson(const Json::Value & v,
                              const std::string & dir_name)
{
  (void)dir_name;
  rhoban_utils::tryRead(v, "quiet"          , &quiet);
  rhoban_utils::tryRead(v, "nb_iterations"  , &nb_iterations  );
  rhoban_utils::tryRead(v, "nb_evaluations" , &nb_evaluations );
  rhoban_utils::tryRead(v, "nb_restarts"    , &nb_restarts    );
  rhoban_utils::tryRead(v, "population_size", &population_size);
  rhoban_utils::tryRead(v, "max_history"    , &max_history    );
  rhoban_utils::tryRead(v, "elitism"        , &elitism        );
  rhoban_utils::tryRead(v, "ftolerance"     , &ftolerance     );
}


}
