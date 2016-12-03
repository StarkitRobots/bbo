#include "libcmaes/cmaes.h"

using namespace libcmaes;

FitFunc easy_func = [] (const double *x, const int N)
{
  double val = 0;
  for (int i = 0; i < N; i++) {
    val += x[i] * x[i];
  }
  return val;
};

/// Those values are shared by all dimensions
double dim_min = 1.0;
double dim_max = 3.0;

void testBounds1(int nb_dims)
{
  // Compute initial guess, step size and boundaries
  std::vector<double> sigma(nb_dims), init_params(nb_dims);
  std::vector<double> lower_bounds(nb_dims), upper_bounds(nb_dims);
  for (int dim = 0; dim < nb_dims; dim++) {
    sigma[dim] = 0.25 * (dim_max - dim_min);// Solution is likely to be in x0 +- 2 * sigma
    init_params[dim] = (dim_min + dim_max) / 2;
    lower_bounds[dim] = dim_min;
    upper_bounds[dim] = dim_max;
  }
  // Setup parameters
  CMAParameters<> cma_params(init_params, sigma, -1, lower_bounds, upper_bounds);
  cma_params.set_quiet(true);
  cma_params.set_mt_feval(false);
  //cma_params.set_str_algo("abipop");
  //cma_params.set_noisy();
  //cma_params.set_elitism(1);
  cma_params.set_restarts(0);
  cma_params.set_max_iter(50);
  cma_params.set_max_fevals(100);
  CMASolutions sols = cmaes<>(easy_func,cma_params);
  Eigen::VectorXd best_sol = sols.get_best_seen_candidate().get_x_dvec();
  std::cout << "runStatus: " << sols.run_status() << std::endl;
  std::cout << "testBounds1: "  << best_sol.transpose() << std::endl;
}

void testBounds2(int nb_dims)
{
  // Compute initial guess, step size and boundaries
  std::vector<double> init_params(nb_dims);
  std::vector<double> lower_bounds(nb_dims), upper_bounds(nb_dims);
  for (int dim = 0; dim < nb_dims; dim++) {
    init_params[dim] = (dim_min + dim_max) / 2;
    lower_bounds[dim] = dim_min;
    upper_bounds[dim] = dim_max;
  }
  double sigma = 0.1;
  // GenoType + PhenoType values (limits + linear scaling)
  GenoPheno<pwqBoundStrategy,linScalingStrategy> gp(lower_bounds.data(),
                                                    upper_bounds.data(),
                                                    nb_dims);
  // Setup parameters
  CMAParameters<GenoPheno<pwqBoundStrategy,linScalingStrategy>> cma_params
    (nb_dims,init_params.data(), sigma,-1,0,gp);
  cma_params.set_quiet(true);
  cma_params.set_mt_feval(false);
  //cma_params.set_str_algo("abipop");
  //cma_params.set_noisy();
  //cma_params.set_elitism(1);
  cma_params.set_restarts(0);
  cma_params.set_max_iter(50);
  cma_params.set_max_fevals(100);
  CMASolutions sols = cmaes<GenoPheno<pwqBoundStrategy,linScalingStrategy>>(easy_func,cma_params);
  Eigen::VectorXd best_sol = gp.pheno(sols.get_best_seen_candidate().get_x_dvec());
  std::cout << "runStatus: " << sols.run_status() << std::endl;
  std::cout << "testBounds2: "  << best_sol.transpose() << std::endl;
  std::cout << "\tAlternative print: ";
  sols.print(std::cout,0,gp);
  std::cout << std::endl;
}

int main()
{
  for (int i = 1; i < 3; i++) {
    std::cout << "==============" << std::endl
              << "nb_dims: " << i << std::endl;
    testBounds1(i);
    testBounds2(i);
  }
}
