#include "rosban_bbo/optimizer_factory.h"

#include "rhoban_random/tools.h"

#include <iostream>
#include <cstdlib>


int main(int argc, char ** argv)
{
  std::string optimizer_path;
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <optimizer_path>" << std::endl;
    exit(EXIT_FAILURE);
  }
  optimizer_path = argv[1];
  
  // A simple function to optimize
  double expected_optimum = 1;
  rosban_bbo::Optimizer::RewardFunc reward_function = [expected_optimum]
    (const Eigen::VectorXd & params,std::default_random_engine * engine) -> double
    {
      std::normal_distribution<double> noise_distrib(0,1);
      double x = params(0);
      double noise = noise_distrib(*engine);
      return noise - (x-expected_optimum) * (x-expected_optimum);
    };

  // Initializing guess and parameter space
  Eigen::Matrix<double,1,2> param_space;
  param_space << -5, 5;
  
  // Loading an optimizer from a Json file
  rosban_bbo::OptimizerFactory f;
  std::unique_ptr<rosban_bbo::Optimizer> optimizer = f.buildFromJsonFile(optimizer_path);
  optimizer->setLimits(param_space);

  // Optimizing
  std::default_random_engine * engine = rhoban_random::newRandomEngine();
  Eigen::VectorXd best_params = optimizer->train(reward_function, engine);

  std::cout << "Best params: " << best_params.transpose() << std::endl;
  
}
