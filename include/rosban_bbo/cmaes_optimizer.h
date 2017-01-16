#include "rosban_bbo/optimizer.h"

namespace rosban_bbo
{

class CMAESOptimizer : public Optimizer {
public:
  CMAESOptimizer();

  virtual Eigen::VectorXd train(RewardFunc & reward,
                                const Eigen::VectorXd & initial_candidate,
                                std::default_random_engine * engine);

  virtual void setMaxCalls(int max_calls) override;

  virtual std::string class_name() const;
  virtual void to_xml(std::ostream &out) const override;
  virtual void from_xml(TiXmlNode *node) override;

private:
  /// Should CMAES write on cout
  bool quiet;
  /// Maximal number of iterations for the CMAES process
  /// unused if <= 0
  int nb_iterations;
  /// Maximal number of function evaluations
  /// (CMAES can actually use a few additional evaluation to end one of its iterations)
  int nb_evaluations;
  /// Number of restarts allowed
  /// Note: parameters such as nb_evaluations and nb_iterations concerns a single run,
  ///       therefore, if you set nb_restart at 5, total number of evaluations will be
  ///       5 * nb_evaluations
  int nb_restarts;
  /// The size of the population used for a single iteration of the CMAES algorithm
  /// Unused if <= 0
  int population_size;
  /// If the highest diff in the last 'max_history' entries is below ftolerance,
  /// then the optimization stops
  /// negative values lead to default cmaes behavior
  double ftolerance;
  /// Size of the history window for ftolerance
  /// negative values lead to default cmaes behavior
  int max_history;

  /// Elitism reinject the best ever seen solution
  /// 0: No elitism
  /// 1: Reinject the best ever seen solution
  /// 2: Reinject x0 as long as it is not improved upon
  /// 3: Initial elitism on restart: restart if final solution is not the best
  ///    ever seen solution and reinjects the best solution until population
  ///    has better fitness
  int elitism;
};

}
