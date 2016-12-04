#include "rosban_bbo/optimizer.h"

namespace rosban_bbo
{

class CMAESOptimizer : public Optimizer {
public:
  CMAESOptimizer();

  virtual Eigen::VectorXd train(RewardFunc & reward,
                                std::default_random_engine * engine);

  virtual std::string class_name() const;
  virtual void to_xml(std::ostream &out) const override;
  virtual void from_xml(TiXmlNode *node) override;

private:
  /// Should CMAES write on cout
  bool quiet;
  /// Nb different set of parameters tested
  int nb_iterations;
  /// Nb function evaluations
  int nb_evaluations;
  /// Number of restarts allowed
  int nb_restarts;
  /// If the highest diff in the last 'max_history' entries is below ftolerance,
  /// then the optimization stops
  /// negative values lead to default cmaes behavior
  double ftolerance;
  /// Size of the history window for ftolerance
  /// negative values lead to default cmaes behavior
  int max_history;
};

}
