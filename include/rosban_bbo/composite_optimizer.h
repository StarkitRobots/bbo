#include "rosban_bbo/optimizer.h"

#include <memory>

namespace rosban_bbo
{

class CompositeOptimizer : public Optimizer {
public:
  CompositeOptimizer();

  virtual Eigen::VectorXd train(RewardFunc & reward,
                                const Eigen::VectorXd & initial_candidate,
                                std::default_random_engine * engine);

  virtual std::string class_name() const;
  virtual void to_xml(std::ostream &out) const override;
  virtual void from_xml(TiXmlNode *node) override;

  virtual void setMaxCalls(int max_calls) override;

  virtual void setLimits(const Eigen::MatrixXd & new_limits) override;

private:
  /// The list of available optimizers
  std::vector<std::unique_ptr<Optimizer>> optimizers;
  /// Names of the optimizers (for debug purpose)
  std::vector<std::string> names;
  /// If weights are specified, the optimizer is sampled randomly
  std::vector<double> weights;

  /// Number of trials used for validation
  int validation_trials;

  /// Verbosity of output
  int debug_level;
};

}
