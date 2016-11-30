#include "rosban_bbo/optimizer.h"

namespace rosban_bbo
{

class MonteCarloOptimizer : public Optimizer {
public:
  MonteCarloOptimizer();

  virtual Eigen::VectorXd train(RewardFunc & reward,
                                std::default_random_engine * engine);

  virtual std::string class_name() const;
  virtual void to_xml(std::ostream &out) const override;
  virtual void from_xml(TiXmlNode *node) override;

private:
  /// Nb different set of parameters tested
  int nb_trials;
};

}
