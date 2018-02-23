#include "rhoban_bbo/optimizer.h"

namespace rhoban_bbo
{

class MonteCarloOptimizer : public Optimizer {
public:
  MonteCarloOptimizer();
  MonteCarloOptimizer(const MonteCarloOptimizer & other);

  virtual Eigen::VectorXd train(RewardFunc & reward,
                                const Eigen::VectorXd & initial_candidate,
                                std::default_random_engine * engine);

  virtual void setMaxCalls(int max_calls) override;

  virtual std::string getClassName() const override;
  virtual Json::Value toJson() const override;
  virtual void fromJson(const Json::Value & v, const std::string & dir_name) override;

  virtual std::unique_ptr<Optimizer> clone() const override;

private:
  /// Nb different set of parameters tested
  int nb_trials;
};

}
