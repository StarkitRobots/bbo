#include "starkit_bbo/optimizer.h"

#include <memory>

namespace starkit_bbo
{
class CompositeOptimizer : public Optimizer
{
public:
  CompositeOptimizer();
  CompositeOptimizer(const CompositeOptimizer& other);

  virtual Eigen::VectorXd train(RewardFunc& reward, const Eigen::VectorXd& initial_candidate,
                                std::default_random_engine* engine);

  virtual std::string getClassName() const override;
  virtual Json::Value toJson() const override;
  virtual void fromJson(const Json::Value& v, const std::string& dir_name) override;

  virtual void setMaxCalls(int max_calls) override;

  virtual void setLimits(const Eigen::MatrixXd& new_limits) override;

  virtual std::unique_ptr<Optimizer> clone() const override;

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

}  // namespace starkit_bbo
