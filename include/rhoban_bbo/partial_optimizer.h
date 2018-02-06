#include "rhoban_bbo/optimizer.h"

#include <memory>

namespace rhoban_bbo
{

/// This class acts allows to ignore easily some of the dimensions of the input 
/// to reduce the size search space. When a dimension is ignored, the 'guess'
/// value is used.
/// PartialOptimizer is not supposed to build a proper optimization in a
/// single-step. However, it can be used to refine solutions in iterated
/// optimizations.
class PartialOptimizer : public Optimizer {
public:
  PartialOptimizer();

  virtual Eigen::VectorXd train(RewardFunc & reward,
                                const Eigen::VectorXd & initial_candidate,
                                std::default_random_engine * engine);

  virtual void setMaxCalls(int max_calls) override;

  virtual std::string getClassName() const override;
  virtual Json::Value toJson() const override;
  virtual void fromJson(const Json::Value & v, const std::string & dir_name) override;

private:
  /// The list of available optimizers
  std::unique_ptr<Optimizer> optimizer;

  /// The ratio of dimensions used for optimization (in [0,1])
  double ratio_used;
};

}
