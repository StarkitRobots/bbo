#include "rosban_bbo/optimizer.h"

namespace rosban_bbo
{

class SimulatedAnnealing : public Optimizer {
public:
  SimulatedAnnealing();

  virtual Eigen::VectorXd train(RewardFunc & reward,
                                const Eigen::VectorXd & initial_candidate,
                                std::default_random_engine * engine);

  /// Return the temperature for the given trial
  double getTemperature(int trial) const;

  /// Choose a neighbor of 'candidate' randomly according to temperature
  Eigen::VectorXd sampleNeighbor(const Eigen::VectorXd & state,
                                 double temperature,
                                 std::default_random_engine * engine);

  virtual void setMaxCalls(int max_calls) override;

  virtual std::string class_name() const;
  virtual void to_xml(std::ostream &out) const override;
  virtual void from_xml(TiXmlNode *node) override;

private:
  /// Initial temperature
  /// amplitude of this parameter should approximately match reward amplitude
  double initial_temperature;

  /// Nb different set of parameters tested
  int nb_trials;

  /// Is the training process spamming output ?
  bool verbose;
};

}
