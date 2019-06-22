#include "starkit_bbo/optimizer.h"

namespace starkit_bbo
{
class SimulatedAnnealing : public Optimizer
{
public:
  SimulatedAnnealing();
  SimulatedAnnealing(const SimulatedAnnealing& other);

  virtual Eigen::VectorXd train(RewardFunc& reward, const Eigen::VectorXd& initial_candidate,
                                std::default_random_engine* engine);

  /// Return the temperature for the given trial
  double getTemperature(int trial) const;

  /// Choose a neighbor of 'candidate' randomly according to temperature
  Eigen::VectorXd sampleNeighbor(const Eigen::VectorXd& state, double temperature, std::default_random_engine* engine);

  virtual void setMaxCalls(int max_calls) override;

  virtual std::string getClassName() const override;
  virtual Json::Value toJson() const override;
  virtual void fromJson(const Json::Value& v, const std::string& dir_name) override;

  virtual std::unique_ptr<Optimizer> clone() const override;

private:
  /// Initial temperature
  /// amplitude of this parameter should approximately match reward amplitude
  double initial_temperature;

  /// Nb different set of parameters tested
  int nb_trials;

  /// Is the training process spamming output ?
  bool verbose;
};

}  // namespace starkit_bbo
