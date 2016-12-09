#include "rosban_bbo/optimizer.h"

namespace rosban_bbo
{

class CrossEntropy : public Optimizer
{
public:
  CrossEntropy();

  virtual Eigen::VectorXd train(RewardFunc & reward,
                                const Eigen::VectorXd & initial_candidate,
                                std::default_random_engine * engine);

  /// Initial covariance matrix is built upon the parameters limits
  Eigen::MatrixXd getInitialCovariance();

  virtual std::string class_name() const;
  virtual void to_xml(std::ostream &out) const override;
  virtual void from_xml(TiXmlNode *node) override;

  typedef std::pair<Eigen::VectorXd, double> ScoredCandidate;

private:
  /// How many generations are used
  int nb_generations;

  /// How many samples are generated at each iteration
  int population_size;

  /// How many samples are considered for the next generation
  int best_set_size;

};

}
