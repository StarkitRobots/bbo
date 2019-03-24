#pragma once

#include "rhoban_bbo/optimizer.h"

namespace rhoban_bbo
{
/// Note: While HOO can support topological spaces, we restrict ourselves to
/// hyperrectangles for the sake of simplicity
class HOONode
{
public:
  HOONode(const Eigen::MatrixXd& space);
  HOONode(HOONode* parent, const Eigen::MatrixXd& space);
  ~HOONode();

  /// Returns a greedy action based on samples collected
  Eigen::VectorXd getGreedyAction() const;

  /// If the node has never been visited:
  /// - Sample randomly in node_space
  /// - Propagate reward to parents
  /// - Create child for current nodes
  /// If the node has already been visited:
  /// - Call sampleNextAction on the child with the best b-value
  void sampleNextAction(Optimizer::RewardFunc rf, std::default_random_engine* engine);

  /// rho and nu1 are parameters of the HOO algorithm
  void updateBValues(int n, double rho, double nu1, int depth = 0);

private:
  /// Propagate the sampling of a new value to all parents
  /// Updates both, nb_visits and avg_reward
  void propagateSample(double reward);

  /// Link toward the parent node, nullptr for the root
  HOONode* parent;
  /// A node with a non-null number of visits should always have a lower child
  HOONode* lower_child;
  /// A node with a non-null number of visits should always have a upper child
  HOONode* upper_child;
  /// The dimension along which the space is separated
  int split_dim;
  /// The value along at which the space is seperated
  double split_value;
  /// The space represented by the node
  Eigen::MatrixXd node_space;
  /// Average reward for all the runs which have been performed
  double avg_reward;
  /// Number of times the exploration went through this node
  int nb_visits;
  /// The value used to choose next action
  double b_value;
};

/// This class implements the Hierchical Optimistic Optimizer, HOO for short.
/// This algorithm was proposed by Bubeck, Munos, Stoltz and Szepesvári in 2009
/// in the article "Online optimization in chi-armed bandits"
///
/// This class has two meta-parameters: rho and nu1
/// In section 4 of the article,  an example of those parameters is presented,
/// with the following parameters
/// - D: dimensionality of the problem
/// - alpha: a parameter related to dissimilarity in input f(x,y) = ||x-y||^alpha
/// - rho: 2^{-alpha/D}
/// - nu1: (sqrt(D)/2)^alpha
class HOO : public Optimizer
{
public:
  HOO();
  HOO(const HOO& other);

  virtual Eigen::VectorXd train(RewardFunc& reward, const Eigen::VectorXd& initial_candidate,
                                std::default_random_engine* engine) override;

  virtual void setMaxCalls(int max_calls) override;

  virtual Json::Value toJson() const override;
  virtual void fromJson(const Json::Value& v, const std::string& dir_name) override;
  virtual std::string getClassName() const override;

  virtual std::unique_ptr<Optimizer> clone() const override;

private:
  /// Number of calls allowed
  int max_calls;

  /// Meta-parameter of the algorithm, see class description
  double rho;

  /// Meta-parameter of the algorithm, see class description
  double nu1;
};

}  // namespace rhoban_bbo
