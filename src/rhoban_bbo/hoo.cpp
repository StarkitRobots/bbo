#include "rhoban_bbo/hoo.h"

#include "rhoban_random/tools.h"

namespace rhoban_bbo
{

HOONode::HOONode(const Eigen::MatrixXd & space)
  : HOONode(nullptr, space)
{}

HOONode::HOONode(HOONode * parent, const Eigen::MatrixXd & space)
  : parent(parent), lower_child(nullptr), upper_child(nullptr),
    split_dim(-1), split_value(0), node_space(space),
    avg_reward(0), nb_visits(0),
    b_value(std::numeric_limits<double>::max())
{}

HOONode::~HOONode() {
  if (lower_child) delete(lower_child);
  if (upper_child) delete(upper_child);
}

Eigen::VectorXd HOONode::getGreedyAction() const {
  if (nb_visits > 0) {
    double diff = lower_child->avg_reward - upper_child->avg_reward;
    if (diff > 0) {
      return lower_child->getGreedyAction();
    } else {
      return upper_child->getGreedyAction();
    }
  }
  return (node_space.col(0) + node_space.col(1)) / 2;
}


void HOONode::sampleNextAction(Optimizer::RewardFunc rf,
                               std::default_random_engine * engine) {
  if (nb_visits > 0) {
    double diff = lower_child->b_value - upper_child->b_value;
    // 'Hacking' diff to make random choice if there is no difference
    if (diff == 0) {
      std::uniform_real_distribution<double> distrib(-1000,1000);
      diff += distrib(*engine);
    }
    if (diff > 0) {
      lower_child->sampleNextAction(rf, engine);
    } else {
      upper_child->sampleNextAction(rf, engine);
    }
    return;
  }
  // Nb visits == 0 -> first visit of the node
  // Chooses input randomly and sample reward
  Eigen::VectorXd input = rhoban_random::getUniformSample(node_space, engine);
  double reward = rf(input, engine);
  propagateSample(reward);
  // Chooses split (determinist currently)
  split_dim = 0;
  if (parent) {// Cycling through dimensions
    split_dim = parent->split_dim+1;
    if (split_dim == node_space.rows()) split_dim = 0;
  }
  split_value = (node_space(split_dim,0) + node_space(split_dim,1)) / 2;
  Eigen::MatrixXd lower_space, upper_space;
  lower_space = node_space;
  lower_space(split_dim,1) = split_value;
  upper_space = node_space;
  upper_space(split_dim,0) = split_value;
  // Create two empty children
  lower_child = new HOONode(this, lower_space);
  upper_child = new HOONode(this, upper_space);
}

void HOONode::updateBValues(int n, double rho, double nu1, int depth) {
  if (nb_visits == 0) {
    b_value = std::numeric_limits<double>::max();
  } else {
    // Start by updating childs
    lower_child->updateBValues(n, rho, nu1, depth+1);
    upper_child->updateBValues(n, rho, nu1, depth+1);
    // Now compute value
    double u = avg_reward
      + std::sqrt(2 * std::log(n)/nb_visits)
      + nu1 * std::pow(rho, depth);
    double best_b = std::max(lower_child->b_value, upper_child->b_value);
    b_value = std::min(u, best_b);
  }
}

void HOONode::propagateSample(double reward) {
  avg_reward = (avg_reward * nb_visits + reward) / (nb_visits+1);
  nb_visits++;
  if (parent) parent->propagateSample(reward);
}

HOO::HOO() : max_calls(100), rho(-1), nu1(-1) {
}

HOO::HOO(const HOO & other)
  : max_calls(other.max_calls),
    rho(other.rho), nu1(other.nu1) {
}

void HOO::setMaxCalls(int new_max_calls) {
  max_calls = new_max_calls;
}

Eigen::VectorXd HOO::train(RewardFunc & reward,
                           const Eigen::VectorXd & initial_candidate,
                           std::default_random_engine * engine) {
  // HOO does not uses initial candidates
  (void) initial_candidate;
  // Initial meta-parameters check
  if (nu1 < 0 || rho < 0) {
    throw std::logic_error("HOO::train: invalid meta-parameters");
  }
  // Create initial node
  HOONode root(getLimits());
  // Optimize
  int nb_trials = 0;
  while (nb_trials < max_calls) {
    nb_trials++;
    root.sampleNextAction(reward, engine);
    root.updateBValues(nb_trials, rho, nu1);
  }
  return root.getGreedyAction();
}

void HOO::fromJson(const Json::Value & v, const std::string & dir_name) {
  (void)dir_name;
  rhoban_utils::tryRead(v, "max_calls", &max_calls);
  rhoban_utils::tryRead(v, "rho", &rho);
  rhoban_utils::tryRead(v, "nu1", &nu1);
}

Json::Value HOO::toJson() const {
  Json::Value v;
  v["max_calls"] = max_calls;
  v["rho"] = rho;
  v["nu1"] = nu1;
  return v;
}

std::string HOO::getClassName() const {
  return "HOO";
}

std::unique_ptr<Optimizer> HOO::clone() const {
  return std::unique_ptr<Optimizer>(new HOO(*this));
}



}
