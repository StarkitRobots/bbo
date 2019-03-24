#include "rhoban_bbo/optimizer_factory.h"

#include "rhoban_utils/serialization/factory.h"
#include "rhoban_random/tools.h"

#include <iostream>
#include <cstdlib>

class OptimizationTask : public rhoban_utils::JsonSerializable
{
public:
  OptimizationTask()
  {
  }

  virtual double sampleReward(const Eigen::VectorXd& parameters, std::default_random_engine* engine) const = 0;

  virtual Json::Value toJson() const override
  {
    Json::Value v;
    v["input_limits"] = rhoban_utils::matrix2Json(input_limits);
    return v;
  }

  virtual void fromJson(const Json::Value& v, const std::string& dir_name)
  {
    (void)dir_name;
    rhoban_utils::tryRead(v, "input_limits", &input_limits);
  }

  /// The limits allowed for input
  Eigen::MatrixXd input_limits;
};

class MSEOptimization : public OptimizationTask
{
public:
  double sampleReward(const Eigen::VectorXd& parameters, std::default_random_engine* engine) const
  {
    if (parameters.rows() != input_limits.rows() || parameters.rows() != optimum.rows())
    {
      throw std::logic_error("MSEOptimization::sampleReward: Size mismatch");
    }
    std::normal_distribution<double> noise_distrib(0, 1);
    double squared_error = (parameters - optimum).transpose() * (parameters - optimum);
    double noise = noise_distrib(*engine);
    return noise - squared_error;
  }

  Json::Value toJson() const override
  {
    Json::Value v = OptimizationTask::toJson();
    v["optimum"] = rhoban_utils::vector2Json(optimum);
    v["stddev"] = stddev;
    return v;
  }

  void fromJson(const Json::Value& v, const std::string& dir_name) override
  {
    (void)dir_name;
    OptimizationTask::fromJson(v, dir_name);
    rhoban_utils::tryRead(v, "optimum", &optimum);
    rhoban_utils::tryRead(v, "stddev", &stddev);
  }

  std::string getClassName() const override
  {
    return "MSEOptimization";
  }

  Eigen::VectorXd optimum;
  double stddev;
};

/// A sum of sinus according different dimensions
/// result(x) = sum_{i\in {1,..., D}} sin(2*pi*x_i / periods(i) + offsets(i))
class SinSum : public OptimizationTask
{
public:
  double sampleReward(const Eigen::VectorXd& parameters, std::default_random_engine* engine) const
  {
    if (parameters.rows() != input_limits.rows() || parameters.rows() != periods.rows() ||
        parameters.rows() != offsets.rows())
    {
      throw std::logic_error("SinSum::sampleReward: Size mismatch");
    }
    // Computing basic result
    Eigen::VectorXd tmp = 2 * M_PI * parameters.cwiseProduct(periods.cwiseInverse()) + offsets;
    double deterministic_part = 0;
    for (int dim = 0; dim < tmp.rows(); dim++)
    {
      deterministic_part += sin(tmp(dim));
    }
    // Adding noise
    std::normal_distribution<double> noise_distrib(0, 1);
    double noise = noise_distrib(*engine);
    return noise + deterministic_part;
  }

  Json::Value toJson() const override
  {
    Json::Value v = OptimizationTask::toJson();
    v["periods"] = rhoban_utils::vector2Json(periods);
    v["offsets"] = rhoban_utils::vector2Json(offsets);
    v["stddev"] = stddev;
    return v;
  }

  void fromJson(const Json::Value& v, const std::string& dir_name) override
  {
    (void)dir_name;
    OptimizationTask::fromJson(v, dir_name);
    rhoban_utils::tryRead(v, "periods", &periods);
    rhoban_utils::tryRead(v, "offsets", &offsets);
    rhoban_utils::tryRead(v, "stddev", &stddev);
  }

  std::string getClassName() const override
  {
    return "SinSum";
  }

  Eigen::VectorXd periods;
  Eigen::VectorXd offsets;
  double stddev;
};

/// A sum of sinus according different dimensions
/// result(x) = prod_{i\in {1,..., D}} sin(2*pi*x_i / periods(i) + offsets(i))
class SinProd : public OptimizationTask
{
public:
  double sampleReward(const Eigen::VectorXd& parameters, std::default_random_engine* engine) const
  {
    if (parameters.rows() != input_limits.rows() || parameters.rows() != periods.rows() ||
        parameters.rows() != offsets.rows())
    {
      throw std::logic_error("SinProd::sampleReward: Size mismatch");
    }
    // Computing basic result
    Eigen::VectorXd tmp = 2 * M_PI * parameters.cwiseProduct(periods.cwiseInverse()) + offsets;
    double deterministic_part = 1;
    for (int dim = 0; dim < tmp.rows(); dim++)
    {
      deterministic_part *= sin(tmp(dim));
    }
    // Adding noise
    std::normal_distribution<double> noise_distrib(0, 1);
    double noise = noise_distrib(*engine);
    return noise + deterministic_part;
  }

  Json::Value toJson() const override
  {
    Json::Value v = OptimizationTask::toJson();
    v["periods"] = rhoban_utils::vector2Json(periods);
    v["offsets"] = rhoban_utils::vector2Json(offsets);
    v["stddev"] = stddev;
    return v;
  }

  void fromJson(const Json::Value& v, const std::string& dir_name) override
  {
    (void)dir_name;
    OptimizationTask::fromJson(v, dir_name);
    rhoban_utils::tryRead(v, "periods", &periods);
    rhoban_utils::tryRead(v, "offsets", &offsets);
    rhoban_utils::tryRead(v, "stddev", &stddev);
  }

  std::string getClassName() const override
  {
    return "SinProd";
  }

  Eigen::VectorXd periods;
  Eigen::VectorXd offsets;
  double stddev;
};

class TaskFactory : public rhoban_utils::Factory<OptimizationTask>
{
public:
  TaskFactory()
  {
    registerBuilder("MSEOptimization", []() { return std::unique_ptr<OptimizationTask>(new MSEOptimization); });
    registerBuilder("SinSum", []() { return std::unique_ptr<OptimizationTask>(new SinSum); });
    registerBuilder("SinProd", []() { return std::unique_ptr<OptimizationTask>(new SinProd); });
  }
};

int main(int argc, char** argv)
{
  if (argc < 3)
  {
    std::cerr << "Usage: " << argv[0] << " <task_path> <optimizer_path>" << std::endl;
    exit(EXIT_FAILURE);
  }
  std::string optimizer_path, task_path;
  task_path = argv[1];
  optimizer_path = argv[2];

  // Loading task from Json
  TaskFactory tf;
  std::unique_ptr<OptimizationTask> task = tf.buildFromJsonFile(task_path);

  // Loading optimizer from Json
  rhoban_bbo::OptimizerFactory of;
  std::unique_ptr<rhoban_bbo::Optimizer> optimizer = of.buildFromJsonFile(optimizer_path);
  optimizer->setLimits(task->input_limits);

  // Optimizing
  std::default_random_engine* engine = rhoban_random::newRandomEngine();
  rhoban_bbo::Optimizer::RewardFunc reward_function = [&task](const Eigen::VectorXd& parameters,
                                                              std::default_random_engine* engine) {
    return task->sampleReward(parameters, engine);
  };
  Eigen::VectorXd best_params = optimizer->train(reward_function, engine);

  std::cout << "Best params: " << best_params.transpose() << std::endl;

  int nb_validations = 1000;
  double total_reward = 0;
  for (int i = 0; i < nb_validations; i++)
  {
    total_reward += task->sampleReward(best_params, engine);
  }
  double avg_reward = total_reward / nb_validations;
  std::cout << "\t-> avg reward: " << avg_reward << std::endl;

  delete (engine);
}
