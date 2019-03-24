#include "rhoban_bbo/optimizer.h"

#include "rhoban_utils/serialization/factory.h"

namespace rhoban_bbo
{
class OptimizerFactory : public rhoban_utils::Factory<Optimizer>
{
public:
  OptimizerFactory();
};

}  // namespace rhoban_bbo
