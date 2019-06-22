#include "starkit_bbo/optimizer.h"

#include "starkit_utils/serialization/factory.h"

namespace starkit_bbo
{
class OptimizerFactory : public starkit_utils::Factory<Optimizer>
{
public:
  OptimizerFactory();
};

}  // namespace starkit_bbo
