#include "rosban_bbo/optimizer.h"

#include "rhoban_utils/serialization/factory.h"

namespace rosban_bbo
{

class OptimizerFactory : public rhoban_utils::Factory<Optimizer> {
public:
  OptimizerFactory();
};

}
