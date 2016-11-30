#include "rosban_bbo/optimizer.h"

#include "rosban_utils/factory.h"

namespace rosban_bbo
{

class OptimizerFactory : public rosban_utils::Factory<Optimizer> {
public:
  OptimizerFactory();
};

}
