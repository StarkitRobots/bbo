cmake_minimum_required(VERSION 2.8.3)
project(starkit_bbo)

find_package(catkin REQUIRED COMPONENTS
  starkit_utils
  starkit_random
  libcmaes
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -std=c++17 -fopenmp")

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES starkit_bbo
  CATKIN_DEPENDS starkit_utils starkit_random libcmaes
  )

option(ROSBAN_BBO_BUILD_EXAMPLES OFF)

###########
## Build ##
###########

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  )

# Directories containing sources files
set(DIRECTORIES
  src/starkit_bbo
  )

# Build ALL_SOURCES
set (SOURCES)
set (ALL_SOURCES)
set (PREFIXED_SOURCES)
foreach (DIRECTORY ${DIRECTORIES})
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/${DIRECTORY}")
    include (${DIRECTORY}/sources.cmake)

    foreach (SOURCE ${SOURCES})
        set (PREFIXED_SOURCES ${PREFIXED_SOURCES} ${DIRECTORY}/${SOURCE})
    endforeach (SOURCE)

    set (ALL_SOURCES ${ALL_SOURCES} ${PREFIXED_SOURCES})
endforeach (DIRECTORY)

# Declare the library
add_library(starkit_bbo ${ALL_SOURCES} )
target_link_libraries(starkit_bbo ${catkin_LIBRARIES})

if (ROSBAN_BBO_BUILD_EXAMPLES)
  add_executable(bbo_example examples/bbo_example.cpp)
  target_link_libraries(bbo_example starkit_bbo ${catkin_LIBRARIES})

  add_executable(test_cmaes_limits examples/test_cmaes_limits.cpp)
  target_link_libraries(test_cmaes_limits starkit_bbo ${catkin_LIBRARIES})
endif()
