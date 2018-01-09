cmake_minimum_required(VERSION 2.8.3)
project(rosban_bbo)

find_package(catkin REQUIRED COMPONENTS
  rhoban_utils
  libcmaes
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -std=c++11 -fopenmp")

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES rosban_bbo
  CATKIN_DEPENDS rhoban_utils libcmaes
)

###########
## Build ##
###########

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  )

# Directories containing sources files
set(DIRECTORIES
  src/rosban_bbo
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
add_library(rosban_bbo ${ALL_SOURCES} )
target_link_libraries(rosban_bbo ${catkin_LIBRARIES})

add_executable(test_cmaes_limits src/test_cmaes_limits.cpp)
target_link_libraries(test_cmaes_limits ${catkin_LIBRARIES})
