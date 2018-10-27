cmake_minimum_required(VERSION 3.0)

# enable C++11 and warnings
set(CMAKE_CXX_FLAGS "--std=gnu++11 ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-Wall ${CMAKE_CXX_FLAGS}")

# where is boost?
include_directories(/opt/boost)
link_directories(/opt/boost/stage/lib)

include_directories(.)

set(AppSources
  main.cpp
)

set(AppLibraries
)

add_executable(bp ${AppSources})
target_link_libraries(bp ${AppLibraries})
