cmake_minimum_required(VERSION 3.12)
project(xgboost_predictor_cpp)

set(CMAKE_CXX_STANDARD 11)

include_directories(${PROJECT_SOURCE_DIR}/xgboost_predictor)
add_subdirectory(xgboost_predictor)

add_executable(xgboost_predictor_cpp main.cpp)

target_link_libraries(xgboost_predictor_cpp xgboost_predictor)