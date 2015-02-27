#------------------------------------------------------------------------------
# Tell cmake where to find the project cmake scripts
#------------------------------------------------------------------------------
PROJECT(cvpr_2012)
cmake_minimum_required(VERSION 2.8)

#-- Where to look first for cmake modules
SET(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/modules)

#-- Type of build
SET(CMAKE_BUILD_TYPE Release)

FIND_PACKAGE(OpenCV REQUIRED)
MESSAGE(STATUS "OPENCV_LIBRARIES=${OpenCV_LIBS}")
MESSAGE(STATUS "OPENCV_INCLUDE_DIRS=${OpenCV_INCLUDE_DIRS}")

FIND_PACKAGE(Boost REQUIRED COMPONENTS serialization system filesystem thread REQUIRED)
MESSAGE(STATUS "BOOST_LIBRARIES=${Boost_LIBRARIES}")
MESSAGE(STATUS "BOOST_INCLUDE_DIRS=${Boost_INCLUDE_DIR}")

#-- Find include files
INCLUDE_DIRECTORIES(
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
)

#-- Executable
ADD_EXECUTABLE(test_cvpr_2012
    ${CMAKE_SOURCE_DIR}/include/trace.hpp
    ${CMAKE_SOURCE_DIR}/include/forest.hpp
    ${CMAKE_SOURCE_DIR}/include/mean_shift.hpp
    ${CMAKE_SOURCE_DIR}/include/opencv_serialization.hpp
    ${CMAKE_SOURCE_DIR}/include/face_forest.hpp
    ${CMAKE_SOURCE_DIR}/src/face_forest.cpp
    ${CMAKE_SOURCE_DIR}/include/head_pose_sample.hpp
    ${CMAKE_SOURCE_DIR}/src/head_pose_sample.cpp
    ${CMAKE_SOURCE_DIR}/include/timing.hpp
    ${CMAKE_SOURCE_DIR}/include/tree_node.hpp
    ${CMAKE_SOURCE_DIR}/include/split_gen.hpp
    ${CMAKE_SOURCE_DIR}/include/tree.hpp
    ${CMAKE_SOURCE_DIR}/include/multi_part_sample.hpp
    ${CMAKE_SOURCE_DIR}/src/multi_part_sample.cpp
    ${CMAKE_SOURCE_DIR}/include/thread_pool.hpp
    ${CMAKE_SOURCE_DIR}/include/feature_channel_factory.hpp
    ${CMAKE_SOURCE_DIR}/include/image_sample.hpp
    ${CMAKE_SOURCE_DIR}/src/image_sample.cpp
    ${CMAKE_SOURCE_DIR}/include/face_utils.hpp
    ${CMAKE_SOURCE_DIR}/src/face_utils.cpp
    ${CMAKE_SOURCE_DIR}/include/Viewer.hpp
    ${CMAKE_SOURCE_DIR}/src/Viewer.cpp
    
    ${CMAKE_SOURCE_DIR}/src/test_cvpr_2012.cpp
    #${CMAKE_SOURCE_DIR}/src/demo.cpp
    #${CMAKE_SOURCE_DIR}/src/eval_ffd.cpp
    #${CMAKE_SOURCE_DIR}/src/train_ffd.cpp
    #${CMAKE_SOURCE_DIR}/src/train_headpose.cpp
)

#-- Link the executable to the libraries
TARGET_LINK_LIBRARIES(test_cvpr_2012
    ${OpenCV_LIBS}   
    jpeg
    tiff
    png
    ${Boost_LIBRARIES}
)
