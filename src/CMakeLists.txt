FIND_PACKAGE( OpenCV REQUIRED )
find_package(PCL 1.7 REQUIRED)

INCLUDE_DIRECTORIES(${G2O_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIR} ${CSPARSE_INCLUDE_DIR} ${PCL_INCLUDE_DIRS})
LINK_LIBRARIES(${PCL_LIBRARY_DIRS})

# Build Library
# Graphic end

ADD_LIBRARY(graphicend
  ImageReader.cpp GraphicEnd.cpp FeatureGrabber.cpp FeatureManager.cpp ParameterReader.cpp FeatureManager2.cpp fabmap.cpp PCL_End.cpp)

TARGET_LINK_LIBRARIES( graphicend
  ${OpenCV_LIBS} yaml-cpp g2o_core ${MOOS_LIBS} pthread ${PCL_LIBRARIES})

# SLAM back end
ADD_LIBRARY(slamend
  Slam_end.cpp
  )
TARGET_LINK_LIBRARIES( slamend
  ${OpenCV_LIBS} yaml-cpp g2o_core
  g2o_types_slam2d g2o_solver_csparse g2o_stuff ${CSPARSE_LIBRARY}
  g2o_csparse_extension)


# Build test executable
ADD_EXECUTABLE( run_graphic run_graphic_end.cpp )
ADD_EXECUTABLE( run_graphic_and_slam run_graphic_and_slam.cpp)
ADD_EXECUTABLE( run_fabmap run_fab.cpp )


TARGET_LINK_LIBRARIES( run_graphic graphicend )
TARGET_LINK_LIBRARIES( run_graphic_and_slam graphicend slamend )
TARGET_LINK_LIBRARIES( run_fabmap graphicend slamend)
