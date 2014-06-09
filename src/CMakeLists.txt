FIND_PACKAGE( OpenCV REQUIRED )
find_package(PCL 1.7 REQUIRED)

INCLUDE_DIRECTORIES(${G2O_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIR} ${CSPARSE_INCLUDE_DIR} ${PCL_INCLUDE_DIRS})
LINK_LIBRARIES(${PCL_LIBRARY_DIRS})

# Build Library
# Graphic end

ADD_LIBRARY(SLAM3d
  GraphicEnd.cpp ParameterReader.cpp
)
TARGET_LINK_LIBRARIES( SLAM3d
  ${OpenCV_LIBS} g2o_core pthread ${PCL_LIBRARIES} g2o_types_slam3d g2o_solver_csparse g2o_stuff g2o_csparse_extension ${CSPARSE_LIBRARY} yaml-cpp
  )

ADD_EXECUTABLE( run_SLAM run_SLAM.cpp )
TARGET_LINK_LIBRARIES( run_SLAM SLAM3d )

ADD_EXECUTABLE( saveOutput saveOutput.cpp )
TARGET_LINK_LIBRARIES( saveOutput SLAM3d )

